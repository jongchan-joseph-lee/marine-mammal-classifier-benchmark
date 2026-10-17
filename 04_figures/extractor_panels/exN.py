import sys; sys.path.insert(0,'/home/claude/b4k'); from emb import *
import json, scipy.signal as ss, torch.nn.functional as Fnn
from transformers import ASTFeatureExtractor, ASTModel
from vggish_reference import VGGish, VGGishInputProcessor
from sklearn.decomposition import PCA; from sklearn.linear_model import LogisticRegression
cid=sys.argv[1]; full=pd.read_csv(PKG+'data/core_3539_metadata.csv',low_memory=False); fn=dict(zip(full.cut_id,full.file_name)); i=int(np.flatnonzero(full.cut_id==cid)[0])
x,fs=read(cid,fn[cid]); print(fs,len(x)/fs,full.species_ko[i],full.group3[i]); R={}
x16=prepare(x,fs,False); nw=int(10.24*16000); REV='f826b80d28226b62986cc218e5cec390b1096902'; NAME='MIT/ast-finetuned-audioset-10-10-0.4593'
fe=ASTFeatureExtractor.from_pretrained(NAME,revision=REV); inp=fe([tile_to(x16,nw).astype(np.float32)],sampling_rate=16000,return_tensors='pt'); R['ast_in']=inp['input_values'][0].numpy()
m=ASTModel.from_pretrained(NAME,revision=REV,attn_implementation='eager').eval()
with torch.no_grad(): o=m(**inp,output_attentions=True)
A=torch.stack(o.attentions)[:,0].mean(1).numpy(); I=np.eye(A.shape[1]); roll=I
for l in range(A.shape[0]): a=0.5*A[l]+0.5*I; a/=a.sum(1,keepdims=True); roll=a@roll
R['roll']=roll[0,2:]
# VGGish activations
vg=VGGish(); vg.load_state_dict(torch.hub.load_state_dict_from_url('https://download.pytorch.org/torchaudio/models/vggish.pt',map_location='cpu',weights_only=True)); vg.eval()
ex=VGGishInputProcessor()(torch.from_numpy(tile_to(x16,16000).astype(np.float32))); R['vgg_in']=ex[:,0].numpy(); h=ex; maps=[]
with torch.no_grad():
    for layer in vg.features_network:
        h=layer(h)
        if isinstance(layer,torch.nn.MaxPool2d): maps.append(h[min(1,len(h)-1)].mean(0).numpy()); print('vgg pool',tuple(h.shape))
for k,mp in enumerate(maps): R[f'vgg_map{k}']=mp
# EfficientNet-B3 (approximate preprocessing, for structure illustration)
import torchvision
eff=torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT).eval()
x40=resample_to(x-x.mean(),fs,40000); x40=tile_to(x40,5*40000)[:5*40000]
f_,t_,S=ss.stft(x40,40000,nperseg=1024,noverlap=768,boundary=None,padded=False); D=10*np.log10(np.abs(S)**2+1e-12); D=np.clip((D-np.percentile(D,5))/(np.percentile(D,99.9)-np.percentile(D,5)),0,1)
img=torch.from_numpy(D[::-1].copy()).float()[None,None]; img=Fnn.interpolate(img,size=(300,800),mode='bilinear',align_corners=False)[0]
R['eff_img']=img[0].numpy(); img3=img.repeat(3,1,1); mean=torch.tensor([0.485,0.456,0.406])[:,None,None]; std=torch.tensor([0.229,0.224,0.225])[:,None,None]; h=((img3-mean)/std)[None]
emaps=[]
with torch.no_grad():
    for k,blk in enumerate(eff.features):
        h=blk(h); emaps.append((tuple(h.shape[1:]),h[0].mean(0).numpy())); print('eff',k,tuple(h.shape[1:]))
    g=h.mean((2,3))[0].numpy()
for k,(shp,mp) in enumerate(emaps): R[f'eff_map{k}']=mp
R['eff_shapes']=np.array([s for s,_ in emaps]); R['eff_vec_now']=g
# MFCC intermediates
from scipy.fft import dct
xm=resample_to(x-x.mean(),fs,16000); win,hop,nfft=400,160,512; nf=1+(len(xm)-win)//hop; idx=np.arange(win)[None,:]+hop*np.arange(nf)[:,None]
spec=np.abs(np.fft.rfft(xm[idx]*np.hanning(win)[None,:],n=nfft,axis=1))**2; fb=mel_filterbank(64,nfft,16000,50,8000)
R['mf_pow']=10*np.log10(spec+1e-10); R['mf_fb']=fb; R['mf_logmel']=10*np.log10(spec@fb.T+1e-10); R['mf_c']=dct(R['mf_logmel'],type=2,axis=1,norm='ortho')[:,1:21]
for f in ['ast_tile','vggish','effb3','mfcc16k','meta']: R[f]=np.load(PKG+f'work/features/{f}.npz')['X'][i]
print('effb3 approx vs stored cos',float(np.dot(g,R['effb3'])/np.linalg.norm(g)/np.linalg.norm(R['effb3'])))
def probe(Xall,y,tr,te):
    Xtr=Xall[tr]; med=np.nanmedian(Xtr,0); med=np.where(np.isfinite(med),med,0); Xtr=np.where(np.isfinite(Xtr),Xtr,med); Xte=np.where(np.isfinite(Xall[te]),Xall[te],med)
    mu,sd=Xtr.mean(0),Xtr.std(0); sd[sd<1e-12]=1; Xtr=np.clip((Xtr-mu)/sd,-8,8); Xte=np.clip((Xte-mu)/sd,-8,8)
    pc=PCA(min(64,Xtr.shape[1]),random_state=0).fit(Xtr); lr=LogisticRegression(C=1,class_weight='balanced',max_iter=3000).fit(pc.transform(Xtr),y[tr])
    return {k:float(v) for k,v in zip(lr.classes_,lr.predict_proba(pc.transform(Xte))[0])}
y=full.task_group3.to_numpy(); fold=full.fold_all.to_numpy(); tr=fold!=fold[i]; P={}
for f in ['meta','mfcc16k','vggish','effb3','ast_tile']:
    P[f]=probe(np.asarray(np.load(PKG+f'work/features/{f}.npz')['X'],float),y,tr,np.array([i])); print(f,{k:round(v,2) for k,v in P[f].items()})
np.savez(f'ex_{cid}.npz',**R); json.dump(P,open(f'ex_{cid}_pred.json','w'))
