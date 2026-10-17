import sys; sys.path.insert(0,'/home/claude/b4k'); from emb import *
import json, scipy.signal as ss
from transformers import ASTFeatureExtractor, ASTModel
from vggish_reference import VGGishInputProcessor
from sklearn.decomposition import PCA; from sklearn.linear_model import LogisticRegression
cid='92301010'; c=pd.read_csv(PKG+'data/core_3539_metadata.csv',low_memory=False); fn=dict(zip(c.cut_id,c.file_name)); row=c[c.cut_id==cid].iloc[0]
x,fs=read(cid,fn[cid]); print(fs,len(x)/fs,row.species_ko,row.group3,row.g_site,row.fold_all)
R={}
x16=prepare(x,fs,False); nw=int(10.24*16000)
fe=ASTFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593',revision='f826b80d28226b62986cc218e5cec390b1096902')
inp=fe([tile_to(x16,nw).astype(np.float32)],sampling_rate=16000,return_tensors='pt'); R['ast_in']=inp['input_values'][0].numpy()
m=ASTModel.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593',revision='f826b80d28226b62986cc218e5cec390b1096902',attn_implementation='eager').eval()
with torch.no_grad(): o=m(**inp,output_attentions=True)
A=torch.stack(o.attentions)[:,0].mean(1).numpy()  # layers x tok x tok (head-avg)
# attention rollout
I=np.eye(A.shape[1]); roll=I
for l in range(A.shape[0]): a=0.5*A[l]+0.5*I; a/=a.sum(1,keepdims=True); roll=a@roll
R['roll']=roll[0,2:]; R['last']=A[-1][0,2:]; print('tokens',A.shape, m.config.frequency_stride, m.config.time_stride)
R['ast_out']=o.pooler_output[0].numpy()
p=VGGishInputProcessor()(torch.from_numpy(tile_to(x16,16000).astype(np.float32))).numpy(); R['vgg_in']=p[:,0]
x40=resample_to(x-x.mean(),fs,40000); f_,t_,S=ss.stft(x40,40000,nperseg=1024,noverlap=768); R['eff_in']=10*np.log10(np.abs(S)**2+1e-12)
from scipy.fft import dct
xm=resample_to(x-x.mean(),fs,16000); win,hop,nfft=400,160,512; nf=1+(len(xm)-win)//hop; idx=np.arange(win)[None,:]+hop*np.arange(nf)[:,None]
spec=np.abs(np.fft.rfft(xm[idx]*np.hanning(win)[None,:],n=nfft,axis=1))**2; lm=10*np.log10(spec@mel_filterbank(64,nfft,16000,50,8000).T+1e-10)
R['mel_frames']=lm; R['mfcc_frames']=dct(lm,type=2,axis=1,norm='ortho')[:,1:21]
# stored outputs + check
full=pd.read_csv(PKG+'data/core_3539_metadata.csv',low_memory=False); i=int(np.flatnonzero(full.cut_id==cid)[0])
for f in ['ast_tile','vggish','effb3','mfcc16k','meta']: R[f]=np.load(PKG+f'work/features/{f}.npz')['X'][i]
print('ast stored vs now cos',np.dot(R['ast_out'],R['ast_tile'])/np.linalg.norm(R['ast_out'])/np.linalg.norm(R['ast_tile']))
# probe prediction (allcv, clip fold held out)
def probe(Xall,y,tr,te):
    Xtr=Xall[tr]; med=np.nanmedian(Xtr,0); med=np.where(np.isfinite(med),med,0); Xtr=np.where(np.isfinite(Xtr),Xtr,med); Xte=np.where(np.isfinite(Xall[te]),Xall[te],med)
    mu,sd=Xtr.mean(0),Xtr.std(0); sd[sd<1e-12]=1; Xtr=np.clip((Xtr-mu)/sd,-8,8); Xte=np.clip((Xte-mu)/sd,-8,8)
    pc=PCA(min(64,Xtr.shape[1]),random_state=0).fit(Xtr); lr=LogisticRegression(C=1,class_weight='balanced',max_iter=3000).fit(pc.transform(Xtr),y[tr])
    return dict(zip(lr.classes_,lr.predict_proba(pc.transform(Xte))[0]))
y=full.task_group3.to_numpy(); fold=full.fold_all.to_numpy(); tr=fold!=fold[i]; te=np.array([i])
P={}
for f in ['meta','mfcc16k','vggish','effb3','ast_tile']:
    P[f]=probe(np.asarray(np.load(PKG+f'work/features/{f}.npz')['X'],float),y,tr,te); print(f,{k:round(v,2) for k,v in P[f].items()})
np.savez('ex2.npz',**{k:v for k,v in R.items()}); json.dump(P,open('ex2_pred.json','w'))
