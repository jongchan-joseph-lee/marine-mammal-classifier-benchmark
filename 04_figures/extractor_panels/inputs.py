import sys; sys.path.insert(0,'/home/claude/b4k'); sys.path.insert(0,'/home/claude/v4')
from emb import *
from spec import CM
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif"],"font.size":7})
from transformers import ASTFeatureExtractor
from vggish_reference import VGGishInputProcessor
cid,fn='5702601X','5702601X.flac'; x,fs=read(cid,fn); x16=prepare(x,fs,False)
OUT='/home/claude/ms/fig_deck/'; SZ=(2.0,0.95)
def show(A,ext,ylab,name,yt=None,ytl=None,grid=None,note=None):
    fig,a=plt.subplots(figsize=SZ); a.imshow(A,origin='lower',aspect='auto',cmap=CM,extent=ext,interpolation='nearest')
    a.set_yticks(yt or []); a.set_yticklabels(ytl or []); a.set_xticks([]); a.tick_params(length=1.5,pad=1,labelsize=6.5)
    if grid: 
        for g in grid[0]: a.axvline(g,color='white',lw=0.35,alpha=0.8)
        for g in grid[1]: a.axhline(g,color='white',lw=0.35,alpha=0.8)
    if note: a.text(0.98,0.92,note,transform=a.transAxes,ha='right',va='top',fontsize=6.5,color='#0B4EA2',fontweight='bold',bbox=dict(fc='white',ec='none',alpha=0.85,pad=0.8))
    a.set_ylabel(ylab,fontsize=6.5,labelpad=1)
    for s in a.spines.values(): s.set_linewidth(0.4)
    fig.tight_layout(pad=0.1); fig.savefig(OUT+name,dpi=300); plt.close(fig)
def norm(A,lo=20,hi=99.7):
    l,h=np.percentile(A,[lo,hi]); return np.clip((A-l)/(h-l),0,1)
# AST: actual fbank (1024 frames x 128 mel), tiled 10.24 s
fe=ASTFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593',revision='f826b80d28226b62986cc218e5cec390b1096902')
nw=int(10.24*16000); f=fe([tile_to(x16,nw).astype(np.float32)],sampling_rate=16000,return_tensors='np')['input_values'][0]
show(norm(f.T),[0,1024,0,128],'mel 128','ast_in.png',[0,128],['0','8 kHz'],grid=(np.arange(0,1024,10*10),np.arange(0,129,16)),note='10.24 s (tiled)')
# VGGish: first 0.96 s example 96x64
p=VGGishInputProcessor()(torch.from_numpy(tile_to(x16,16000).astype(np.float32))).numpy()
print('vgg examples',p.shape); E=np.concatenate([p[i,0] for i in range(min(2,len(p)))],0)
show(norm(E.T),[0,E.shape[0],0,64],'mel 64','vgg_in.png',[0,64],['0.1','7.5 kHz'],grid=([96],[]),note='0.96 s patch')
# EffNet: 40 kHz STFT nfft 1024 hop 256, 5 s
x40=resample_to(x-x.mean(),fs,40000); dur40=len(x40)/40000
import scipy.signal as ss
fq,tt,S=ss.stft(x40,40000,nperseg=1024,noverlap=1024-256,window='hann'); A=10*np.log10(np.abs(S)**2+1e-12)
show(norm(A,30),[0,dur40,0,20],'linear','eff_in.png',[0,20],['0','20 kHz'],note='40 kHz STFT')
# MFCC coefficients c1..c20 over time
from scipy.fft import dct
xm=resample_to(x-x.mean(),fs,16000); win,hop=400,160; nfft=512; nf=1+(len(xm)-win)//hop
idx=np.arange(win)[None,:]+hop*np.arange(nf)[:,None]; spec=np.abs(np.fft.rfft(xm[idx]*np.hanning(win)[None,:],n=nfft,axis=1))**2
lm=10*np.log10(spec@mel_filterbank(64,nfft,16000,50,8000).T+1e-10); c=dct(lm,type=2,axis=1,norm='ortho')[:,1:21]
cz=(c-c.mean(0))/(c.std(0)+1e-9)
show(np.clip((cz.T+2.5)/5,0,1),[0,nf,1,21],'c1–c20','mfcc_in.png',[1,20],['1','20'],note='25 ms frames')
print('ok')
