import sys, os, json, time, numpy as np, pandas as pd, soundfile as sf, torch
from fractions import Fraction
from scipy.signal import resample_poly
torch.set_num_threads(2)
AUD='/home/claude/audio_core/'; PKG='/home/claude/pkg_in/'
def resample_to(x,fs,t):
    if fs==t: return x
    fr=Fraction(t,fs).limit_denominator(2000); return resample_poly(x,fr.numerator,fr.denominator)
def read(cid,fn):
    info=sf.info(AUD+fn); fr=min(int(info.frames),int(round(120*info.samplerate)))
    x,fs=sf.read(AUD+fn,frames=fr,dtype='float64',always_2d=True); return x[:,0].copy(),int(fs)
def prepare(x,fs,band4k):
    x=x-x.mean(); p=np.abs(x).max()
    if p>0: x=x/p*0.5
    if band4k:
        if fs<8000: return None
        return resample_to(resample_to(x,fs,8000),8000,16000)
    return resample_to(x,fs,16000)
def tile_to(x,n):
    if len(x)>=n: return x
    return np.tile(x,int(np.ceil(n/max(len(x),1))))[:n]
def window_starts(n,nw):
    if n<=nw: return [0]
    k=int(np.ceil(n/nw)); return [int(round(v)) for v in np.linspace(0,n-nw,k)]
# MFCC (b05)
def mel_filterbank(n_mels,nfft,fs,fmin,fmax):
    hz2mel=lambda f:2595.0*np.log10(1.0+f/700.0); mel2hz=lambda m:700.0*(10**(m/2595.0)-1.0)
    pts=mel2hz(np.linspace(hz2mel(fmin),hz2mel(fmax),n_mels+2)); freqs=np.fft.rfftfreq(nfft,1/fs); fb=np.zeros((n_mels,len(freqs)))
    for i in range(n_mels):
        lo,ce,hi=pts[i],pts[i+1],pts[i+2]; up=(freqs-lo)/max(ce-lo,1e-9); dn=(hi-freqs)/max(hi-ce,1e-9); fb[i]=np.clip(np.minimum(up,dn),0,None)
    return fb
def mfcc_summary(x,fs,n_mels,fmin,fmax,n_mfcc=20):
    from scipy.fft import dct
    win,hop=int(round(0.025*fs)),int(round(0.010*fs)); nfft=int(2**np.ceil(np.log2(win)))
    if len(x)<win: x=np.pad(x,(0,win-len(x)))
    nf=1+(len(x)-win)//hop; idx=np.arange(win)[None,:]+hop*np.arange(nf)[:,None]
    spec=np.abs(np.fft.rfft(x[idx]*np.hanning(win)[None,:],n=nfft,axis=1))**2
    logmel=10*np.log10(spec@mel_filterbank(n_mels,nfft,fs,fmin,fmax).T+1e-10)
    c=dct(logmel,type=2,axis=1,norm='ortho')[:,1:n_mfcc+1]; d=np.gradient(c,axis=0) if len(c)>=3 else np.zeros_like(c)
    return np.concatenate([c.mean(0),c.std(0),d.mean(0),d.std(0)]).astype(np.float32)
class AST:
    def __init__(self):
        from transformers import ASTFeatureExtractor, ASTModel
        N='MIT/ast-finetuned-audioset-10-10-0.4593'; R='f826b80d28226b62986cc218e5cec390b1096902'
        self.fe=ASTFeatureExtractor.from_pretrained(N,revision=R); self.model,info=ASTModel.from_pretrained(N,revision=R,output_loading_info=True,attn_implementation='sdpa')
        assert not info.get('missing_keys') and not info.get('mismatched_keys'), info
        self.model.eval()
    def __call__(self,x16):
        nw=int(10.24*16000); ch=[tile_to(x16,nw)] if len(x16)<=nw else [x16[s:s+nw] for s in window_starts(len(x16),nw)]
        with torch.inference_mode():
            inp=self.fe([c.astype(np.float32) for c in ch],sampling_rate=16000,return_tensors='pt'); out=self.model(**inp).pooler_output
        return out.mean(0).numpy()
class VGG:
    def __init__(self):
        from vggish_reference import VGGish, VGGishInputProcessor
        self.proc=VGGishInputProcessor(); self.model=VGGish()
        w=torch.hub.load_state_dict_from_url('https://download.pytorch.org/torchaudio/models/vggish.pt',map_location='cpu',weights_only=True)
        self.model.load_state_dict(w,strict=True); self.model.embedding_network[-1]=torch.nn.Identity(); self.model.eval()
    def __call__(self,x16):
        x=tile_to(x16,16000).astype(np.float32)
        with torch.inference_mode(): e=self.model(self.proc(torch.from_numpy(x)))
        return e.reshape(-1,128).mean(0).numpy()
