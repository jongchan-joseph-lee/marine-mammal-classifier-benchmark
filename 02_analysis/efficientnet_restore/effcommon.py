import numpy as np, soundfile as sf, torch, torchvision, cv2, matplotlib
from scipy.signal import resample_poly, stft
from math import gcd
torch.set_num_threads(2)
_M=None
def model():
    global _M
    if _M is None:
        # torchvision ImageNet weights (efficientnet_b3_rwightman-b3899882.pth, = EfficientNet_B3_Weights.DEFAULT)
        m=torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT); m.eval(); _M=m
    return _M
def load40(cid):
    x,fs=sf.read(f'/home/claude/audio_core/{cid}.flac'); x=x[:,0] if x.ndim>1 else x; x=x-x.mean()
    g=gcd(40000,fs); return resample_poly(x,40000//g,fs//g)
def spec_db(x, nfft=1024, hop=256, lib='scipy'):
    if lib=='scipy':
        f,t,Z=stft(x,40000,nperseg=nfft,noverlap=nfft-hop,boundary='zeros',padded=True); P=np.abs(Z)**2
    elif lib=='torch':
        Z=torch.stft(torch.tensor(x,dtype=torch.float32),n_fft=nfft,hop_length=hop,window=torch.hann_window(nfft),center=True,return_complex=True); P=(Z.abs()**2).numpy()
    else: # plain magnitude no center
        n=1+(len(x)-nfft)//hop; idx=np.arange(nfft)[None,:]+hop*np.arange(max(n,1))[:,None]; idx=np.clip(idx,0,len(x)-1)
        P=(np.abs(np.fft.rfft(x[idx]*np.hanning(nfft)[None,:],axis=1))**2).T
    return 10*np.log10(P+1e-12)   # (F,T) low freq first
CM={k:(matplotlib.colormaps[k](np.linspace(0,1,256))[:,:3]).astype(np.float32) for k in ['viridis','magma','jet','gray','inferno']}
def to_img(S, scale='minmax', cmap='gray', flip=True, fill_rows=None):
    if scale=='minmax': v=(S-S.min())/(S.max()-S.min()+1e-9)
    elif scale.startswith('top'):
        r=float(scale[3:]); hi=S.max(); v=np.clip((S-(hi-r))/r,0,1)
    elif scale=='pct': lo,hi=np.percentile(S,[1,99]); v=np.clip((S-lo)/(hi-lo+1e-9),0,1)
    elif scale.startswith('fix'):  # fixed dB range e.g. fix-100_0
        lo,hi=map(float,scale[3:].split('_')); v=np.clip((S-lo)/(hi-lo),0,1)
    if flip: v=v[::-1]
    idx=np.round(v*255).astype(np.uint8)
    rgb=CM[cmap][idx]                      # (F,T,3) float 0..1
    rgb=(rgb*255).round().astype(np.uint8)  # emulate PNG
    return cv2.resize(rgb,(800,300),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
@torch.no_grad()
def embed(imgs, norm='raw'):
    a=np.stack(imgs)
    if norm=='imagenet': a=(a-MEAN)/STD
    t=torch.tensor(a).permute(0,3,1,2)
    m=model(); f=m.features(t); return m.avgpool(f).flatten(1).numpy()
def windows(x, n=200000, mode='zero', agg='first'):
    if len(x)<n:
        if mode=='zero': return [np.pad(x,(0,n-len(x)))]
        if mode=='tile': return [np.tile(x,int(np.ceil(n/len(x))))[:n]]
        if mode=='none': return [x]
    if agg=='first': return [x[:n]]
    k=int(np.ceil(len(x)/n)); return [x[i*n:(i+1)*n] if len(x[i*n:(i+1)*n])==n else x[-n:] for i in range(k)]
def cos(a,b): return float(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b))
