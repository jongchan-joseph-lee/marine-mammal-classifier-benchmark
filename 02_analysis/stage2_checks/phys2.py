"""Recompute physical descriptors with (a) two band-SNR definitions and (b) Nyquist-aware masking,
plus frame-level attribute features. Original-fs, same STFT as b04/b05 (32 ms Hann, 50 % overlap)."""
import numpy as np, soundfile as sf, pandas as pd, sys, json
from multiprocessing import Pool
from scipy.signal import butter, sosfiltfilt, hilbert
NOISE=10*np.log10(np.log(20)/np.log(2)); MARGIN=3.0; TF_DB=9.0; WIN_S=0.032
P10_TO_MED=np.log(2)/(-np.log(0.9))   # exponential noise: median / P10
THIRD=[10**(k/10) for k in range(18,49)]
def nfft_for(fs,n):
    k=int(2**round(np.log2(max(fs*WIN_S,64))))
    while k>64 and n<4*k: k//=2
    return int(min(max(k,64),8192))
def stft_power(x,nfft):
    hop=nfft//2
    if len(x)<nfft: x=np.pad(x,(0,nfft-len(x)))
    n=1+(len(x)-nfft)//hop; idx=np.arange(nfft)[None,:]+hop*np.arange(n)[:,None]
    return np.abs(np.fft.rfft(x[idx]*np.hanning(nfft)[None,:],axis=1))**2
def band_feats(rise, freqs, lo, fs):
    rs=np.median(np.lib.stride_tricks.sliding_window_view(np.pad(rise,1,mode='edge'),3),axis=1)
    w=np.clip(rs-(NOISE+MARGIN),0,None); w[~lo]=0
    out=dict(f05=np.nan,f50=np.nan,f95=np.nan,bw=np.nan)
    if w.sum()>0:
        c=np.cumsum(w)/w.sum(); q=[float(freqs[min(np.searchsorted(c,p),len(freqs)-1)]) for p in (0.05,0.5,0.95)]
        out.update(f05=np.log10(max(q[0],1)),f50=np.log10(max(q[1],1)),f95=np.log10(max(q[2],1)),bw=np.log2(max(q[2],1)/max(q[0],1)))
    out['bins_frac']=float((w[lo]>0).mean()); out['peak_db']=float(rs[lo].max())
    b0=[];bn=[]
    for fc in THIRD:
        lo_e,hi_e=fc*2**(-1/6),fc*2**(1/6); band=(freqs>=lo_e)&(freqs<hi_e)&lo
        v=float(w[band].mean()) if band.any() else 0.0
        b0.append(v); bn.append(v if hi_e<=fs/2 and band.any() else np.nan)   # NaN above Nyquist / unresolved
    return out,b0,bn
def feats(cid):
    try:
        x,fs=sf.read(f'/home/claude/audio_core/{cid}.flac'); x=x[:,0] if x.ndim>1 else x; x=x[:int(60*fs)]; x=x-x.mean()
        n=len(x); nfft=nfft_for(fs,n); hop_s=(nfft//2)/fs
        P=stft_power(x,nfft); T=P.shape[0]; freqs=np.fft.rfftfreq(nfft,1/fs); lo=freqs>=max(10.0,2*fs/nfft)
        p50=np.percentile(P,50,axis=0)+1e-30; p95=np.percentile(P,95,axis=0); p10=np.percentile(P,10,axis=0)+1e-30
        nf1=p10*P10_TO_MED
        R={}
        for tag,ref in [('d0',p50),('d1',nf1)]:
            rise=10*np.log10((p95+1e-30)/ref); a,b0,bn=band_feats(rise,freqs,lo,fs)
            R.update({f'{tag}_{k}':v for k,v in a.items()}); R[f'{tag}_B0']=b0; R[f'{tag}_Bn']=bn
            # duty under this floor
            W=P/ref[None,:]; hot=(W>10**(TF_DB/10))&lo[None,:]; act=hot[:,lo].mean(1)>0.02; R[f'{tag}_duty']=float(act.mean())
        # ---------- frame-level attributes (floor d1)
        W=P/nf1[None,:]; hot=(W>10**(TF_DB/10))&lo[None,:]; act=hot[:,lo].mean(1)>0.02
        A=dict(n_frames=int(T),n_active=int(act.sum()))
        if act.sum()>=3:
            E=np.clip(W-1,0,None)*lo[None,:]; Ea=E[act]; pk=Ea.argmax(1); tot=Ea.sum(1)+1e-30
            idx3=np.clip(pk[:,None]+np.array([-1,0,1])[None,:],0,Ea.shape[1]-1)
            tc=np.take_along_axis(Ea,idx3,axis=1).sum(1)/tot
            A['tc_med']=float(np.median(tc)); A['frac_tonal']=float((tc>=0.08).mean()); A['frac_impulsive']=float((tc<0.08).mean())
            fp=np.log2(np.maximum(freqs[pk],freqs[1])); ai=np.flatnonzero(act); adj=np.diff(ai)==1
            A['contour_cont']=float((np.abs(np.diff(fp))[adj]<1/6).mean()) if adj.any() else np.nan
            # harmonicity: autocorrelation of excess spectrum over spacing 100 Hz .. min(5 kHz, fs/4)
            df=freqs[1]; l0=max(2,int(np.ceil(100/df))); l1=int(min(5000,fs/4)/df)
            hs=[];sp=[]
            if l1>l0+2:
                for e in Ea[np.argsort(-tot)[:min(40,len(Ea))]]:
                    e=e-e.mean(); den=np.dot(e,e)+1e-30
                    ac=np.array([np.dot(e[:-L],e[L:]) for L in range(l0,l1)])/den
                    k=ac.argmax(); hs.append(ac[k]); sp.append((l0+k)*df)
                A['harm']=float(np.median(hs)); A['harm_spacing']=float(np.median(sp))
                A['fpk_over_spacing']=float(np.median(2**fp)/np.median(sp))
        # envelope modulation (pulse repetition), 50 ms frames, top-energy 30 %
        sos=butter(4,100,'hp',fs=fs,output='sos'); xh=sosfiltfilt(sos,x); env=np.abs(hilbert(xh))
        N=int(0.05*fs); H=N//2
        if len(x)>=2*N:
            st=np.arange(0,len(x)-N,H); en=np.array([np.sum(xh[s:s+N]**2) for s in st]); top=st[en>=np.quantile(en,0.7)]
            fm=np.fft.rfftfreq(N,1/fs); mb=(fm>=100)&(fm<=min(5000,fs/4)); w=np.hanning(N); pkh=[];pd_=[];crest=[]
            for s in top:
                e=env[s:s+N]; S=np.abs(np.fft.rfft((e-e.mean())*w))**2; Sm=S[mb]
                pkh.append(fm[mb][Sm.argmax()]); pd_.append(10*np.log10(Sm.max()/(np.median(Sm)+1e-30))); crest.append(e.max()/(np.sqrt(np.mean(e**2))+1e-30))
            A['prr_hz']=float(np.median(pkh)); A['prr_prom_db']=float(np.median(pd_)); A['env_crest']=float(np.median(crest))
        R['attr']=A; R['fs']=int(fs)
        return cid,R
    except Exception as ex:
        return cid,{'err':str(ex)}
if __name__=='__main__':
    ids=np.load('/home/claude/pkg_in/work/features/physv0.npz',allow_pickle=True)['cut_id'].astype(str).tolist()
    if len(sys.argv)>1: ids=ids[:int(sys.argv[1])]
    with Pool(2) as p: res=p.map(feats,ids,chunksize=8)
    json.dump(dict(res),open('phys2.json','w'))
    print('done',len(res),sum('err' in r for _,r in res))
