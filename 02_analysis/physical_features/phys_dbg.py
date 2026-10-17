import numpy as np, soundfile as sf, pandas as pd
WIN_S=0.032; NOISE=10*np.log10(np.log(20)/np.log(2)); MARGIN=3.0; TF_DB=9.0
THIRD=[10**(k/10) for k in range(18,49)]
def nfft_for(fs,n):
    k=int(2**round(np.log2(max(fs*WIN_S,64))))
    while k>64 and n<4*k: k//=2
    return int(min(max(k,64),8192))
def stft_power(x,nfft):
    hop=nfft//2
    if len(x)<nfft: x=np.pad(x,(0,nfft-len(x)))
    nf=1+(len(x)-nfft)//hop; win=np.hanning(nfft)[None,:]; idx=np.arange(nfft)[None,:]+hop*np.arange(nf)[:,None]
    return np.abs(np.fft.rfft(x[idx]*win,axis=1))**2
def _runs(m):
    if not m.any(): return []
    d=np.diff(np.concatenate([[0],m.astype(int),[0]])); return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))
def phys(x,fs):
    D={}; out=np.full(51,np.nan); n=len(x); x=x-x.mean(); nfft=nfft_for(fs,n); hop_s=(nfft//2)/fs
    P=stft_power(x,nfft); T=P.shape[0]; freqs=np.fft.rfftfreq(nfft,1/fs); lo=freqs>=max(10.0,2*fs/nfft)
    p50=np.percentile(P,50,axis=0)+1e-30; p95=np.percentile(P,95,axis=0); rise=10*np.log10((p95+1e-30)/p50)
    rs=np.median(np.lib.stride_tricks.sliding_window_view(np.pad(rise,1,mode='edge'),3),axis=1)
    wgt=np.clip(rs-(NOISE+MARGIN),0,None); wgt[~lo]=0
    D.update(P=P,freqs=freqs,rs=rs,hop_s=hop_s,nfft=nfft,lo=lo)
    if wgt.sum()>0:
        c=np.cumsum(wgt)/wgt.sum(); f=[float(freqs[min(np.searchsorted(c,q),len(freqs)-1)]) for q in (0.05,0.5,0.95)]
        out[0:4]=[np.log10(max(f[0],1)),np.log10(max(f[1],1)),np.log10(max(f[2],1)),np.log2(max(f[2],1)/max(f[0],1))]; D['f']=f
    out[4]=float((wgt[lo]>0).mean()); out[5]=float(rs[lo].max()); W=P/p50[None,:]; hot=(W>10**(TF_DB/10))&lo[None,:]; out[6]=float(hot[:,lo].mean())
    for j,fc in enumerate(THIRD):
        b=(freqs>=fc*2**(-1/6))&(freqs<fc*2**(1/6))&lo; out[7+j]=float(wgt[b].mean()) if b.any() else 0.0
    a=hot[:,lo].mean(1); act=a>0.02; runs=_runs(act); merged=[]
    for s,e in runs:
        if merged and s-merged[-1][1]<=1: merged[-1]=(merged[-1][0],e)
        else: merged.append((s,e))
    dur=max(T*hop_s,1e-9); out[38]=float(act.mean()); out[39]=float(np.log1p(len(merged)/dur))
    if merged:
        ds=np.array([(e-s)*hop_s for s,e in merged]); out[40]=np.log10(np.median(ds)); out[41]=np.log10(ds.max())
    D.update(a=a,act=act,merged=merged)
    if act.sum()>=2:
        E=np.clip(W-1,0,None)*lo[None,:]; Ea=E[act]; pk=Ea.argmax(1); tot=Ea.sum(1)+1e-30
        i3=np.clip(pk[:,None]+np.array([-1,0,1])[None,:],0,Ea.shape[1]-1)
        out[42]=float(np.median(np.take_along_axis(Ea,i3,axis=1).sum(1)/tot))
        fpk=np.log2(np.maximum(freqs[pk],freqs[1])); ai=np.flatnonzero(act); adj=np.diff(ai)==1
        if adj.any(): out[43]=float(np.log1p(np.median(np.abs(np.diff(fpk))[adj])/hop_s))
        out[44]=float(np.percentile(fpk,75)-np.percentile(fpk,25)); D.update(pk_t=ai*hop_s,pk_f=freqs[pk])
    n2=int(2**round(np.log2(max(fs*0.001,32))))
    if n>=8*n2 and fs>=8000:
        P2=stft_power(x,n2); f2=np.fft.rfftfreq(n2,1/fs); lo2=f2>=max(500.0,2*fs/n2)
        if lo2.sum()>=2 and P2.shape[0]>=16:
            W2=P2[:,lo2]/(np.median(P2[:,lo2],axis=0)[None,:]+1e-30); env=W2.mean(1); m,sd=env.mean(),env.std()+1e-30
            out[45]=float(np.log10(((env-m)**4).mean()/sd**4)); thr=8.0*np.median(env)
            pm=(env>thr)&(env>=np.roll(env,1))&(env>np.roll(env,-1)); tp=np.flatnonzero(pm)*(n2//2)/fs
            out[46]=float(np.log1p(len(tp)/(len(env)*(n2//2)/fs)))
            if len(tp)>=3: ici=np.diff(tp); out[47]=float(np.log10(np.median(ici))); out[48]=float(ici.std()/(ici.mean()+1e-30))
            D.update(env=env,env_t=np.arange(len(env))*(n2//2)/fs,thr=thr,tp=tp)
    if T>=16 and a.std()>0:
        z=a-a.mean(); ac=np.correlate(z,z,mode='full')[T-1:]; ac=ac/(ac[0]+1e-30); fn=np.flatnonzero(ac<0)
        if len(fn) and fn[0]<len(ac)-2:
            seg=ac[fn[0]:]; k=int(seg.argmax()); out[49]=float((fn[0]+k)*hop_s); out[50]=float(seg[k])
        D.update(ac=ac,ac_lag=np.arange(len(ac))*hop_s)
    return out,D
if __name__=='__main__':
    z=np.load('/home/claude/pkg_in/work/features/physv0.npz'); ids=list(z['cut_id'])
    c=pd.read_csv('/home/claude/pkg_in/data/core_3539_metadata.csv',low_memory=False); fn=dict(zip(c.cut_id,c.file_name))
    for cid in ['5702601X','9420108A','6700900B','8800601U','6002602W','9420108L']:
        x,fs=sf.read('/home/claude/v4/audio/'+fn[cid],dtype='float64',always_2d=True); x=x[:,0]
        o,_=phys(x,fs); ref=z['X'][ids.index(cid)]
        ok=np.isclose(o,ref,rtol=1e-3,atol=1e-3,equal_nan=True); print(cid, ok.sum(),'/51', [str(z['names'][i]) for i in np.flatnonzero(~ok)][:6])
