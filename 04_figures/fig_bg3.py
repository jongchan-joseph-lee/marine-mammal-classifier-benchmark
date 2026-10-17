import numpy as np, soundfile as sf, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, stft
from math import gcd
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif"],"font.size":8.5,"axes.linewidth":0.5,
 "xtick.major.width":0.5,"ytick.major.width":0.5,"xtick.major.size":2,"ytick.major.size":2})
NAVY="#0B4EA2"; ORG="#E07B00"
def spec(cid):
    x,fs=sf.read(f"/home/claude/audio_core/{cid}.flac")
    if x.ndim>1: x=x[:,0]
    g=gcd(16000,fs); y=resample_poly(x,16000//g,fs//g)
    f,t,Z=stft(y,16000,nperseg=512,noverlap=384)
    S=20*np.log10(np.abs(Z)+1e-10); return f,t,S,fs
fig=plt.figure(figsize=(6.2,3.6),dpi=300)
gs=fig.add_gridspec(2,2,height_ratios=[1.55,1],hspace=0.62,wspace=0.28,left=0.17,right=0.9,top=0.93,bottom=0.07)
for k,(cid,lab) in enumerate([("8800601U","(a) Bowhead moan, fs 10 kHz → 16 kHz input"),("5702601X","(b) Common dolphin whistle, fs 40 kHz → 16 kHz")]):
    ax=fig.add_subplot(gs[0,k]); f,t,S,fs=spec(cid)
    vmax=np.percentile(S,99.5); ax.pcolormesh(t,f/1000,S,shading="auto",cmap="magma",vmin=vmax-60,vmax=vmax,rasterized=True)
    ax.set_ylim(0,8); ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (kHz)" if k==0 else "")
    ax.set_title(lab,fontsize=8.5,loc="left")
    ny=fs/2000
    if ny<8:
        ax.axhspan(ny,8,facecolor="white",edgecolor="#7F8C99",hatch="////",lw=0)
        ax.axhline(ny,color=ORG,lw=1.2)
        ax.text(t[-1]*0.5,(ny+8)/2,f"not recorded ({ny:g}–8 kHz)",ha="center",va="center",fontsize=8,color="#2B2B2B",
                bbox=dict(fc="white",ec="none",pad=1.5))
    else:
        ax.text(t[-1]*0.97,7.6,"recorded to 20 kHz:\n0–8 kHz all filled,\nwhistle (> 8 kHz) cut",ha="right",va="top",fontsize=7.5,color="white")
# (c)
d=pd.read_csv("/home/claude/pkg_in/data/bench_manifest.csv",low_memory=False)
m=d[(d.eligible==True)&(d.fs_family!="LF(<5k)")]
fams=["5k","10k","20k","40k","80k","160k"]; cols=["#DCEBF7","#A9CDEB","#6FA8DC","#3D7CC9","#0B4EA2","#062C5E"]
ax=fig.add_subplot(gs[1,:])
tax=[("Mysticeti","Baleen whales"),("Odontoceti","Toothed whales"),("Pinnipedia","Pinnipeds")]
for i,(g,lab) in enumerate(tax):
    sub=m[m.group3==g]; n=len(sub); left=0
    for fa,c in zip(fams,cols):
        v=(sub.fs_family==fa).mean()
        if v>0:
            ax.barh(i,v,left=left,color=c,edgecolor="white",lw=0.6,height=0.68)
            if v>=0.07: ax.text(left+v/2,i,f"{v*100:.0f}%",ha="center",va="center",fontsize=7,color="white" if fa in("40k","80k","160k","20k") else "#2B2B2B")
        left+=v
    ax.text(1.01,i,f"n = {n:,}",va="center",fontsize=7.5)
ax.set_yticks(range(3)); ax.set_yticklabels([l for _,l in tax]); ax.invert_yaxis()
ax.set_xlim(0,1); ax.set_xticks([0,0.25,0.5,0.75,1]); ax.set_xticklabels(["0","25","50","75","100%"])
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.set_title("(c) Sample rate by taxon",fontsize=8.5,loc="left")
hs=[plt.Rectangle((0,0),1,1,color=c) for c in cols]
ax.legend(hs,[f.replace("k"," kHz") for f in fams],ncol=6,frameon=False,fontsize=7,loc="lower right",bbox_to_anchor=(1.0,1.0),handlelength=1,columnspacing=1)
fig.savefig("bg3_fs.png",dpi=300)
