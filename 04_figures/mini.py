import numpy as np, soundfile as sf, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from phys_dbg import phys, THIRD
import sys; sys.path.insert(0,'/home/claude/v4'); from spec import spec, CM
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif"],"font.size":8.5,"axes.linewidth":0.5,"xtick.major.width":0.5,"ytick.major.width":0.5,"xtick.major.size":2,"ytick.major.size":2,"axes.spines.top":False,"axes.spines.right":False})
K="#0B4EA2"; C2="#00B5E2"; GR="0.55"
c=pd.read_csv('/home/claude/pkg_in/data/core_3539_metadata.csv',low_memory=False); fn=dict(zip(c.cut_id,c.file_name))
def load(cid):
    x,fs=sf.read('/home/claude/v4/audio/'+fn[cid],dtype='float64',always_2d=True); return x[:,0],fs
SZ=(2.0,1.8); OUT='/home/claude/ms/fig_deck/'
def save(fig,name): fig.tight_layout(pad=0.2); fig.savefig(OUT+name,dpi=300); plt.close(fig)
x,fs=load('5702601X'); o,D=phys(x,fs)
# A
fig,a=plt.subplots(figsize=SZ); f=D['freqs']/1000; a.plot(f,D['rs'],color=K,lw=0.8); a.axhline(9.36,color=GR,ls='--',lw=0.6)
a.fill_between(f,9.36,D['rs'],where=D['rs']>9.36,color=C2,alpha=0.5,lw=0)
for q,lab in zip(D['f'],['f05','f50','f95']): a.axvline(q/1000,color=K,lw=0.5,ls=':'); a.text(q/1000,a.get_ylim()[1]*0.98,lab,ha='center',va='top',fontsize=7.5,color=K,bbox=dict(fc='white',ec='none',pad=0.5))
a.text(19.5,9.9,'9.4 dB',fontsize=7,color=GR,ha='right'); a.set_xlabel('kHz',labelpad=1); a.set_ylabel('band SNR (dB)',labelpad=1); a.set_xlim(0,20); save(fig,'pf_A.png')
# B
fig,a=plt.subplots(figsize=SZ); v=o[7:38]; fc=np.array(THIRD)/1000; col=[K if f_<fs/2000 else '0.85' for f_ in fc]
a.bar(np.arange(31),v,color=col,width=0.8); a.set_xticks([2,12,22,30]); a.set_xticklabels(['$10^2$','$10^3$','$10^4$','63 k']); a.set_xlabel('1/3-octave band centre (Hz)',labelpad=1); a.set_ylabel('mean excess (dB)',labelpad=1)
k=np.flatnonzero(fc>=fs/2000)[0]; a.axvspan(k-0.5,30.5,color='0.93',lw=0); a.text((k+30)/2,max(v)*0.6,'above\nNyquist\n= 0',ha='center',fontsize=7,color=GR); save(fig,'pf_B.png')
# D
fig,a=plt.subplots(figsize=SZ); t,ff,Dd,_,_=spec('/home/claude/v4/audio/'+fn['5702601X'],fmax=20000); a.pcolormesh(t,ff/1000,Dd*0.45,cmap=CM,vmin=0,vmax=1,shading='auto',rasterized=True)
a.plot(D['pk_t'],D['pk_f']/1000,'.',ms=1.6,color=K); a.set_ylim(0,20); a.set_xlabel('s',labelpad=1); a.set_ylabel('kHz',labelpad=1)
a.text(0.03,0.95,f"tonal conc. {o[42]:.2f}\nFM rate {np.expm1(o[43]):.1f} oct/s",transform=a.transAxes,va='top',fontsize=7.5,color=K,bbox=dict(fc='white',ec='none',alpha=0.8,pad=1)); save(fig,'pf_D.png')
# C, F : harp seal
x,fs=load('6700900B'); o,D=phys(x,fs); tt=np.arange(len(D['a']))*D['hop_s']
fig,a=plt.subplots(figsize=SZ); a.plot(tt,D['a'],color=K,lw=0.6); a.axhline(0.02,color=GR,ls='--',lw=0.6)
for s,e in D['merged']: a.axvspan(s*D['hop_s'],e*D['hop_s'],color=C2,alpha=0.35,lw=0)
a.set_xlabel('s',labelpad=1); a.set_ylabel('active T–F fraction',labelpad=1)
a.text(0.98,0.95,f"duty {o[38]:.2f}\n{np.expm1(o[39]):.1f} events/s",transform=a.transAxes,ha='right',va='top',fontsize=7.5,color=K,bbox=dict(fc='white',ec='none',alpha=0.8,pad=1)); save(fig,'pf_C.png')
fig,a=plt.subplots(figsize=SZ); m=D['ac_lag']<=1.5; a.plot(D['ac_lag'][m],D['ac'][m],color=K,lw=0.8); a.axhline(0,color=GR,lw=0.4)
a.plot(o[49],o[50],'o',ms=3.5,mfc='white',mec=K); a.annotate(f"lag {o[49]:.2f} s\nstrength {o[50]:.2f}",(o[49],o[50]),xytext=(o[49]+0.15,o[50]+0.25),fontsize=7.5,color=K,arrowprops=dict(arrowstyle='-',lw=0.5,color=K))
a.set_xlabel('lag (s)',labelpad=1); a.set_ylabel('autocorrelation',labelpad=1); save(fig,'pf_F.png')
# E : clicks
x,fs=load('9420108A'); o,D=phys(x,fs)
fig,a=plt.subplots(figsize=SZ); a.semilogy(D['env_t'],D['env'],color=K,lw=0.4); a.axhline(D['thr'],color=GR,ls='--',lw=0.6)
yy=np.interp(D['tp'],D['env_t'],D['env']); a.plot(D['tp'],yy*1.6,'v',ms=2.2,color=C2)
a.set_xlabel('s',labelpad=1); a.set_ylabel('1-ms envelope',labelpad=1)
a.text(0.98,0.97,f"ICI median {10**o[47]*1000:.0f} ms\n{len(D['tp'])} peaks",transform=a.transAxes,ha='right',va='top',fontsize=7.5,color=K,bbox=dict(fc='white',ec='none',alpha=0.8,pad=1)); save(fig,'pf_E.png')
print('ok')
