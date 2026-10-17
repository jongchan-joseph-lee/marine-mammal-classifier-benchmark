import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys; sys.path.insert(0,'/home/claude/v4'); from spec import spec, CM
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif"],"font.size":7,"axes.linewidth":0.4})
R=np.load('ex_8101403Z.npz'); O='/home/claude/ms/fig_x/'
def norm(A,lo=20,hi=99.7): l,h=np.percentile(A,[lo,hi]); return np.clip((A-l)/(h-l),0,1)
def fixed(name,A,ext,yt,ytl,ylab,grid=None):
    fig=plt.figure(figsize=(2,0.9)); a=fig.add_axes([0.24,0.06,0.74,0.9])
    a.imshow(A,origin='lower',aspect='auto',cmap=CM,extent=ext,interpolation='nearest')
    a.set_xticks([]); a.set_yticks(yt); a.set_yticklabels(ytl,fontsize=6.5); a.tick_params(length=1.5,pad=1)
    a.set_ylabel(ylab,fontsize=6.5,labelpad=1)
    a.yaxis.set_label_coords(-0.2,0.5)
    if grid:
        for g in grid: a.axvline(g,color='white',lw=0.5)
    for s in a.spines.values(): s.set_linewidth(0.4)
    fig.savefig(O+name,dpi=300); plt.close(fig)
mel=lambda f:1127*np.log(1+f/700); inv=lambda m:700*(np.exp(m/1127)-1)
A=R['ast_in'].T; fixed('m_in_ast.png',norm(A),[0,1024,0,128],[0,128],['0','8k'],'mel 128')
VV=np.concatenate(list(R['vgg_in']),0); fixed('m_in_vgg.png',norm(VV.T),[0,VV.shape[0],0,64],[0,64],['0.1','7.5k'],'mel 64',grid=[96])
fixed('m_in_eff.png',R['eff_img'][::-1],[0,800,0,300],[0,300],['0','20k'],'300 px')
c=R['mf_c']; cz=(c-c.mean(0))/(c.std(0)+1e-9); fixed('m_in_mfcc.png',np.clip((cz.T+2.5)/5,0,1),[0,len(c),1,21],[1,20],['1','20'],'c1–c20')
t,f,D,fs,dur=spec('/home/claude/audio_core/8101403Z.flac'); fixed('m_in_phys.png',D,[0,dur,0,fs/2000],[0,fs/2000],['0','7.45k'],'native')
# AST patch figure: true mel tick labels + patch annotation
lo,hi=mel(20),mel(8000); hz=lambda b: inv(lo+(hi-lo)*b/128)
fig,a=plt.subplots(figsize=(2.6,2.0)); a.imshow(norm(A[:,:300]),origin='lower',aspect='auto',cmap=CM,extent=[0,300,0,128],interpolation='nearest')
for g in np.arange(0,301,10): a.axvline(g,color='white',lw=0.35)
for g in np.arange(0,129,10): a.axhline(g,color='white',lw=0.35)
from matplotlib.patches import Rectangle
a.add_patch(Rectangle((90,14),16,16,fill=False,ec='#E8590C',lw=1.3))
a.annotate('patch 1개\n(16×16)',xy=(106,24),xytext=(150,42),fontsize=6.5,color='#E8590C',fontweight='bold',fontfamily='Noto Sans CJK KR',arrowprops=dict(arrowstyle='-',color='#E8590C',lw=0.8),bbox=dict(fc='white',ec='none',alpha=0.85,pad=1))
yt=[0,32,64,96,128]; a.set_yticks(yt); a.set_yticklabels([f"{hz(b)/1000:.1f}" if hz(b)>=1000 else f"{hz(b):.0f}" for b in yt],fontsize=6.5); a.set_ylabel('mel band centre (kHz)',fontsize=6.5,labelpad=1)
a.set_xticks([0,100,200,300]); a.set_xticklabels(['0','1 s','2 s','3 s'],fontsize=6.5); a.set_xlabel('frame (10 ms)',fontsize=6.5,labelpad=1); a.tick_params(length=1.5,pad=1)
fig.tight_layout(pad=0.1); fig.savefig(O+'m_ast_patch.png',dpi=300); plt.close(fig)
print([round(hz(b)) for b in yt])
