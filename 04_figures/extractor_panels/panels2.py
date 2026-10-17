import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys; sys.path.insert(0,'/home/claude/v4'); from spec import spec, CM
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif"],"font.size":7,"axes.linewidth":0.4})
import sys
CID=sys.argv[1]; PRE=sys.argv[2]; R=np.load(f'ex_{CID}.npz'); O='/home/claude/ms/fig_x/'+PRE; K='#0B4EA2'
def norm(A,lo=20,hi=99.7): l,h=np.percentile(A,[lo,hi]); return np.clip((A-l)/(h-l),0,1)
def im(name,A,ext=None,size=(2,1),yt=None,ytl=None,ylab=None,cmap=CM,grid=None,aspect='auto',vmin=None,vmax=None,origin='lower',hl=None,xlab=None,xt=None,xtl=None):
    fig,a=plt.subplots(figsize=size); a.imshow(A,origin=origin,aspect=aspect,cmap=cmap,extent=ext,interpolation='nearest',vmin=vmin,vmax=vmax)
    a.set_xticks(xt or []); a.set_xticklabels(xtl or []); a.set_yticks(yt or []); a.set_yticklabels(ytl or []); a.tick_params(length=1.5,pad=1,labelsize=6.5)
    if ylab: a.set_ylabel(ylab,fontsize=6.5,labelpad=1)
    if xlab: a.set_xlabel(xlab,fontsize=6.5,labelpad=1)
    if grid:
        for g in grid[0]: a.axvline(g,color='white',lw=0.35)
        for g in grid[1]: a.axhline(g,color='white',lw=0.35)
    if hl: a.add_patch(Rectangle(hl[:2],hl[2],hl[3],fill=False,ec='#E8590C',lw=1.2))
    fig.tight_layout(pad=0.1); fig.savefig(O+name,dpi=300); plt.close(fig)
def vec(name,v,shape,size=(1.6,0.7)):
    v=v[:np.prod(shape)].reshape(shape); z=(v-v.mean())/(v.std()+1e-9)
    im(name,z,size=size,cmap='RdBu_r',vmin=-2.5,vmax=2.5,origin='upper')
t,f,D,fs,dur=spec(f'/home/claude/audio_core/{CID}.flac')
im('s_orig.png',D,[0,dur,0,fs/2000],(1.9,1.5),[0,2,4,6],['0','2','4','6'],'kHz',xlab='s',xt=[0,1,2],xtl=['0','1','2'])
A=R['ast_in'].T
im('in_ast.png',norm(A),[0,1024,0,128],(2,0.9),[0,128],['0','8k'],'mel 128')
VV=np.concatenate(list(R['vgg_in']),0); im('in_vgg.png',norm(VV.T),[0,VV.shape[0],0,64],(2,0.9),[0,64],['0.1','7.5k'],'mel 64',grid=([96*k for k in range(1,len(R['vgg_in']))],[]))
im('in_eff.png',R['eff_img'][::-1],[0,800,0,300],(2,0.9),[0,300],['0','20k'],'300 px')
c=R['mf_c']; cz=(c-c.mean(0))/(c.std(0)+1e-9)
im('in_mfcc.png',np.clip((cz.T+2.5)/5,0,1),[0,len(c),1,21],(2,0.9),[1,20],['1','20'],'c1–c20')
for n,k,sh in [('out_ast.png','ast_tile',(24,32)),('out_vgg.png','vggish',(8,16)),('out_eff.png','effb3',(32,48)),('out_mfcc.png','mfcc16k',(4,20))]: vec(n,R[k],sh)
# AST detail
im('ast_patch.png',norm(A[:,:300]),[0,300,0,128],(2.6,2.0),[0,64,128],['0','4','8 kHz'],'mel (0–8 kHz)',grid=(np.arange(0,301,10),np.arange(0,129,10)),hl=(90,14,16,16),xlab='frame (10 ms)',xt=[0,100,200,300],xtl=['0','1 s','2 s','3 s'])
im('ast_full.png',norm(A),[0,1024,0,128],(3.0,0.95),[0,128],['0','8k'],'mel')
ra=R['roll'].reshape(12,101); im('ast_roll.png',(ra-ra.min())/(ra.max()-ra.min()),[0,1024,0,128],(3.0,0.95),[0,128],['0','8k'],'attn',cmap='magma')
# VGGish detail
k1=min(1,len(R['vgg_in'])-1); im('vgg_in96.png',norm(R['vgg_in'][k1].T),[0,96,0,64],(1.7,1.25),[0,64],['0.1','7.5 kHz'],'mel 64',xlab='96 frames = 0.96 s')
for k in range(4):
    mp=R[f'vgg_map{k}'].T; im(f'vgg_m{k}.png',norm(mp,5,99.5),None,(1.0,0.75),cmap='Blues')
# EfficientNet detail
im('eff_img.png',R['eff_img'][::-1],[0,800,0,300],(2.6,1.2),[0,150,300],['0','10','20 kHz'],'300 px',xlab='800 px (5 s)')
for k in [0,3,4,8]: im(f'eff_m{k}.png',norm(R[f'eff_map{k}'][::-1],5,99.5),None,(1.0,0.55),cmap='Blues')
# MFCC detail
P=R['mf_pow']; im('mf_pow.png',norm(P.T),[0,len(P),0,8],(1.6,1.0),[0,8],['0','8 kHz'],'|FFT|²',xlab='25 ms frames')
fb=R['mf_fb']; fig,a=plt.subplots(figsize=(1.6,1.0)); fr=np.linspace(0,8,fb.shape[1])
for j in range(0,64,1): a.plot(fr,fb[j],lw=0.4,color=K if j%8==0 else '#7FCDEB')
a.set_yticks([]); a.set_xticks([0,4,8]); a.set_xticklabels(['0','4','8 kHz'],fontsize=6.5); a.tick_params(length=1.5,pad=1); a.spines[['top','right','left']].set_visible(False)
fig.tight_layout(pad=0.1); fig.savefig(O+'mf_fb.png',dpi=300); plt.close(fig)
im('mf_logmel.png',norm(R['mf_logmel'].T),[0,len(P),0,64],(1.6,1.0),[0,64],['1','64'],'mel band')
im('mf_c.png',np.clip((cz.T+2.5)/5,0,1),[0,len(c),1,21],(1.6,1.0),[1,20],['c1','c20'],'DCT')
v=R['mfcc16k'].reshape(4,20); z=(v-v.mean(1,keepdims=True))/(v.std(1,keepdims=True)+1e-9)
im('mf_out.png',z,None,(1.8,0.75),cmap='RdBu_r',vmin=-2.5,vmax=2.5,origin='upper',yt=[0,1,2,3],ytl=['c mean','c SD','Δ mean','Δ SD'])
print('ok')
