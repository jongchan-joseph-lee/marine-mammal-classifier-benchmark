import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif"],"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
K="#0B4EA2"; R=json.load(open('/home/claude/b4k/b4k_results.json'))
T=[('taxon','Taxon (3)'),('species','Species (10)'),('calltype','Call type (7)'),('signal','Signal type (3)')]
P=[('mfcc16k','mfcc_b4k','MFCC'),('vggish','vggish_b4k','VGGish'),('ast_tile','ast_tile_b4k','AST')]
fig,ax=plt.subplots(1,4,figsize=(12,3.1),sharey=True)
for a,(tk,tt) in zip(ax,T):
    r=R[f'{tk}_allcv']; F=r['feat']; m=F['meta']['ba']
    a.axvspan(F['meta']['lo'],F['meta']['hi'],color='#E8EEF4',zorder=0); a.axvline(m,color=K,ls='--',lw=0.9); a.axvline(r['chance'],color='0.55',ls=':',lw=0.8)
    for i,(s,b,lab) in enumerate(P):
        y=2-i; vs,vb=F[s]['ba'],F[b]['ba']
        a.plot([F[b]['lo'],F[b]['hi']],[y-0.12]*2,color=K,lw=1.2)
        a.annotate('',xy=(vb,y-0.12),xytext=(vs,y+0.12),arrowprops=dict(arrowstyle='-',color='#9FB3C8',lw=0.8))
        a.plot(vs,y+0.12,'o',mfc='white',mec=K,ms=6.5); a.plot(vb,y-0.12,'o',color=K,ms=6.5)
        sig=F[b]['ds_lo']>0 or F[b]['ds_hi']<0
        a.text(0.99,y,("0.00" if abs(F[b]['d_std'])<0.005 else f"{F[b]['d_std']:+.2f}")+('*' if sig else ''),transform=a.get_yaxis_transform(),ha='right',va='center',fontsize=9.5,color=K if sig else '0.35')
    a.set_xlim(0,1.05); a.set_xticks([0,0.25,0.5,0.75]); a.set_title(f"{tt}  ·  {r['n']:,} clips",fontsize=10.5,loc='left'); a.set_xlabel('balanced accuracy (sources apart)')
ax[0].set_yticks([2,1,0]); ax[0].set_yticklabels([p[2] for p in P]); ax[0].set_ylim(-0.6,2.6)
fig.legend([Line2D([],[],marker='o',ls='',mfc='white',mec=K,ms=7),Line2D([],[],marker='o',ls='-',color=K,ms=7,lw=1.2),Line2D([],[],color=K,ls='--',lw=0.9),Line2D([],[],color='0.55',ls=':',lw=0.8)],
           ['16 kHz input (standard)','0–4 kHz common band (95% CI)','metadata baseline (shaded: 95% CI)','chance'],ncol=4,loc='lower center',bbox_to_anchor=(0.5,-0.05),fontsize=9.5)
fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig('fig_deck/fig_b4k.png',dpi=300,bbox_inches='tight'); print('ok')
