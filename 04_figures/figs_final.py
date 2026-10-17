import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif"],"font.size":10,"axes.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
K="#0B4EA2"; C2="#00B5E2"; TXC={"Odontoceti":K,"Mysticeti":C2,"Pinnipedia":"#9AA5AE"}
m=pd.read_csv('/home/claude/pkg_in/data/main_10694_metadata.csv',low_memory=False)
M=pd.crosstab(m.species,m.g_site); tax=m.groupby('species').group3.first()
# order species by taxon then by size; sources by first species they belong to (diagonal look)
order_t={"Mysticeti":0,"Odontoceti":1,"Pinnipedia":2}
sp=sorted(M.index,key=lambda s:(order_t[tax[s]],-M.loc[s].sum()))
M=M.loc[sp]; main=M.idxmax(1)
cols=[];[cols.append(c) for s in sp for c in M.columns[M.loc[s]>0].tolist()[::-1] if c not in cols]
cols=sorted(M.columns,key=lambda c:(min(sp.index(s) for s in M.index[M[c]>0]), -M[c].sum()))
M=M[cols]
CM=LinearSegmentedColormap.from_list('b',['#E3F4FB','#33B5E5','#0B4EA2','#06244D'])
fig=plt.figure(figsize=(12.6,3.75)); gs=fig.add_gridspec(2,3,width_ratios=[1.55,0.95,1.15],height_ratios=[1,1],wspace=0.32,hspace=0.75)
a=fig.add_subplot(gs[:,0]); A=M.values.astype(float); A[A==0]=np.nan
im=a.imshow(A,aspect='auto',cmap=CM,norm=LogNorm(1,A[~np.isnan(A)].max()),interpolation='nearest')
for i,s in enumerate(sp): a.add_patch(Rectangle((-2.6,i-0.5),1.6,1,color=TXC[tax[s]],clip_on=False,lw=0))
a.set_xlim(-0.5,M.shape[1]-0.5); a.set_xticks([]); a.set_yticks([]); a.set_xlabel(f"{M.shape[1]} recording sources",labelpad=2); a.set_ylabel(f"{M.shape[0]} species",labelpad=14)
for s_ in ['left','bottom']: a.spines[s_].set_visible(False)
a.set_title("(a) Watkins: species and sources are nested",loc='left',fontsize=11)
cb=fig.colorbar(im,ax=a,fraction=0.035,pad=0.02); cb.set_label('clips',fontsize=9); cb.ax.tick_params(labelsize=8)
nsrc=(M>0).sum(1); n1=(nsrc==1).sum(); nsp=(M>0).sum(0); s1=(nsp==1).sum()
a.set_xlabel(f"{M.shape[1]} recording sources\n{n1} of {M.shape[0]} species come from one source; {s1} of {M.shape[1]} sources hold one species",labelpad=2,fontsize=9.5,color=K)
a.legend([Rectangle((0,0),1,1,color=v) for v in TXC.values()],["Odontocetes","Mysticetes","Pinnipeds"],ncol=1,fontsize=9.5,loc='upper right',handlelength=0.9,handletextpad=0.4)
# (b) schematic nested vs crossed
for r,(title,grid) in enumerate([("(b) Nested (Watkins-like)",np.eye(5)),("    Crossed (needed)",np.ones((5,5)))]):
    b=fig.add_subplot(gs[r,1]); b.imshow(np.where(grid>0,1,np.nan),cmap=LinearSegmentedColormap.from_list('x',[K,K]),vmin=0,vmax=1,aspect='equal')
    for k in range(6): b.axhline(k-0.5,color='white',lw=1.2); b.axvline(k-0.5,color='white',lw=1.2)
    b.set_facecolor('#EEF2F5'); b.set_xticks([]); b.set_yticks([]); [b.spines[x].set_visible(False) for x in b.spines]
    b.set_title(title,loc='left',fontsize=10.5); b.set_ylabel('species',fontsize=9,labelpad=1); b.set_xlabel('sources / regions',fontsize=9,labelpad=1)
    b.text(5.2,2,"species = source\n→ effects cannot\n    be separated" if r==0 else "each species in\nseveral sources\n→ separable",va='center',fontsize=9,color=K if r else '0.2')
# (c) top: species by number of sources with >=20 clips ; bottom: label status
c1=fig.add_subplot(gs[0,2]); s20=(M>=20).sum(1); cats=['0','1','2','≥3']; vals=[(s20==0).sum(),(s20==1).sum(),(s20==2).sum(),(s20>=3).sum()]
bars=c1.bar(cats,vals,color=['0.8','0.8','0.8',K],width=0.62)
for bb,v in zip(bars,vals): c1.text(bb.get_x()+bb.get_width()/2,v+0.4,str(v),ha='center',fontsize=9)
c1.set_title("(c) Species by sources with ≥ 20 clips",loc='left',fontsize=10.5); c1.set_ylim(0,22); c1.tick_params(labelsize=9)
c2=fig.add_subplot(gs[1,2]); seg=[("single label, usable",1557,K),("multiple",1373,C2),("none",429,'0.75'),("single, other",180,'0.88')]; x=0
for lab,v,cl in seg:
    c2.barh(0,v,left=x,color=cl,height=0.5,edgecolor='white',lw=0.8); 
    if v>300: c2.text(x+v/2,0,f"{lab.split(',')[0]}\n{v/3539*100:.0f}%",ha='center',va='center',fontsize=9,color='white' if cl==K else 'black')
    x+=v
c2.barh(-0.75,3539,color='white',edgecolor='0.6',lw=0.6,height=0.4,ls='--'); c2.text(1770,-0.75,'listening-verified: 0',ha='center',va='center',fontsize=9,color='0.35')
c2.set_xlim(0,3539); c2.set_ylim(-1.1,0.5); c2.set_yticks([]); c2.spines['left'].set_visible(False); c2.set_xlabel('core-sample clips (3,539)',fontsize=9); c2.tick_params(labelsize=8.5)
c2.set_title("     Call-type labels in clip notes",loc='left',fontsize=10.5)
fig.savefig('fig_deck/fig_final.png',dpi=300,bbox_inches='tight'); print(vals, n1, s1)
