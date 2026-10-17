import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif"],"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
K="#0B4EA2"; CM=LinearSegmentedColormap.from_list('b',['#FFFFFF','#C8E9F8','#33B5E5','#0B4EA2'])
R=json.load(open('ct6.json')); AB=['W','M','C','PT','PC','G']
ORD=['meta','physv0','mfcc16k','vggish','effb3','ast_tile']; NM={'meta':'Metadata baseline','physv0':'Physical','mfcc16k':'MFCC','vggish':'VGGish','effb3':'EfficientNet-B3','ast_tile':'AST'}
fig=plt.figure(figsize=(12.4,3.8)); gs=fig.add_gridspec(2,5,width_ratios=[1,1,1,0.62,2.4],wspace=0.15,hspace=0.45)
for k,f in enumerate(ORD):
    a=fig.add_subplot(gs[k//3,k%3]); A=np.array(R['allcv'][f]['cmn']); a.imshow(A,cmap=CM,vmin=0,vmax=0.9)
    for i in range(6): a.text(i,i,f"{A[i,i]*100:.0f}",ha='center',va='center',fontsize=8,color='white' if A[i,i]>0.5 else 'black',fontweight='bold')
    for b in [0,2,4]: a.add_patch(Rectangle((b-0.5,b-0.5),2,2,fill=False,ec='#E8590C',lw=1.1))
    a.set_xticks(range(6)); a.set_yticks(range(6)); a.set_xticklabels(AB if k//3==1 else [],fontsize=7.5); a.set_yticklabels(AB if k%3==0 else [],fontsize=7.5); a.tick_params(length=0,pad=1)
    for s in a.spines.values(): s.set_visible(False)
    a.set_title(f"{NM[f]}  {R['allcv'][f]['ba6']:.2f}",fontsize=9.5,loc='left',pad=2,color=K if f!='meta' else '0.35',fontweight='bold')
b=fig.add_subplot(gs[:,4]); ys=np.arange(len(ORD))[::-1]
for y,f in zip(ys,ORD):
    r=R['allcv'][f]
    for k,(key,off,mk) in enumerate([('ba6',0.15,'o'),('ba3',-0.15,'s')]):
        sig=f!='meta' and (r[key+'_dci'][0]>0); col=K if f!='meta' else '0.45'
        b.plot(r[key+'_ci'],[y+off]*2,color=col,lw=1.2); b.plot(r[key],y+off,mk,color=col if (sig or f=='meta') else 'white',mec=col,ms=7)
b.axvline(R['allcv']['meta']['ba6'],color='0.5',ls='--',lw=0.8); b.axvline(R['allcv']['meta']['ba3'],color='0.5',ls=':',lw=0.8)
b.axvline(1/6,color='0.75',lw=0.6); b.axvline(1/3,color='0.75',lw=0.6)
b.set_yticks(ys); b.set_yticklabels([NM[f] for f in ORD],fontsize=9.5); b.set_xlim(0,0.9); b.set_xlabel('balanced accuracy, sources apart',fontsize=9.5)
from matplotlib.lines import Line2D
b.legend([Line2D([],[],marker='o',color=K,ls=''),Line2D([],[],marker='s',color=K,ls=''),Line2D([],[],marker='o',mfc='white',mec=K,ls='')],['6 call types (chance 0.17)','3 signal types (chance 0.33)','open: not different from baseline'],fontsize=8.5,loc='upper center',bbox_to_anchor=(0.45,-0.16),ncol=3,frameon=False,columnspacing=0.8,handletextpad=0.3)
b.set_title('6-class probe: 6 call types and collapsed 3 types',fontsize=10,loc='left')
fig.text(0.02,-0.01,'W whistle · M moan · C click · PT pulse train · PC pulsed call · G growl   (rows = true, row %; orange = same signal type; title = 6-type BA)',fontsize=8.5,color='0.35')
fig.savefig('/home/claude/ms/fig_x/fig_ct6sum.png',dpi=300,bbox_inches='tight'); print('ok')
