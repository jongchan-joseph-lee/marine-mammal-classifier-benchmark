import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif"],"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
K="#0B4EA2"; C2="#00B5E2"; CM=LinearSegmentedColormap.from_list('b',['#FFFFFF','#C8E9F8','#33B5E5','#0B4EA2'])
R=json.load(open('ct6.json')); CL=['whistle','moan','click','pulse_train','pulsed_call','growl']; LB=['whistle','moan','click','pulse train','pulsed call','growl']
FE=['meta','physv0','mfcc16k','vggish','effb3','ast_tile']; NM={'meta':'Metadata baseline','physv0':'Physical descriptors','mfcc16k':'MFCC','vggish':'VGGish','effb3':'EfficientNet-B3','ast_tile':'AST'}
def cmplot(a,cmn,title,n=None):
    A=np.array(cmn); a.imshow(A,cmap=CM,vmin=0,vmax=0.9)
    for i in range(6):
        for j in range(6):
            v=A[i,j]; a.text(j,i,f"{v*100:.0f}",ha='center',va='center',fontsize=8.5,color='white' if v>0.5 else 'black',fontweight='bold' if i==j else 'normal')
    for b in [0,2,4]: a.add_patch(Rectangle((b-0.5,b-0.5),2,2,fill=False,ec='#E8590C',lw=1.4))
    a.set_xticks(range(6)); a.set_xticklabels(LB,rotation=35,ha='right',fontsize=8.5); a.set_yticks(range(6)); a.set_yticklabels(LB,fontsize=8.5)
    a.set_xlabel('predicted',fontsize=9,labelpad=1); a.set_ylabel('true (note label)',fontsize=9); a.set_title(title,fontsize=10.5,loc='left')
    for s in a.spines.values(): s.set_visible(False)
    a.tick_params(length=0)
for f in FE:
    fig=plt.figure(figsize=(12.2,3.75)); gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.35],wspace=0.42)
    cmplot(fig.add_subplot(gs[0]),R['allcv'][f]['cmn'],'(a) Sources apart (row %)')
    cmplot(fig.add_subplot(gs[1]),R['filecv'][f]['cmn'],'(b) Random split of clips (row %)')
    a=fig.add_subplot(gs[2]); mets=[('ba6','6 call types\n(chance 0.17)'),('ba3','3 signal types\n(chance 0.33)'),('within','within type\n(given right type)')]
    for i,(k,lab) in enumerate(mets):
        y=2-i
        for g in FE:
            if g in (f,'meta'): continue
            a.plot(R['allcv'][g][k],y,'o',color='#C9D3DD',ms=5,zorder=1)
        rb=R['allcv']['meta']; a.plot(rb[k+'_ci'],[y-0.13]*2,color='0.5',lw=1.2); a.plot(rb[k],y-0.13,'D',color='0.45',ms=6,zorder=3)
        if f!='meta':
            r=R['allcv'][f]; sig=r[k+'_dci'][0]>0 or r[k+'_dci'][1]<0
            a.plot(r[k+'_ci'],[y+0.13]*2,color=K,lw=1.4); a.plot(r[k],y+0.13,'o',color=K if sig else 'white',mec=K,ms=8,zorder=4)
            a.text(1.0,y+0.13,f"{r[k]:.2f} ({r[k+'_dmeta']:+.2f}{'*' if sig else ''})",va='center',ha='right',fontsize=9,color=K,transform=a.get_yaxis_transform())
        a.text(1.0,y-0.15,f"baseline {rb[k]:.2f}",va='center',ha='right',fontsize=8.5,color='0.4',transform=a.get_yaxis_transform())
    a.set_yticks([2,1,0]); a.set_yticklabels([m[1] for m in mets],fontsize=9); a.set_xlim(0,1.15); a.set_xticks([0,0.25,0.5,0.75,1]); a.set_ylim(-0.6,2.6)
    a.set_xlabel('balanced accuracy, sources apart',fontsize=9); a.set_title('(c) Baseline vs other representations' if f=='meta' else f'(c) {NM[f]} vs baseline',fontsize=10.5,loc='left')
    a.text(0.0,-0.22,'grey dots: the other representations   ·   filled = 95% CI of difference excludes 0 (*)',transform=a.transAxes,fontsize=7.5,color='0.4')
    fig.savefig(f'/home/claude/ms/fig_x/ct6_{f}.png',dpi=300,bbox_inches='tight'); plt.close(fig)
print('ok')
