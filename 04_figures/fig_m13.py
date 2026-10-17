import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif"],"font.size":9.5,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
K="#0B4EA2"; S=json.load(open("/home/claude/summary/summary.json"))
fig=plt.figure(figsize=(12.4,3.9)); gs=fig.add_gridspec(3,2,width_ratios=[1,2.05],hspace=0.95,wspace=0.62)
sty=[(K,"-"),("#00B5E2","-"),(K,":")]
spec=[("(a) Tonal concentration",None,None,{"tonal":"tonal","impulsive":"impulsive","pulsed":"pulsed"}),
      ("(b) Centre frequency: whistle vs moan",[2,3,4,5],["0.1","1","10","100 kHz"],{"whistle":"whistle","moan":"moan"}),
      ("(c) Inter-click interval: click vs pulse train",[-3,-2,-1,0],["1","10","100","1000 ms"],{"click":"click","pulse_train":"pulse train"})]
for r,(d,(t,tk,tl,lab)) in enumerate(zip(S["D"],spec)):
    a=fig.add_subplot(gs[r,0]); e=np.array(d["bins"])
    for g,(c,ls) in zip(d["groups"],sty):
        h=np.array(g["h"]); fill=(ls=="-" and c!=K)
        a.stairs(h,e,color=c,ls=ls,lw=1.1,label=lab[g['name']],fill=fill,alpha=0.35 if fill else 1)
        if fill: a.stairs(h,e,color=c,lw=0.9)
    if tk: a.set_xticks(tk); a.set_xticklabels(tl,fontsize=8.5)
    else: a.tick_params(labelsize=8.5)
    a.set_yticks([]); a.spines["left"].set_visible(False); a.set_title(f"{t}   AUROC {d['auroc']:.2f}",fontsize=9.5,loc='left',pad=2)
    a.set_ylim(0,a.get_ylim()[1]*1.5); a.legend(fontsize=8,loc='upper right',handlelength=1.2,borderaxespad=0.1)
# right: type x band -> taxon
E_=S["E"]; a=fig.add_subplot(gs[:,1]); cols=[K,"#00B5E2","white"]; hat=[None,None,"////"]; y=0; yt,yl=[],[]
SUB={"tonal":"whistle · moan","impulsive":"click · pulse train","pulsed":"pulsed call · growl"}; bandn={"1 kHz 미만":"≤ 1 kHz","1–8 kHz":"1–8 kHz","8 kHz 초과":"> 8 kHz"}
for ty in ["pulsed","impulsive","tonal"]:
    tn=0
    for bnd in ["8 kHz 초과","1–8 kHz","1 kHz 미만"]:
        c=next(c for c in E_["cells"] if c["type"]==ty and c["band"]==bnd); x=0; tn+=c["n"]
        if bnd=="1–8 kHz": a.axhspan(y-0.46,y+0.46,color="#DFF3FB",zorder=0)
        if c["n"]<20: a.barh(y,100,color="white",edgecolor="0.6",linewidth=0.5,linestyle="--",height=0.62); a.text(50,y,"fewer than 20 clips",ha="center",va="center",fontsize=8,color="0.4")
        else:
            for sh,fc,h in zip(c["share"],cols,hat):
                a.barh(y,sh*100,left=x,color=fc,edgecolor=K,linewidth=0.5,hatch=h,height=0.62)
                if sh>0.08: a.text(x+sh*50,y,f"{sh*100:.0f}",ha="center",va="center",fontsize=8,color="white" if fc==K else "black",bbox=dict(boxstyle="round,pad=0.12",fc="white",ec="none") if h else None)
                x+=sh*100
        a.text(101.5,y,f"n = {c['n']}",va="center",fontsize=8.5); yt.append(y); yl.append(bandn[bnd]); y+=1
    a.text(-17,y-2+0.45,ty,va="center",ha="right",fontsize=10,fontweight="bold"); a.text(-17,y-2-0.1,SUB[ty],va="center",ha="right",fontsize=8,color="0.45"); y+=0.5
a.set_yticks(yt); a.set_yticklabels(yl,fontsize=9); a.set_xlim(0,100); a.set_xlabel("share of clips (%)"); a.spines["left"].set_visible(False); a.tick_params(axis="y",length=0); a.set_ylim(-0.6,y+0.9)
for k,(f,lab,fc,h) in enumerate(zip(["ic_odo.png","ic_mys.png","ic_pin.png"],["Odontocetes","Mysticetes","Pinnipeds"],cols,hat)):
    xc=12+k*31; img=mpimg.imread("fig_deck/"+f); a.add_artist(AnnotationBbox(OffsetImage(img,zoom=0.06),(xc,y+0.35),frameon=False,annotation_clip=False))
    a.add_patch(plt.Rectangle((xc+5,y+0.12),3.2,0.42,facecolor=fc,edgecolor=K,hatch=h,linewidth=0.5,clip_on=False)); a.text(xc+9.5,y+0.33,lab,va="center",fontsize=9.5)
a.set_title("(d) Taxon share by signal type × centre-frequency band",fontsize=10,loc='left',pad=22)
fig.savefig("fig_x/fig_m13.png",dpi=300,bbox_inches="tight"); print("ok")
