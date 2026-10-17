import json, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
PKG='/home/claude/pkg_in/'; rng=np.random.default_rng(0)
m=pd.read_csv(PKG+'data/core_3539_metadata.csv',low_memory=False)
FE=['meta','physv0','mfcc16k','vggish','effb3','ast_tile']
X={f:np.asarray(np.load(PKG+f'work/features/{f}.npz')['X'],float) for f in FE}
CL=['whistle','moan','click','pulse_train','pulsed_call','growl']; G={'whistle':'tonal','moan':'tonal','click':'impulsive','pulse_train':'impulsive','pulsed_call':'pulsed','growl':'pulsed'}
sel=m.task_type_single.isin(CL).to_numpy(); rows=np.flatnonzero(sel); y=m.task_type_single.to_numpy()[sel]; site=m.g_site.to_numpy()[sel]
def fit_predict(Xtr,ytr,Xte):
    med=np.nanmedian(Xtr,0); med=np.where(np.isfinite(med),med,0); Xtr=np.where(np.isfinite(Xtr),Xtr,med); Xte=np.where(np.isfinite(Xte),Xte,med)
    mu,sd=Xtr.mean(0),Xtr.std(0); sd[sd<1e-12]=1; Xtr=np.clip((Xtr-mu)/sd,-8,8); Xte=np.clip((Xte-mu)/sd,-8,8)
    k=min(64,Xtr.shape[1],Xtr.shape[0]-1); p=PCA(k,random_state=0).fit(Xtr)
    return LogisticRegression(C=1,class_weight='balanced',max_iter=3000).fit(p.transform(Xtr),ytr).predict(p.transform(Xte))
def ba(y,p,cl): return float(np.mean([np.mean(p[y==c]==c) for c in cl if (y==c).any()]))
def metrics(y,p):
    ty=np.array([G[v] for v in y]); tp=np.array([G[v] for v in p])
    out=dict(ba6=ba(y,p,CL),ba3=ba(ty,tp,['tonal','impulsive','pulsed']))
    # within-type: among clips whose type predicted correctly, accuracy of subtype (balanced over 6)
    ok=ty==tp; out['within']=float(np.mean([np.mean(p[(y==c)&ok]==c) for c in CL if ((y==c)&ok).any()]))
    return out
OUT={'n':int(len(y)),'counts':{c:int((y==c).sum()) for c in CL},'sites':int(len(np.unique(site)))}
for prot in ['allcv','filecv']:
    if prot=='allcv': folds=m.fold_all.to_numpy()[sel]
    else:
        folds=np.zeros(len(y),int)
        for k,(_,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=0).split(rows,y)): folds[te]=k
    P={}
    for f in FE:
        pr=np.empty(len(y),dtype=object)
        for k in np.unique(folds):
            te=folds==k; pr[te]=fit_predict(X[f][rows[~te]],y[~te],X[f][rows[te]])
        P[f]=pr
    sites=np.unique(site); idx={s:np.flatnonzero(site==s) for s in sites}; B={f:[] for f in FE}
    for _ in range(2000):
        ii=np.concatenate([idx[s] for s in rng.choice(sites,len(sites))])
        for f in FE: B[f].append(metrics(y[ii],P[f][ii]))
    res={}
    for f in FE:
        mt=metrics(y,P[f]); cm=pd.crosstab(pd.Categorical(y,CL),pd.Categorical(P[f],CL),dropna=False).values; cmn=(cm/cm.sum(1,keepdims=True)).tolist()
        r=dict(cm=cm.tolist(),cmn=cmn,recall={c:float(np.mean(P[f][y==c]==c)) for c in CL},**mt)
        for k in ['ba6','ba3','within']:
            b=np.array([d[k] for d in B[f]]); bm=np.array([d[k] for d in B['meta']])
            r[k+'_ci']=[float(np.percentile(b,2.5)),float(np.percentile(b,97.5))]; r[k+'_dmeta']=mt[k]-metrics(y,P['meta'])[k]; r[k+'_dci']=[float(np.percentile(b-bm,2.5)),float(np.percentile(b-bm,97.5))]
        res[f]=r; print(prot,f,{k:round(mt[k],3) for k in mt},'d',round(r['ba6_dmeta'],3),[round(v,2) for v in r['ba6_dci']],flush=True)
    OUT[prot]=res
json.dump(OUT,open('ct6.json','w'),indent=1)
