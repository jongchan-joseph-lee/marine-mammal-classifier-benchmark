from common import *
import json, sys
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
OUT={}
# ---------- A. species x fold table (fold_all)
m=task_mask('species'); s=M[m]
tab=pd.crosstab(s.species_ko, s.fold_all); src=s.groupby('species_ko').g_site.nunique()
OUT['A_species_fold']={'rows':[{'sp':k,'src':int(src[k]),'n':int(tab.loc[k].sum()),'folds':[int(v) for v in tab.loc[k].values]} for k in tab.index]}
print(tab.assign(src=src))
# ---------- fold builders
def folds_for(mask, y, groups, seed=0):
    sg=StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    f=np.zeros(mask.sum(),int)
    for k,(_,te) in enumerate(sg.split(np.zeros(len(y)),y,groups)): f[te]=k
    return f
def rand_folds(y,seed=0):
    f=np.zeros(len(y),int)
    for k,(_,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)): f[te]=k
    return f
# ---------- B. source definition sensitivity
B=[]
for t in ['species','group3','type7']:
    m=task_mask(t); y=M[TASKS[t]].values[m].astype(str); site=M.g_site.values[m]
    defs={'random (clip)':rand_folds(y),'session':folds_for(m,y,M.g_session.values[m]),'campaign':folds_for(m,y,M.g_campaign.values[m]),
          'source (re-split)':folds_for(m,y,site),'source (original fold_all)':M.fold_all.values[m]}
    grp={'random (clip)':None,'session':M.g_session.values[m],'campaign':M.g_campaign.values[m],'source (re-split)':site,'source (original fold_all)':site}
    for dn,f in defs.items():
        g=grp[dn]
        unseen=sum(((~pd.Series(y[f==k]).isin(set(y[f!=k]))).sum()) for k in np.unique(f))
        maxshare=max((f==k).mean() for k in np.unique(f))
        preds={feat:cv_predict(X[feat][m],y,f) for feat in FEATS}
        bg=g if g is not None else site
        for feat in FEATS:
            r=boot(y,preds[feat],None if feat=='meta' else preds['meta'],bg,n=500)
            B.append(dict(task=t,split=dn,feat=feat,n=int(m.sum()),groups=None if g is None else int(len(np.unique(g))),unseen=int(unseen),maxfold=round(float(maxshare),3),**{k:round(float(v),3) for k,v in r.items()}))
        print(t,dn,{feat:round(ba(y,preds[feat]),3) for feat in FEATS},'groups',None if g is None else len(np.unique(g)),'unseen',unseen,'maxfold',round(maxshare,2),flush=True)
OUT['B_split']=B
json.dump(OUT,open('out_AB.json','w'),ensure_ascii=False)
