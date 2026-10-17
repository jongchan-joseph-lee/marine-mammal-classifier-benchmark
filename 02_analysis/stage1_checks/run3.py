from common import *
import json
from sklearn.ensemble import RandomForestClassifier
dev=(M.split.values=='dev'); y=M.species.values[dev].astype(str); f=M.fold.values[dev]; site=M.g_site.values[dev]
print('dev',dev.sum(),'species',len(set(y)),'folds',np.unique(f))
R=[]
for feat in FEATS:
    Xd=np.nan_to_num(X[feat][dev])
    prf=np.array([None]*len(y),dtype=object); ppr=prf.copy(); seen=np.zeros(len(y),bool)
    for k in np.unique(f):
        tr,te=f!=k,f==k
        rf=RandomForestClassifier(n_estimators=300,class_weight='balanced_subsample',n_jobs=2,random_state=0).fit(Xd[tr],y[tr]); prf[te]=rf.predict(Xd[te])
        sc=StandardScaler().fit(Xd[tr]); Z=np.clip(sc.transform(Xd),-8,8); Z/=np.linalg.norm(Z,axis=1,keepdims=True)+1e-9
        cls=np.unique(y[tr]); C=np.array([Z[tr][y[tr]==c].mean(0) for c in cls]); C/=np.linalg.norm(C,axis=1,keepdims=True)+1e-9
        ppr[te]=cls[(Z[te]@C.T).argmax(1)]; seen[te]=np.isin(y[te],cls)
    r=dict(feat=feat,rf_all=round(ba(y,prf),3),rf_seen=round(ba(y[seen],prf[seen]),3),proto_all_acc=round(float((ppr==y).mean()),3),proto_seen_acc=round(float((ppr[seen]==y[seen]).mean()),3),
           n_seen=int(seen.sum()),n=int(len(y)),sp_seen=int(len(set(y[seen]))))
    R.append(r); print(r,flush=True)
json.dump(R,open('out_A7.json','w'))
