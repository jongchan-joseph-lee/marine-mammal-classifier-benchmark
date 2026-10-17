import sys; sys.path.insert(0,'/home/claude/s1')
from common import *
import json
from sklearn.metrics import roc_auc_score
T=pd.read_csv('attributes_3539.csv',index_col=0); T.index=T.index.astype(str); T=T.loc[IDS]
cols=['imp_frac','imp_kurt','rep_ici_ms','rep_prr_hz','rep_prr_prom','harm','harm_spacing_hz','fpk_over_spacing','tonal_frac','contour_cont','fm_oct_s','f50_khz','env_crest']
AT=np.c_[T[cols].values]; AT[:,[2,3,6,7,10,11]]=np.log10(np.clip(AT[:,[2,3,6,7,10,11]],1e-3,None))
tags=M.task_type_multi_tags.fillna('').astype(str).values; single=M.task_type_single.fillna('').astype(str).values
lab=np.array([s if s else m for s,m in zip(single,tags)])
has=lambda keys: np.array([any(k in s.split(';') for k in keys) for s in lab])
labeled=np.array([bool(s) for s in lab])
targets={'tonal 있음 (whistle·moan)':['whistle','moan'],'impulsive 있음 (click·pulse train)':['click','pulse_train'],'pulsed 있음 (pulsed call·growl)':['pulsed_call','growl']}
f=M.fold_all.values; O={'presence':[]}
def probe_auc(Xm,y,mask):
    p=np.full(len(y),np.nan)
    for k in np.unique(f[mask]):
        tr=mask&(f!=k); te=mask&(f==k)
        mdl=model(Xm.shape[1]); mdl.fit(Xm[tr],y[tr]); p[te]=mdl.predict_proba(Xm[te])[:,1]
    return roc_auc_score(y[mask],p[mask])
for tn,keys in targets.items():
    y=has(keys).astype(int); mk=labeled
    row=dict(target=tn,n=int(mk.sum()),pos=int(y[mk].sum()),multi_n=int((mk&(single=='')).sum()))
    for nm,Xm in [('메타데이터',X['meta']),('물리 특징 51',X['physv0']),('속성 13',AT),('속성 13 + 메타',np.c_[AT,X['meta']])]:
        row[nm]=round(probe_auc(Xm,y,mk),3)
    # same, evaluated only on multi-label clips (trained on all)
    print(row,flush=True); O['presence'].append(row)
json.dump(O,open('attr2.json','w'),ensure_ascii=False)
