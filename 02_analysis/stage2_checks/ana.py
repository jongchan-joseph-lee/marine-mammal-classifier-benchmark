import sys; sys.path.insert(0,'/home/claude/s1')
from common import *
import json
from sklearn.metrics import roc_auc_score
R=json.load(open('phys2.json'))
ids=IDS; P0=X['physv0'].copy(); pn=PHYS_NAMES
iA=[pn.index(k) for k in ['log_sig_f05','log_sig_f50','log_sig_f95','sig_bw_oct','sig_bins_frac','rise_peak_db']]
iB=[i for i,n in enumerate(pn) if n.startswith('rise3oct_')]
def build(tag, Bkey):
    Z=P0.copy()
    for j,k in zip(iA,['f05','f50','f95','bw','bins_frac','peak_db']): Z[:,j]=[R[c][f'{tag}_{k}'] for c in ids]
    Z[:,iB]=np.array([R[c][f'{tag}_{Bkey}'] for c in ids],float)
    return Z
V={'현재 (d0, Nyquist 위 = 0)':build('d0','B0'),'d0, Nyquist 위 = 결측':build('d0','Bn'),'d1 (최소통계 잡음), 위 = 0':build('d1','B0'),'d1, 위 = 결측':build('d1','Bn')}
common=[i for i,n in enumerate(pn) if not n.startswith('rise3oct_') or float(n.split('_')[1])*2**(1/6)<=2500]
V['d1, 결측 + 공통 관측 대역(≤2.2 kHz)만']=build('d1','Bn')[:,common]
V['d1, 1/3 옥타브 제외(20)']=np.delete(build('d1','B0'),iB,axis=1)
assert np.allclose(np.nan_to_num(V['현재 (d0, Nyquist 위 = 0)']),np.nan_to_num(P0),atol=1e-4)
OUT={'probe':[],'fsfam':[]}
fam=M.fs_family.values.astype(str)
for name,Z in V.items():
    row={'set':name,'k':Z.shape[1]}
    for t in ['species','group3','type7']:
        m=task_mask(t); y=M[TASKS[t]].values[m].astype(str); f=M.fold_all.values[m]; site=M.g_site.values[m]
        pm=cv_predict(X['meta'][m],y,f); p=cv_predict(Z[m],y,f); r=boot(y,p,pm,site,500)
        row[t]=round(r['ba'],3); row[t+'_d']=round(r['d'],3); row[t+'_sig']=bool(r['dlo']>0 or r['dhi']<0)
    # how much sample-rate family can be read from features (source split)
    m=np.ones(len(ids),bool); pf=cv_predict(Z,fam,M.fold_all.values); row['fsfam_ba']=round(ba(fam,pf),3)
    OUT['probe'].append(row); print(row,flush=True)
pm=cv_predict(X['meta'],fam,M.fold_all.values); OUT['fsfam_meta']=round(ba(fam,pm),3); print('meta fsfam',OUT['fsfam_meta'])
json.dump(OUT,open('ana_probe.json','w'),ensure_ascii=False)
