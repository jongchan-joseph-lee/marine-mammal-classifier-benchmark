import json, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
PKG='/home/claude/pkg_in/'; rng=np.random.default_rng(0)
m=pd.read_csv(PKG+'data/core_3539_metadata.csv',low_memory=False); keep=(m.fs_hz>=8000).to_numpy(); m=m[keep].reset_index(drop=True)
X={}
for f in ['meta','mfcc16k','vggish','ast_tile']:
    z=np.load(PKG+f'work/features/{f}.npz'); assert (z['cut_id'].astype(str)==pd.read_csv(PKG+'data/core_3539_metadata.csv',low_memory=False).cut_id.astype(str).to_numpy()).all(); X[f]=np.asarray(z['X'],float)[keep]
for f in ['mfcc_b4k','vggish_b4k','ast_tile_b4k']:
    z=np.load(f'{f}.npz'); assert (z['cut_id'].astype(str)==m.cut_id.astype(str).to_numpy()).all(); X[f]=np.asarray(z['X'],float)
FE=list(X)
def fit_predict(Xtr,ytr,Xte):
    med=np.nanmedian(Xtr,0); med=np.where(np.isfinite(med),med,0); Xtr=np.where(np.isfinite(Xtr),Xtr,med); Xte=np.where(np.isfinite(Xte),Xte,med)
    mu,sd=Xtr.mean(0),Xtr.std(0); sd[sd<1e-12]=1; Xtr=np.clip((Xtr-mu)/sd,-8,8); Xte=np.clip((Xte-mu)/sd,-8,8)
    k=min(64,Xtr.shape[1],Xtr.shape[0]-1); p=PCA(k,random_state=0).fit(Xtr)
    return LogisticRegression(C=1,class_weight='balanced',max_iter=3000).fit(p.transform(Xtr),ytr).predict(p.transform(Xte))
def ba(y,p,cl): return float(np.mean([np.mean(p[y==c]==c) for c in cl if (y==c).any()]))
G3={'whistle':'tonal','moan':'tonal','click':'impulsive','pulse_train':'impulsive','pulsed_call':'pulsed','growl':'pulsed'}
tasks={'taxon':m.task_group3,'species':m.task_species,'calltype':m.task_type_single.where(m.task_type_single!='bark'),'signal':m.task_type_single.map(G3)}
PAIRS=[('mfcc16k','mfcc_b4k'),('vggish','vggish_b4k'),('ast_tile','ast_tile_b4k')]
OUT={}
for tn,lab in tasks.items():
    sel=lab.notna().to_numpy(); rows=np.flatnonzero(sel); y=lab[sel].to_numpy(); site=m.g_site.to_numpy()[sel]
    for prot in ['allcv','filecv']:
        if prot=='allcv': folds=m.fold_all.to_numpy()[sel]
        else:
            folds=np.zeros(len(y),int)
            for k,(_,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=0).split(rows,y)): folds[te]=k
        P={}
        for f in FE:
            pr=np.empty(len(y),dtype=object)
            for k in np.unique(folds):
                te=folds==k
                pr[te]=fit_predict(X[f][rows[~te]],y[~te],X[f][rows[te]]) if len(np.unique(y[~te]))>1 else y[~te][0]
            P[f]=pr
        cl=np.unique(y); sites=np.unique(site); idx={s:np.flatnonzero(site==s) for s in sites}; B={f:[] for f in FE}
        for _ in range(2000):
            ii=np.concatenate([idx[s] for s in rng.choice(sites,len(sites))])
            for f in FE: B[f].append(ba(y[ii],P[f][ii],cl))
        B={f:np.array(v) for f,v in B.items()}
        r=dict(n=int(len(y)),sites=int(len(sites)),chance=1/len(cl),feat={})
        for f in FE:
            v=ba(y,P[f],cl); r['feat'][f]=dict(ba=v,lo=float(np.percentile(B[f],2.5)),hi=float(np.percentile(B[f],97.5)),
                d_meta=v-ba(y,P['meta'],cl),dm_lo=float(np.percentile(B[f]-B['meta'],2.5)),dm_hi=float(np.percentile(B[f]-B['meta'],97.5)))
        for a,b in PAIRS:
            r['feat'][b].update(d_std=r['feat'][b]['ba']-r['feat'][a]['ba'],ds_lo=float(np.percentile(B[b]-B[a],2.5)),ds_hi=float(np.percentile(B[b]-B[a],97.5)))
        OUT[f'{tn}_{prot}']=r
        print(f"\n## {tn} {prot}: n={r['n']} sites={r['sites']} chance={r['chance']:.2f}",flush=True)
        for f in FE:
            q=r['feat'][f]; s=f"  {f:13s} {q['ba']:.3f} [{q['lo']:.2f},{q['hi']:.2f}] vs meta {q['d_meta']:+.3f} [{q['dm_lo']:+.2f},{q['dm_hi']:+.2f}]"
            if 'd_std' in q: s+=f"  vs std {q['d_std']:+.3f} [{q['ds_lo']:+.2f},{q['ds_hi']:+.2f}]"
            print(s,flush=True)
json.dump(OUT,open('b4k_results.json','w'),indent=1)
