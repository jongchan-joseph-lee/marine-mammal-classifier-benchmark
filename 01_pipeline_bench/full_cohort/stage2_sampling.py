"""Selection sensitivities: descriptive structure and development-only probes.

Sampling follows the existing source split. The held-out sources are not used
to choose a sample, balance classes, fit PCA, or tune the probe.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from analysis_common import BASE,OUT,SEED,FEATURES,FORMS,metadata,features,rr,nearest10,log

GROUPS=['Odontoceti','Mysticeti','Pinnipedia']
SAMPLE_SEED=20261007


def quotas(counts,n):
    expected=counts.to_numpy(float)*n/counts.sum()
    q=np.floor(expected).astype(int)
    order=np.argsort(-(expected-q),kind='stable')
    q[order[:n-q.sum()]]+=1
    return dict(zip(counts.index,map(int,q)))


def draw(m,n,method,seed):
    rng=np.random.default_rng(seed)
    if method=='simple_random':
        return np.sort(rng.choice(len(m),n,replace=False))
    q=quotas(m.group3.value_counts().reindex(GROUPS,fill_value=0),n)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(m.group3.eq(g)),q[g],replace=False) for g in GROUPS]))


def structural(m,X,feature,cohort,weights=None):
    Z,_=rr.transform(X)
    lab=KMeans(n_clusters=8,n_init=10,random_state=SEED).fit(Z,sample_weight=weights).labels_
    nn=nearest10(Z)
    same=(m.g_site.to_numpy()[:,None]==m.g_site.to_numpy()[nn]).mean(1)
    labeled=m.type_single.isin(FORMS).to_numpy()
    return dict(feature=feature,cohort=cohort,n=len(m),sources=m.g_site.nunique(),species=m.species.nunique(),
        labeled_forms_n=int(labeled.sum()),same_source_NN10=float(same.mean()),
        same_source_NN10_source_macro=float(np.mean([same[m.g_site.eq(s)].mean() for s in m.g_site.unique()])),
        form_AMI=float(adjusted_mutual_info_score(m.type_single[labeled],lab[labeled])),
        source_AMI=float(adjusted_mutual_info_score(m.g_site,lab)),
        fs_AMI=float(adjusted_mutual_info_score(m.fs_family,lab)),
        taxon_AMI=float(adjusted_mutual_info_score(m.group3,lab)),
        weighting='none' if weights is None else cohort,
        labeled_form_classes_present=int(m.type_single[labeled].nunique()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',default=','.join(FEATURES))
    ap.add_argument('--repeats',type=int,default=10);args=ap.parse_args()
    m=metadata();arrays,_,_=features(args.features.split(','));out=OUT/'stage2';out.mkdir(exist_ok=True)
    old=pd.read_csv(BASE/'deep_review/output/tables/canonical_core_metadata.csv',dtype={'cut_id':str}).cut_id
    oldpos=np.flatnonzero(m.cut_id.isin(set(old)));assert len(oldpos)==3539
    length=m.dur_s.between(.25,60).to_numpy();lp=np.flatnonzero(length);pool=m.iloc[lp].reset_index(drop=True)
    selections={'full_eligible':np.arange(len(m)),'original_core':oldpos,'eligible_025_60':lp}
    for method in ['simple_random','taxon_proportional']:
        p=lp[draw(pool,3539,method,SAMPLE_SEED)];selections[method+'_3539']=p
        expected=pd.read_csv(BASE/f'cohort_review/data/diagnostic_{method}_3539_seed{SAMPLE_SEED}.csv',dtype={'cut_id':str}).cut_id
        assert set(m.iloc[p].cut_id)==set(expected)
    pd.concat([m.iloc[p].assign(cohort=name) for name,p in selections.items()]).to_csv(out/'descriptive_cohort_membership.csv',index=False)
    # Development-only alternatives use the same five source folds, with no
    # redistribution between development and test.
    di=np.flatnonzero(m.split.eq('dev'));dev=m.iloc[di].reset_index(drop=True)
    dp=np.flatnonzero(dev.dur_s.between(.25,60));dpool=dev.iloc[dp].reset_index(drop=True)
    od=np.flatnonzero(dev.cut_id.isin(set(old)));assert len(od)==2769
    sampling={'full_dev':[(SAMPLE_SEED,np.arange(len(dev)))],
              'original_core_dev':[(SAMPLE_SEED,od)]}
    for method in ['simple_random','taxon_proportional']:
        sampling[method+'_dev_2769']=[(SAMPLE_SEED+i,dp[draw(dpool,2769,method,SAMPLE_SEED+i)]) for i in range(args.repeats)]
    membership=[]
    for name,items in sampling.items():
        for seed,p in items:
            membership.extend(dict(cohort=name,seed=seed,cut_id=c) for c in dev.iloc[p].cut_id)
    pd.DataFrame(membership).to_csv(out/'development_sampling_membership.csv',index=False)
    for f,X in arrays.items():
        structure=[];scores=[];recall=[]
        for name,p in selections.items():
            structure.append(structural(m.iloc[p].reset_index(drop=True),X[p],f,name))
        # Weighted KMeans changes the clustering objective only. Preprocessing
        # and neighbours remain unweighted so that the diagnostic is explicit.
        for column,name in [('g_site','full_source_equal_kmeans'),('group3','full_taxon_equal_kmeans')]:
            count=m[column].value_counts();w=m[column].map(lambda v:len(m)/(len(count)*count[v])).to_numpy(float)
            structure.append(structural(m,X,f,name,w))
        pd.DataFrame(structure).to_csv(out/(f+'_structure_comparison.csv'),index=False)
        Xd=X[di]
        for name,items in sampling.items():
            for seed,p in items:
                d=dev.iloc[p].reset_index(drop=True)
                for task,classes,col in [('group3',GROUPS,'group3'),('type_single',FORMS,'type_single')]:
                    keep=np.ones(len(d),bool) if task=='group3' else d.type_single.isin(FORMS).to_numpy()
                    q=p[keep];qmeta=d[keep];y=qmeta[col].to_numpy();fold=qmeta.fold.to_numpy(int)
                    pred=rr.oof_classification(Xd[q],y,fold)
                    val=(pred==y).astype(float);lo,hi,valid=rr.source_boot(qmeta.g_site.to_numpy(),y,val,classes)
                    scores.append(dict(feature=f,cohort=name,seed=seed,task=task,n=len(y),sources=qmeta.g_site.nunique(),
                        classes_present=len(set(y)),BA=rr.fixed_ba(y,pred,classes),BA_lo=lo,BA_hi=hi,
                        boot_valid=valid,protocol='fixed_source_development_cv',all_classes_required=True))
                    for cl in classes:
                        mask=y==cl
                        recall.append(dict(feature=f,cohort=name,seed=seed,task=task,class_name=cl,n=int(mask.sum()),
                            sources=int(qmeta.loc[mask,'g_site'].nunique()),recall=float(val[mask].mean()) if mask.any() else np.nan))
                log(f'Stage2 {f}: {name}, seed {seed}, source-separated development probes complete.')
        pd.DataFrame(scores).to_csv(out/(f+'_probe_comparison.csv'),index=False)
        pd.DataFrame(recall).to_csv(out/(f+'_probe_recalls.csv'),index=False)
    for suffix in ['structure_comparison','probe_comparison','probe_recalls']:
        parts=[pd.read_csv(p) for p in out.glob('*_'+suffix+'.csv')]
        if parts:pd.concat(parts,ignore_index=True).to_csv(out/(suffix+'.csv'),index=False)
    (out/'sampling_protocol.json').write_text(json.dumps(dict(repeats=args.repeats,seed=SAMPLE_SEED,
        structural_population='All eligible main cuts; common length-limited sampling pool=10471.',
        probe_population='Development sources only; samples of 2769 from duration0.25-60s eligible development pool.',
        source_split='Original g_site splits and folds are fixed. No development/test mixing.',
        intervals='Source bootstrap per sample; variability across seeds is selection sensitivity, not an independent confidence interval.',
        balanced_clustering='Inverse source or taxon frequency weights in KMeans only. Taxon-balanced diagnostic uses labels.',
        probe_class_weight='balanced for every cohort, same as original frozen probe; no hyperparameter tuning.'),indent=2))


if __name__=='__main__':main()
