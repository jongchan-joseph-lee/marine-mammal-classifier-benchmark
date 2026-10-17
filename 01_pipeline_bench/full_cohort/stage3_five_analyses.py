"""Rerun the five exploratory analyses on all usable main cuts.

Predictions and each source-removal refit are checkpointed. All evaluation
preprocessing is fitted to training rows only; random-file CV is explicitly a
source-overlap diagnostic. The original held-out sources were already seen.
"""
import argparse
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score
from sklearn.model_selection import StratifiedKFold

from analysis_common import OUT,SEED,FEATURES,FORMS,COMMON,metadata,features,rr,FullContext,configure_original,nearest10,stable_topk,log

AN=OUT/'five_analyses'
PRED=AN/'predictions'


def prediction(feature):
    m=metadata();X=features([feature])[0][feature];PRED.mkdir(parents=True,exist_ok=True)
    target=PRED/(feature+'.csv')
    if target.exists():
        d=pd.read_csv(target,dtype={'cut_id':str});assert len(d)==3*(len(m)+int(m.type_single.isin(FORMS).sum()))
        return d
    allrows=[];warnings_out=[];splits=[]
    for task,classes,col in [('group3',['Mysticeti','Odontoceti','Pinnipedia'],'group3'),('type_single',FORMS,'type_single')]:
        pos=np.arange(len(m)) if task=='group3' else np.flatnonzero(m.type_single.isin(FORMS))
        d=m.iloc[pos].reset_index(drop=True);xx=X[pos];y=d[col].to_numpy()
        for protocol in ['allcv','filecv','devcv','test']:
            if protocol=='filecv':
                pairs=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(xx,y))
            elif protocol=='allcv':
                folds=d.fold_all.to_numpy(int);pairs=[(np.flatnonzero(folds!=k),np.flatnonzero(folds==k)) for k in sorted(set(folds))]
            elif protocol=='devcv':
                fold=d.fold.to_numpy(int);dev=d.split.eq('dev').to_numpy()
                pairs=[(np.flatnonzero(dev&(fold!=k)),np.flatnonzero(dev&(fold==k))) for k in sorted(set(fold[dev]))]
            else:pairs=[(np.flatnonzero(d.split.eq('dev')),np.flatnonzero(d.split.eq('test')))]
            for k,(tr,ev) in enumerate(pairs):
                overlap=set(d.iloc[tr].g_site)&set(d.iloc[ev].g_site)
                if protocol!='filecv':assert not overlap
                a,b=rr.transform(xx[tr],xx[ev]);clf=LogisticRegression(C=1.,class_weight='balanced',max_iter=3000)
                with warnings.catch_warnings(record=True) as ww:clf.fit(a,y[tr])
                for w in ww:warnings_out.append(dict(feature=feature,task=task,protocol=protocol,fold=k,message=str(w.message)))
                p=clf.predict(b)
                rows=pd.DataFrame(dict(feature=feature,task=task,protocol=protocol,fold=k,cut_id=d.iloc[ev].cut_id.to_numpy(),
                    true=y[ev],pred=p,g_site=d.iloc[ev].g_site.to_numpy(),fs_family=d.iloc[ev].fs_family.to_numpy()))
                allrows.append(rows)
                splits.append(dict(feature=feature,task=task,protocol=protocol,fold=k,n_train=len(tr),n_eval=len(ev),
                    source_overlap=len(overlap),missing_training_classes=';'.join(c for c in classes if c not in set(y[tr]))))
            log(f'Analysis1 {feature}: {task} {protocol} completed.')
    ans=pd.concat(allrows,ignore_index=True)
    assert len(ans)==3*(len(m)+int(m.type_single.isin(FORMS).sum())),len(ans)
    tmp=target.with_suffix('.csv.pending');ans.to_csv(tmp,index=False);tmp.replace(target)
    pd.DataFrame(splits).to_csv(PRED/(feature+'_fold_validation.csv'),index=False)
    (PRED/(feature+'_warnings.json')).write_text(json.dumps(warnings_out,indent=2))
    return ans


def analysis12():
    parts=[pd.read_csv(PRED/(f+'.csv'),dtype={'cut_id':str}) for f in FEATURES if (PRED/(f+'.csv')).exists()]
    if not parts:return
    d=pd.concat(parts,ignore_index=True);names=d.feature.unique().tolist();m=metadata().set_index('cut_id')
    out=AN/'tables';out.mkdir(exist_ok=True);scores=[];recalls=[];four=[];fourrec=[];support=[]
    for (task,prot,f),g in d.groupby(['task','protocol','feature'],sort=False):
        classes=FORMS if task=='type_single' else ['Mysticeti','Odontoceti','Pinnipedia']
        y=g.true.to_numpy();p=g.pred.to_numpy();s=g.g_site.to_numpy();v=(y==p).astype(float)
        lo,hi,valid=rr.source_boot(s,y,v,classes)
        scores.append(dict(task=task,protocol=prot,feature=f,n=len(g),sources=g.g_site.nunique(),classes=len(classes),
            BA=rr.fixed_ba(y,p,classes),BA_lo=lo,BA_hi=hi,boot_valid=valid,accuracy=float(v.mean())))
        for c in classes:
            keep=y==c
            recalls.append(dict(task=task,protocol=prot,feature=f,class_name=c,n=int(keep.sum()),
                sources=int(g.loc[keep,'g_site'].nunique()),recall=float(v[keep].mean()) if keep.any() else np.nan))
    if 'meta' in names:
        ref=d[(d.task=='type_single')&(d.protocol=='allcv')&(d.feature=='meta')].set_index('cut_id')
        for scope in ['all','20k','40k','80k']:
            ids=ref.index[ref.true.isin(COMMON)]
            if scope!='all':ids=ids[m.loc[ids,'fs_family'].eq(scope)]
            y=ref.loc[ids,'true'].to_numpy();s=ref.loc[ids,'g_site'].to_numpy();mv=(ref.loc[ids,'pred'].to_numpy()==y).astype(float)
            for c in COMMON:
                q=y==c;support.append(dict(scope=scope,form=c,n=int(q.sum()),sources=len(set(s[q]))))
            for f in names:
                ff=d[(d.task=='type_single')&(d.protocol=='allcv')&(d.feature==f)].set_index('cut_id').loc[ids]
                v=(ff.pred.to_numpy()==y).astype(float);lo,hi,valid=rr.source_boot(s,y,v,COMMON)
                dl,dh,dv=rr.source_boot(s,y,v-mv,COMMON)
                four.append(dict(scope=scope,feature=f,n=len(ids),sources=len(set(s)),BA=rr.fixed_ba(y,ff.pred.to_numpy(),COMMON),
                    BA_lo=lo,BA_hi=hi,BA_boot_valid=valid,delta_meta=rr.macro_mean(y,v-mv,COMMON),
                    delta_lo=dl,delta_hi=dh,delta_boot_valid=dv))
                for c in COMMON:
                    q=y==c;fourrec.append(dict(scope=scope,feature=f,form=c,n=int(q.sum()),sources=len(set(s[q])),recall=float(v[q].mean()) if q.any() else np.nan))
    # Paired acoustic-minus-metadata intervals use exactly the same queries.
    for row in scores:
        task=row['task'];protocol=row['protocol'];f=row['feature']
        if 'meta' not in names:continue
        g=d[(d.task==task)&(d.protocol==protocol)&(d.feature==f)].set_index('cut_id')
        ref=d[(d.task==task)&(d.protocol==protocol)&(d.feature=='meta')].set_index('cut_id').loc[g.index]
        y=g.true.to_numpy();classes=FORMS if task=='type_single' else ['Mysticeti','Odontoceti','Pinnipedia']
        delta=(g.pred.to_numpy()==y).astype(float)-(ref.pred.to_numpy()==y).astype(float)
        dl,dh,dv=rr.source_boot(g.g_site.to_numpy(),y,delta,classes)
        row.update(delta_meta=rr.macro_mean(y,delta,classes),delta_lo=dl,delta_hi=dh,delta_boot_valid=dv)
    for name,rows in [('01_task_protocol_scores',scores),('01_per_class_recall',recalls),('01_matched_four_form_scores',four),
                      ('01_matched_four_form_recall',fourrec),('01_matched_four_form_support',support)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    full=metadata();composition=[];concentration=[]
    for scope,x in [('all',full),('dev',full[full.split.eq('dev')])]:
        for form in FORMS+['bark']:
            q=x[x.type_single.eq(form)];cnt=q.g_site.value_counts();prob=cnt/cnt.sum()
            concentration.append(dict(scope=scope,form=form,n=len(q),sources=len(cnt),largest_source=cnt.index[0] if len(cnt) else '',
                largest_source_n=int(cnt.iloc[0]) if len(cnt) else 0,largest_source_share=float(prob.iloc[0]) if len(cnt) else np.nan,
                effective_sources=float(np.exp(-(prob*np.log(prob)).sum())) if len(cnt) else 0.,species=q.species.nunique(),
                included_in_comparative_probe=form in FORMS))
            for factor in ['g_site','fs_family']:
                for level,n in q[factor].value_counts().items():
                    composition.append(dict(scope=scope,factor=factor,form=form,level=level,n=int(n),form_total=len(q),share=n/len(q)))
    pd.DataFrame(composition).to_csv(out/'02_form_composition.csv',index=False)
    pd.DataFrame(concentration).to_csv(out/'02_source_concentration.csv',index=False)


def fullcontext(names):
    ctx=FullContext(names)
    ctx.pred=pd.concat([pd.read_csv(PRED/(f+'.csv'),dtype={'cut_id':str}) for f in names],ignore_index=True)
    return ctx


def stability(ctx,feature,target):
    """Same 29-source diagnostic, with exact neighbours and per-source resume."""
    out=target/'tables';out.mkdir(parents=True,exist_ok=True);X=ctx.X[feature];Z,_=rr.transform(X)
    sources=ctx.dev.g_site.to_numpy();fs=ctx.dev.fs_family.to_numpy();us=np.unique(sources)
    same=float(np.mean(sources[:,None]==sources[nearest10(Z)]));composition=[]
    for k in [4,8,12]:
        labels=KMeans(n_clusters=k,n_init=10,random_state=SEED).fit_predict(Z);subset=labels[ctx.idx]
        composition.append(dict(feature=feature,k=k,form_AMI=adjusted_mutual_info_score(ctx.y,subset),
            source_AMI=adjusted_mutual_info_score(ctx.sites,subset),fs_AMI=adjusted_mutual_info_score(fs[ctx.idx],subset),
            same_source_NN10=same,n_labeled=len(ctx.y)))
        if k==8:baseline=labels.copy()
    pd.DataFrame(composition).to_csv(out/'05_cluster_composition.csv',index=False)
    pd.DataFrame(dict(feature=feature,cut_id=ctx.dev.cut_id,cluster=baseline,source=sources,fs_family=fs)).to_csv(out/'05_baseline_cluster_labels.csv',index=False)
    # For any single removed source, its count plus ten nearest neighbours
    # contains at least ten retained neighbours. This yields the exact same
    # baseline ordering as recomputing distances after every removal.
    width=int(ctx.dev.g_site.value_counts().max())+10
    width=min(len(Z)-1,width);full_nn=np.empty((len(Z),width),np.int32)
    for start in range(0,len(Z),256):
        end=min(len(Z),start+256);D=cdist(Z[start:end],Z,'sqeuclidean')
        D[np.arange(end-start),np.arange(start,end)]=np.inf
        full_nn[start:end]=stable_topk(D,width)
    records=[];details=[];cp=target/'checkpoints';cp.mkdir(exist_ok=True)
    for no,s in enumerate(us):
        path=cp/(s+'_stability.json')
        if path.exists():
            r=json.loads(path.read_text());records.append(r['row']);details.extend(r['details']);continue
        keep=sources!=s;Zs,_=rr.transform(X[keep]);new=KMeans(n_clusters=8,n_init=10,random_state=SEED).fit_predict(Zs)
        jac,mapping=rr.cluster_jaccard(baseline[keep],new)
        ordered=full_nn[keep];valid=keep[ordered]
        selected=ordered[(valid&(np.cumsum(valid,axis=1)<=10))].reshape(int(keep.sum()),10)
        inverse=np.full(len(Z),-1,int);inverse[keep]=np.arange(keep.sum());n0=inverse[selected]
        assert (n0>=0).all()
        n1=nearest10(Zs)
        js=np.array([len(set(a)&set(b))/len(set(a)|set(b)) for a,b in zip(n0,n1)])
        remaining=sources[keep]
        row=dict(feature=feature,removed_source=s,removed_n=int((~keep).sum()),retained_n=int(keep.sum()),
            cluster_ARI=float(adjusted_rand_score(baseline[keep],new)),cluster_mean_Jaccard=jac,
            NN10_Jaccard_micro=float(js.mean()),NN10_Jaccard_source_macro=float(np.mean([js[remaining==t].mean() for t in np.unique(remaining)])))
        det=[dict(feature=feature,removed_source=s,baseline_cluster=a,matched_cluster=b,Jaccard=j) for a,b,j in mapping]
        path.write_text(json.dumps(dict(row=row,details=det),indent=2));records.append(row);details.extend(det)
        pd.DataFrame(records).to_csv(out/'05_source_removal_stability.csv',index=False)
        log(f'Analysis5 {feature}: source removal {no+1}/{len(us)} completed.')
    d=pd.DataFrame(records);summary=dict(feature=feature,source_removals=len(d))
    for col in ['cluster_ARI','cluster_mean_Jaccard','NN10_Jaccard_source_macro']:
        for name,value in [('median',d[col].median()),('q10',d[col].quantile(.1)),('q90',d[col].quantile(.9)),('min',d[col].min()),('max',d[col].max())]:summary[col+'_'+name]=float(value)
    d.to_csv(out/'05_source_removal_stability.csv',index=False)
    pd.DataFrame(details).to_csv(out/'05_cluster_jaccard_detail.csv',index=False)
    pd.DataFrame([summary]).to_csv(out/'05_stability_summary.csv',index=False)


def merge():
    central=AN/'tables';central.mkdir(exist_ok=True)
    keys=set(p.name for f in FEATURES for p in (AN/f/'tables').glob('*.csv'))
    for name in keys:
        parts=[pd.read_csv(AN/f/'tables'/name,dtype={'cut_id':str}) for f in FEATURES if (AN/f/'tables'/name).exists()]
        df=pd.concat(parts,ignore_index=True).drop_duplicates()
        if name.endswith('summary.csv') and name.startswith('03_'):
            qname='03_source_excluded_fit_per_query.csv' if 'source_excluded' in name else '03_retrieval_per_query.csv'
            paths=[AN/f/'tables'/qname for f in FEATURES if (AN/f/'tables'/qname).exists()]
            q=pd.concat([pd.read_csv(p,dtype={'cut_id':str}) for p in paths],ignore_index=True).drop_duplicates()
            if q.duplicated(['feature','condition','cut_id']).any():
                raise RuntimeError(f'Conflicting retrieval rows in {qname}; refusing duplicated query weights')
            df=rr.retrieval_summary(q,q[q.feature=='meta'])
        df.to_csv(central/name,index=False)
    for p in (AN/'physical_ablations/tables').glob('04_*.csv'):shutil.copyfile(p,central/p.name)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',default=','.join(FEATURES));ap.add_argument('--stage',default='all',choices=['predictions','retrieval','stability','ablation','all'])
    args=ap.parse_args();names=args.features.split(',');AN.mkdir(exist_ok=True)
    if args.stage in ['predictions','all']:
        for f in names:prediction(f)
        analysis12()
    if args.stage in ['retrieval','stability','all']:
        for f in names:
            representations=list(dict.fromkeys(['meta',f]))
            ctx=fullcontext(representations);target=AN/f;configure_original(target,representations)
            if args.stage in ['retrieval','all']:
                if not (target/'tables/03_distance_2x2_descriptive.csv').exists():rr.retrieval(ctx)
                if not (target/'tables/03_source_excluded_fit_summary.csv').exists():rr.sourcefit_retrieval(ctx)
            if args.stage in ['stability','all']:stability(ctx,f,target)
            merge()
    if args.stage in ['ablation','all'] and set(['meta','physv0'])<=set(names):
        ctx=fullcontext(['meta','physv0']);target=AN/'physical_ablations';configure_original(target,['meta','physv0'])
        if not (target/'tables/04_ablation_probes.csv').exists():
            shutil.copyfile(AN/'meta/tables/03_retrieval_per_query.csv',target/'tables/03_retrieval_per_query.csv')
            rr.ablations(ctx)
        merge()
    log('Requested full-cohort five-analysis stages completed.')


if __name__=='__main__':main()
