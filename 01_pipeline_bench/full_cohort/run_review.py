#!/usr/bin/env python3
"""Watkins: five predeclared exploratory analyses, development-only new fits."""
from pathlib import Path
import argparse, hashlib, json, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'
TAB = OUT / 'tables'
SEED = 20261004
FEATURES = ['meta', 'mfcc16k', 'physv0', 'vggish', 'effb3', 'ast_zero', 'ast_tile']
FORMS = ['click', 'growl', 'moan', 'pulse_train', 'pulsed_call', 'song', 'whistle']
COMMON = ['click', 'pulse_train', 'pulsed_call', 'whistle']
GROUPS = {'band': np.arange(0,7), 'octave':np.arange(7,38),
          'timing':np.arange(38,42), 'tonality':np.arange(42,45),
          'impulse':np.arange(45,49), 'repeat':np.arange(49,51)}
T0=time.time()

def log(s):
    print(f'[{time.time()-T0:6.1f}s] {s}', flush=True)

def save_table(df, name):
    df.to_csv(TAB / name, index=False)

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def fixed_ba(y,p,classes):
    if any(not np.any(y==c) for c in classes):return np.nan
    return float(np.mean([np.mean(p[y==c]==c) for c in classes]))

def macro_mean(y, values, classes):
    if any(not np.any(y==c) for c in classes):return np.nan
    return float(np.mean([np.mean(values[y==c]) for c in classes]))

def source_boot(sources, y, values, classes, n_boot=2000):
    """Class-macro means with a FIXED denominator; require all classes per draw."""
    us, gi=np.unique(sources,return_inverse=True)
    rng=np.random.default_rng(SEED)
    w=np.stack([np.bincount(rng.choice(len(us),len(us),replace=True),minlength=len(us))
                for _ in range(n_boot)])
    den=np.zeros((len(us),len(classes))); num=np.zeros_like(den)
    for j,c in enumerate(classes):
        idx=y==c
        den[:,j]=np.bincount(gi[idx],minlength=len(us))
        num[:,j]=np.bincount(gi[idx],weights=values[idx],minlength=len(us))
    d=w@den;good=np.all(d>0,axis=1)
    vals=np.mean((w[good]@num)/d[good],axis=1)
    if len(vals)<n_boot*.5:return np.nan,np.nan,int(len(vals))
    lo,hi=np.quantile(vals,[.025,.975])
    return float(lo),float(hi),int(len(vals))

def transform(Xtr,Xev=None):
    """Original frozen probe preprocessing; train-only fit."""
    if Xev is None:Xev=Xtr
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        med=np.nanmedian(np.where(np.isfinite(Xtr),Xtr,np.nan),axis=0)
    med=np.where(np.isfinite(med),med,0)
    a=np.where(np.isfinite(Xtr),Xtr,med).astype(np.float64)
    b=np.where(np.isfinite(Xev),Xev,med).astype(np.float64)
    mu=a.mean(0);sd=a.std(0);sd=np.where(sd>1e-12,sd,1.)
    a=np.clip((a-mu)/sd,-8,8);b=np.clip((b-mu)/sd,-8,8)
    k=min(64,a.shape[1],a.shape[0]-1)
    if k<a.shape[1]:
        pc=PCA(n_components=k,random_state=SEED).fit(a)
        a=pc.transform(a);b=pc.transform(b)
    return a,b

class Context:
    def __init__(self):
        # The delivered package contains canonical tables and dev-only arrays.
        # This branch reproduces all analyses without downloading the original archives.
        if not (ROOT/'raw/eval/reference_v1/predictions.csv').exists():
            self.pred=pd.read_csv(TAB/'saved_original_predictions.csv',dtype={'cut_id':str})
            self.core=pd.read_csv(TAB/'canonical_core_metadata.csv',dtype={'cut_id':str}).fillna('')
            self.dev=pd.read_csv(TAB/'development_metadata.csv',dtype={'cut_id':str}).fillna('')
            self.typ=pd.read_csv(TAB/'development_single_form_metadata.csv',dtype={'cut_id':str}).fillna('')
            self.idx=self.dev.set_index('cut_id').index.get_indexer(self.typ.cut_id)
            z=np.load(OUT/'methods/development_features.npz',allow_pickle=False)
            assert np.all(z['cut_id']==self.dev.cut_id.to_numpy())
            self.X={f:z[f] for f in FEATURES};self.paths={};self.names={}
            groups=json.loads((OUT/'methods/physical_feature_groups.json').read_text())
            self.names['physv0']=np.array([name for g in GROUPS for name in groups[g]])
            self.y=self.typ.type_single.to_numpy();self.sites=self.typ.g_site.to_numpy()
            self.folds=self.dev.fold.to_numpy().astype(int);self.tf=self.typ.fold.to_numpy().astype(int)
            return
        self.pred=pd.read_csv(ROOT/'raw/eval/reference_v1/predictions.csv',dtype={'cut_id':str})
        meta=pd.read_csv(ROOT/'raw_new/data/bench_core.csv',dtype={'cut_id':str}).fillna('')
        assert meta.cut_id.is_unique
        ref=self.pred.query("task=='group3' and protocol=='allcv' and feature=='meta'")
        self.core=meta.set_index('cut_id').loc[ref.cut_id].reset_index()
        assert len(self.core)==3539 and self.core.g_site.nunique()==49
        assert np.all(self.core.group3.to_numpy()==ref.true.to_numpy())
        self.dev=self.core.query("split=='dev'").reset_index(drop=True)
        assert len(self.dev)==2769 and self.dev.g_site.nunique()==29
        tr=self.pred.query("task=='type_single' and protocol=='devcv' and feature=='meta'")
        self.typ=self.dev.set_index('cut_id').loc[tr.cut_id].reset_index()
        assert len(self.typ)==1156
        assert np.all(self.typ.type_single.to_numpy()==tr.true.to_numpy())
        assert np.all(self.typ.g_site.to_numpy()==tr.g_site.to_numpy())
        self.idx=self.dev.set_index('cut_id').index.get_indexer(self.typ.cut_id)
        self.X={};self.paths={};self.names={}
        for f in FEATURES:
            p=ROOT/(f'raw_new/work/features/{f}.npz' if f in ['meta','mfcc16k','physv0']
                    else f'raw/features/{f}.npz')
            z=np.load(p,allow_pickle=False)
            ix=pd.Index(z['cut_id'].astype(str)).get_indexer(self.dev.cut_id)
            assert np.min(ix)>=0
            self.X[f]=z['X'][ix];self.paths[f]=p;self.names[f]=z['names'].astype(str)
        self.y=self.typ.type_single.to_numpy()
        self.sites=self.typ.g_site.to_numpy()
        self.folds=self.dev.fold.to_numpy().astype(int)
        self.tf=self.typ.fold.to_numpy().astype(int)

    def original_predictions(self, task, protocol, feature, ids=None):
        t=self.pred.query('task==@task and protocol==@protocol and feature==@feature').set_index('cut_id')
        if ids is not None:t=t.loc[ids]
        return t

def initial(ctx):
    log('Cohort and arrays verified; saving input provenance.')
    hashes={f:{'file':str(p.relative_to(ROOT)),'sha256':digest(p),'dev_shape':list(ctx.X[f].shape),
               'dev_nonfinite':int((~np.isfinite(ctx.X[f])).sum())} for f,p in ctx.paths.items()}
    predpath=ROOT/'raw/eval/reference_v1/predictions.csv'
    hashes['predictions']={'sha256':digest(predpath if predpath.exists() else TAB/'saved_original_predictions.csv')}
    validation_name='input_validation.json' if ctx.paths else 'rerun_input_validation.json'
    if not ctx.paths:hashes['development_arrays']={'sha256':digest(OUT/'methods/development_features.npz')}
    (OUT/'methods'/validation_name).write_text(json.dumps({
        'n_core':len(ctx.core),'n_dev':len(ctx.dev),'n_test':770,'dev_sources':29,'core_sources':49,
        'n_dev_form':len(ctx.typ),'label_agreement':1.0,'input_hashes':hashes},indent=2))
    save_table(ctx.core,'canonical_core_metadata.csv')
    save_table(ctx.dev,'development_metadata.csv')
    save_table(ctx.typ,'development_single_form_metadata.csv')
    # Compact inputs for reproducing analysis 1, avoiding diagnostic task bulk.
    orig=ctx.pred[(ctx.pred.task.isin(['group3','type_single'])) &
                  (ctx.pred.protocol.isin(['allcv','filecv','devcv'])) &
                  (ctx.pred.feature.isin(FEATURES))].copy()
    save_table(orig,'saved_original_predictions.csv')
    np.savez_compressed(OUT/'methods/development_features.npz',cut_id=ctx.dev.cut_id.to_numpy(dtype=str),
                        **ctx.X)
    (OUT/'methods/physical_feature_groups.json').write_text(json.dumps({
        g:ctx.names['physv0'][ix].tolist() for g,ix in GROUPS.items()},indent=2))
    rows=[];per=[];support=[]
    full=ctx.original_predictions('type_single','allcv','meta')
    fm=ctx.core.set_index('cut_id').loc[full.index]
    for scope in ['all','20k','40k','80k']:
        ids=full.index if scope=='all' else full.index[fm.fs_family==scope]
        ids=ids[full.loc[ids,'true'].isin(COMMON)]
        y=full.loc[ids,'true'].to_numpy();s=full.loc[ids,'g_site'].to_numpy();mp=full.loc[ids,'pred'].to_numpy()
        for cl in COMMON:
            i=y==cl;support.append({'scope':scope,'form':cl,'n':int(i.sum()),'sources':len(np.unique(s[i]))})
        for f in FEATURES:
            p=ctx.original_predictions('type_single','allcv',f,ids).pred.to_numpy()
            good=(p==y).astype(float); mg=(mp==y).astype(float)
            lo,hi,valid=source_boot(s,y,good,COMMON)
            dlo,dhi,dvalid=source_boot(s,y,good-mg,COMMON)
            rows.append({'scope':scope,'feature':f,'n':len(ids),'sources':len(np.unique(s)),
                         'BA':fixed_ba(y,p,COMMON),'BA_lo':lo,'BA_hi':hi,'BA_boot_valid':valid,
                         'delta_meta':fixed_ba(y,p,COMMON)-fixed_ba(y,mp,COMMON),
                         'delta_lo':dlo,'delta_hi':dhi,'delta_boot_valid':dvalid})
            for cl in COMMON:
                i=y==cl
                per.append({'scope':scope,'feature':f,'form':cl,'n':int(i.sum()),
                            'sources':len(np.unique(s[i])),'recall':float(np.mean(p[i]==y[i]))})
    save_table(pd.DataFrame(rows),'01_matched_four_form_scores.csv')
    save_table(pd.DataFrame(per),'01_matched_four_form_recall.csv')
    save_table(pd.DataFrame(support),'01_matched_four_form_support.csv')
    # Precisely explain the original random/source gap; keep tasks separate.
    rows=[]
    for task in ['group3','type_single']:
        for prot in ['filecv','allcv','devcv']:
            for f in FEATURES:
                t=ctx.original_predictions(task,prot,f)
                cls=sorted(t.true.unique());y=t.true.to_numpy();p=t.pred.to_numpy()
                rows.append({'task':task,'protocol':prot,'feature':f,'n':len(t),'classes':len(cls),
                             'BA':fixed_ba(y,p,cls)})
    save_table(pd.DataFrame(rows),'01_original_task_protocol_scores.csv')
    rc=[]
    for f in ['meta','physv0','effb3']:
        t=ctx.original_predictions('type_single','allcv',f)
        for cl,g in t.groupby('true'):
            rc.append({'feature':f,'form':cl,'n':len(g),'recall':float(np.mean(g.pred==g.true))})
    save_table(pd.DataFrame(rc),'01_original_seven_form_recalls.csv')
    log('Analysis 1 complete: identical forms, saved predictions, paired source intervals.')
    alltypes=ctx.original_predictions('type_single','allcv','meta')
    sets={'all':ctx.core.set_index('cut_id').loc[alltypes.index].reset_index(),'dev':ctx.typ}
    composition=[]; concentration=[]
    for scope,d in sets.items():
        for column in ['g_site','fs_family']:
            t=pd.crosstab(d.type_single,d[column]).reindex(FORMS,fill_value=0)
            for cl in FORMS:
                n=int(t.loc[cl].sum())
                for level,count in t.loc[cl].items():
                    composition.append({'scope':scope,'factor':column,'form':cl,'level':level,
                                        'n':int(count),'form_total':n,'share':float(count/n)})
        for cl in FORMS:
            dd=d[d.type_single==cl]; counts=dd.g_site.value_counts();probs=counts/counts.sum()
            concentration.append({'scope':scope,'form':cl,'n':len(dd),'sources':len(counts),
                                  'largest_source':counts.index[0],'largest_source_n':int(counts.iloc[0]),
                                  'largest_source_share':float(probs.iloc[0]),
                                  'effective_sources':float(np.exp(-np.sum(probs*np.log(probs)))),
                                  'species':dd.species.nunique()})
    save_table(pd.DataFrame(composition),'02_form_composition.csv')
    save_table(pd.DataFrame(concentration),'02_source_concentration.csv')
    log('Analysis 2 complete: source concentration and sampling composition, all + dev scopes.')
    retrieval(ctx)

def candidate_masks(ctx):
    n=len(ctx.typ);notself=~np.eye(n,dtype=bool)
    cross=ctx.sites[:,None]!=ctx.sites[None,:]
    fam=ctx.typ.fs_family.to_numpy();dur=ctx.typ.dur_s.to_numpy(dtype=float)
    logdur=np.log2(dur)
    matched=cross & (fam[:,None]==fam[None,:]) & (np.abs(logdur[:,None]-logdur[None,:])<=1.)
    return {'any_source':notself,'cross_source':cross,'cross_source_matched':matched}

def retrieval_queries(ctx, Xt, mask, feature, condition, query_indices=None):
    """Query-level scores, with fixed candidates and marginal random-draw expectation."""
    qi=np.arange(len(ctx.typ)) if query_indices is None else np.asarray(query_indices)
    d=cdist(Xt[qi],Xt,'sqeuclidean');d[~mask[qi]]=np.inf
    # Stable tie rule by canonical row order, disclosed for metadata-only control.
    top=np.argsort(d,axis=1,kind='stable')[:,:5]
    rows=[]
    for rownum,i in enumerate(qi):
        cand=np.flatnonzero(mask[i]);n=len(cand)
        if n<5:continue
        hit=ctx.y[top[rownum]]==ctx.y[i];positive=int(np.sum(ctx.y[cand]==ctx.y[i]))
        chance_any=1.
        for t in range(5):chance_any*=max(n-positive-t,0)/(n-t)
        r={'feature':feature,'condition':condition,'cut_id':ctx.typ.cut_id.iloc[i],
           'form':ctx.y[i],'source':ctx.sites[i],'candidates':n,'positive_candidates':positive,
           'feasible':int(positive>0),'P5':float(np.mean(hit)),'top1':int(hit[0]),
           'hit5':int(hit.any()),'null_P5':positive/n,'null_hit5':1-chance_any}
        r.update({f'neighbor_{j+1}':ctx.typ.cut_id.iloc[v] for j,v in enumerate(top[rownum])})
        rows.append(r)
    return pd.DataFrame(rows)

def retrieval_summary(q, meta=None):
    rows=[]
    for (f,cond),d in q.groupby(['feature','condition'],sort=False):
        classes=[cl for cl in FORMS if np.any(d.form==cl)]
        y=d.form.to_numpy();s=d.source.to_numpy()
        vals=d.P5.to_numpy();lo,hi,valid=source_boot(s,y,vals,classes)
        row={'feature':f,'condition':cond,'n_query':len(d),'sources':d.source.nunique(),
             'forms':len(classes),'P5_macro':macro_mean(y,vals,classes),'P5_lo':lo,'P5_hi':hi,
             'boot_valid':valid,'top1_macro':macro_mean(y,d.top1.to_numpy(),classes),
             'hit5_macro':macro_mean(y,d.hit5.to_numpy(),classes),
             'null_P5_macro':macro_mean(y,d.null_P5.to_numpy(),classes),
             'feasibility_macro':macro_mean(y,d.feasible.to_numpy(),classes),
             'P5_micro':float(d.P5.mean())}
        null=vals-d.null_P5.to_numpy();nlo,nhi,nvalid=source_boot(s,y,null,classes)
        row.update(delta_null=macro_mean(y,null,classes),delta_null_lo=nlo,delta_null_hi=nhi)
        if meta is not None:
            md=meta[meta.condition==cond].set_index('cut_id').loc[d.cut_id]
            delta=vals-md.P5.to_numpy();dl,dh,dv=source_boot(s,y,delta,classes)
            row.update(delta_meta=macro_mean(y,delta,classes),delta_meta_lo=dl,delta_meta_hi=dh,
                       delta_meta_boot_valid=dv)
        rows.append(row)
    return pd.DataFrame(rows)

def retrieval(ctx):
    masks=candidate_masks(ctx);queries=[]
    for f in FEATURES:
        Z,_=transform(ctx.X[f]); Xt=Z[ctx.idx]
        for cond,mask in masks.items():queries.append(retrieval_queries(ctx,Xt,mask,f,cond))
        log(f'Analysis 3 retrieval: {f}')
    q=pd.concat(queries,ignore_index=True)
    save_table(q,'03_retrieval_per_query.csv')
    save_table(retrieval_summary(q,q[q.feature=='meta']),'03_retrieval_summary.csv')
    per=q.groupby(['feature','condition','form'],sort=False).agg(
        n_query=('cut_id','size'),sources=('source','nunique'),P5=('P5','mean'),
        top1=('top1','mean'),hit5=('hit5','mean'),null_P5=('null_P5','mean'),
        feasible=('feasible','mean'),mean_candidates=('candidates','mean')).reset_index()
    save_table(per,'03_retrieval_per_form.csv')
    pair_distance(ctx)
    log('Analysis 3 complete: different-source retrieval and sampling/duration-matched sensitivity.')

def pair_distance(ctx):
    """Descriptive 2x2 distance check; pairs are dependent, so no pair-level p-values."""
    source_same=ctx.sites[:,None]==ctx.sites[None,:]
    form_same=ctx.y[:,None]==ctx.y[None,:]
    offdiag=~np.eye(len(ctx.y),dtype=bool)
    fam=ctx.typ.fs_family.to_numpy();ld=np.log2(ctx.typ.dur_s.to_numpy(dtype=float))
    conditions={'all':offdiag,'same_fs_duration':offdiag & (fam[:,None]==fam[None,:]) &
                (np.abs(ld[:,None]-ld[None,:])<=1)}
    categories={'same_source_same_form':source_same & form_same,
                'same_source_different_form':source_same & ~form_same,
                'different_source_same_form':~source_same & form_same,
                'different_source_different_form':~source_same & ~form_same}
    rows=[]
    for f in FEATURES:
        Z,_=transform(ctx.X[f]);d=cdist(Z[ctx.idx],Z[ctx.idx])
        scale=np.median(d[np.triu_indices(len(d),1)]);d=d/(scale if scale>0 else 1.)
        for condition,allowed in conditions.items():
            common=np.all(np.stack([(allowed&m).sum(1)>0 for m in categories.values()]),axis=0)
            for query_scope in ['available_queries','common_queries']:
                for category,m in categories.items():
                    mask=allowed&m;n=mask.sum(1);ok=(n>0)&(common if query_scope=='common_queries' else True)
                    vals=np.divide(np.sum(np.where(mask,d,0),axis=1),n,
                                   out=np.full(len(d),np.nan),where=n>0)
                    classes=[c for c in FORMS if np.any(ctx.y[ok]==c)]
                    means=[np.mean(vals[(ctx.y==c)&ok]) for c in classes]
                    # Equal weight to each query within form, then equal weight to forms.
                    rows.append({'feature':f,'condition':condition,'category':category,'query_scope':query_scope,
                                 'query_n':int(np.sum(ok)),'forms':len(classes),
                                 'ordered_pair_n':int(mask[ok].sum()),
                                 'macro_query_mean_distance':float(np.mean(means)),
                                 'normalizer':float(scale)})
    save_table(pd.DataFrame(rows),'03_distance_2x2_descriptive.csv')

def oof_classification(X,y,folds):
    pred=np.empty(len(y),dtype=object)
    for k in sorted(np.unique(folds)):
        tr=folds!=k;ev=folds==k;a,b=transform(X[tr],X[ev])
        clf=LogisticRegression(C=1.,class_weight='balanced',max_iter=3000)
        with warnings.catch_warnings(record=True) as ww:
            clf.fit(a,y[tr])
        if ww:log('Classifier warnings: '+str([str(w.message) for w in ww]))
        pred[ev]=clf.predict(b)
    return pred.astype(str)

def ablations(ctx):
    variants={'all':np.arange(51)}
    for g,ix in GROUPS.items():variants[f'only_{g}']=ix
    for g,ix in GROUPS.items():variants[f'without_{g}']=np.setdiff1d(np.arange(51),ix)
    variants['spectral']=np.r_[GROUPS['band'],GROUPS['octave']]
    variants['shape']=np.r_[GROUPS['timing'],GROUPS['tonality'],GROUPS['impulse'],GROUPS['repeat']]
    masks=candidate_masks(ctx);qs=[];summary=[];predrows=[];recrows=[]
    yt=ctx.y; yg=ctx.dev.group3.to_numpy();sg=ctx.dev.g_site.to_numpy()
    mp_t=oof_classification(ctx.X['meta'][ctx.idx],yt,ctx.tf)
    mp_g=oof_classification(ctx.X['meta'],yg,ctx.folds)
    omt=ctx.original_predictions('type_single','devcv','meta',ctx.typ.cut_id).pred.to_numpy()
    omg=ctx.original_predictions('group3','devcv','meta',ctx.dev.cut_id).pred.to_numpy()
    meta_repro={'form_agreement':float(np.mean(mp_t==omt)),
                'group_agreement':float(np.mean(mp_g==omg)),
                'form_BA':fixed_ba(yt,mp_t,FORMS),
                'group_BA':fixed_ba(yg,mp_g,sorted(np.unique(yg)))}
    (OUT/'methods/meta_probe_reproduction.json').write_text(json.dumps(meta_repro,indent=2))
    srtrue=np.log10(ctx.dev.fs_hz.to_numpy(dtype=float))
    for name,ix in variants.items():
        X=ctx.X['physv0'][:,ix]
        pt=oof_classification(X[ctx.idx],yt,ctx.tf)
        pg=oof_classification(X,yg,ctx.folds)
        original_match=None
        if name=='all':
            ot=ctx.original_predictions('type_single','devcv','physv0',ctx.typ.cut_id).pred.to_numpy()
            og=ctx.original_predictions('group3','devcv','physv0',ctx.dev.cut_id).pred.to_numpy()
            original_match={'form_prediction_agreement':float(np.mean(pt==ot)),
                            'group_prediction_agreement':float(np.mean(pg==og)),
                            'form_BA_new':fixed_ba(yt,pt,FORMS),'form_BA_original':fixed_ba(yt,ot,FORMS)}
            (OUT/'methods/native_probe_reproduction.json').write_text(json.dumps(original_match,indent=2))
            assert np.mean(pt==ot)>.98, original_match
            assert np.mean(pg==og)>.98, original_match
        srp=np.zeros(len(yg))
        for k in sorted(np.unique(ctx.folds)):
            tr=ctx.folds!=k;ev=ctx.folds==k;a,b=transform(X[tr],X[ev])
            reg=Ridge(alpha=1.).fit(a,srtrue[tr]);srp[ev]=reg.predict(b)
        row={'variant':name,'dimensions':len(ix),'logfs_R2':r2_score(srtrue,srp)}
        for task,y,p,s,classes,mp in [('form',yt,pt,ctx.sites,FORMS,mp_t),
                                     ('group3',yg,pg,sg,sorted(np.unique(yg)),mp_g)]:
            val=(p==y).astype(float);mg=(mp==y).astype(float)
            lo,hi,valid=source_boot(s,y,val,classes);dl,dh,dv=source_boot(s,y,val-mg,classes)
            row.update({task+'_BA':fixed_ba(y,p,classes),task+'_lo':lo,task+'_hi':hi,
                        task+'_delta_meta':macro_mean(y,val-mg,classes),
                        task+'_delta_lo':dl,task+'_delta_hi':dh,task+'_boot_valid':valid})
            for c in classes:
                recrows.append({'variant':name,'task':task,'class':c,'n':int(np.sum(y==c)),
                                'recall':float(np.mean(p[y==c]==c))})
            ids=ctx.typ.cut_id.to_numpy() if task=='form' else ctx.dev.cut_id.to_numpy()
            predrows.extend({'variant':name,'task':task,'cut_id':i,'source':ss,'true':yy,'pred':pp}
                            for i,ss,yy,pp in zip(ids,s,y,p))
        Z,_=transform(X)
        for cond in ['cross_source','cross_source_matched']:
            qs.append(retrieval_queries(ctx,Z[ctx.idx],masks[cond],name,cond))
        summary.append(row)
        log(f'Analysis 4 {name}: form BA={row["form_BA"]:.3f}, group BA={row["group3_BA"]:.3f}, SR R2={row["logfs_R2"]:.3f}')
    q=pd.concat(qs,ignore_index=True)
    mq=pd.read_csv(TAB/'03_retrieval_per_query.csv',dtype={'cut_id':str})
    rs=retrieval_summary(q,mq[mq.feature=='meta'])
    save_table(rs,'04_ablation_retrieval_summary.csv')
    save_table(q,'04_ablation_retrieval_per_query.csv')
    save_table(pd.DataFrame(summary),'04_ablation_probes.csv')
    save_table(pd.DataFrame(predrows),'04_ablation_predictions.csv')
    save_table(pd.DataFrame(recrows),'04_ablation_per_class_recall.csv')
    ablation_paired(ctx)
    log('Analysis 4 complete: all 15 predefined variants, no test fits or tuning.')

def ablation_paired(ctx):
    d=pd.read_csv(TAB/'04_ablation_predictions.csv',dtype={'cut_id':str})
    scores=pd.read_csv(TAB/'04_ablation_probes.csv').set_index('variant')
    for task in ['form','group3']:
        for variant in scores.index:
            p=d[(d.task==task)&(d.variant==variant)].set_index('cut_id')
            b=d[(d.task==task)&(d.variant=='all')].set_index('cut_id').loc[p.index]
            y=p.true.to_numpy();classes=sorted(np.unique(y))
            delta=(p.pred.to_numpy()==y).astype(float)-(b.pred.to_numpy()==y).astype(float)
            dl,dh,dv=source_boot(p.source.to_numpy(),y,delta,classes)
            scores.loc[variant,task+'_delta_all']=macro_mean(y,delta,classes)
            scores.loc[variant,task+'_delta_all_lo']=dl;scores.loc[variant,task+'_delta_all_hi']=dh
    save_table(scores.reset_index(),'04_ablation_probes.csv')
    q=pd.read_csv(TAB/'04_ablation_retrieval_per_query.csv',dtype={'cut_id':str})
    rs=pd.read_csv(TAB/'04_ablation_retrieval_summary.csv')
    for idx,r in rs.iterrows():
        p=q[(q.feature==r.feature)&(q.condition==r.condition)].set_index('cut_id')
        b=q[(q.feature=='all')&(q.condition==r.condition)].set_index('cut_id').loc[p.index]
        y=p.form.to_numpy();classes=[c for c in FORMS if np.any(y==c)]
        delta=p.P5.to_numpy()-b.P5.to_numpy();dl,dh,dv=source_boot(p.source.to_numpy(),y,delta,classes)
        rs.loc[idx,'delta_all']=macro_mean(y,delta,classes)
        rs.loc[idx,'delta_all_lo']=dl;rs.loc[idx,'delta_all_hi']=dh
    save_table(rs,'04_ablation_retrieval_summary.csv')

def sourcefit_retrieval(ctx):
    """Additional robustness check: the query source is absent from preprocessing fit."""
    masks=candidate_masks(ctx);rows=[]
    fullsources=ctx.dev.g_site.to_numpy()
    for f in FEATURES:
        for s in np.unique(ctx.sites):
            keep=fullsources!=s;tr,ev=transform(ctx.X[f][keep],ctx.X[f][~keep])
            Z=np.zeros((len(ctx.dev),tr.shape[1]));Z[keep]=tr;Z[~keep]=ev
            qi=np.flatnonzero(ctx.sites==s)
            for condition in ['cross_source','cross_source_matched']:
                rows.append(retrieval_queries(ctx,Z[ctx.idx],masks[condition],f,condition,qi))
        log(f'Source-excluded preprocessing retrieval: {f} done')
    q=pd.concat(rows,ignore_index=True)
    save_table(q,'03_source_excluded_fit_per_query.csv')
    save_table(retrieval_summary(q,q[q.feature=='meta']),'03_source_excluded_fit_summary.csv')
    per=q.groupby(['feature','condition','form'],sort=False).agg(n_query=('cut_id','size'),
          P5=('P5','mean'),null_P5=('null_P5','mean'),feasible=('feasible','mean')).reset_index()
    save_table(per,'03_source_excluded_fit_per_form.csv')

def nearest10(Z):
    d=cdist(Z,Z,'sqeuclidean');np.fill_diagonal(d,np.inf)
    return np.argsort(d,axis=1,kind='stable')[:,:10]

def cluster_jaccard(a,b,k=8):
    tab=np.zeros((k,k),dtype=float)
    for i in range(k):
        for j in range(k):
            aa=a==i;bb=b==j
            u=np.sum(aa|bb);tab[i,j]=np.sum(aa&bb)/u if u else 0.
    r,c=linear_sum_assignment(-tab)
    return float(np.mean(tab[r,c])), [(int(i),int(j),float(tab[i,j])) for i,j in zip(r,c)]

def stability(ctx):
    rows=[];compose=[];labels=[];cluster_detail=[];us=np.unique(ctx.dev.g_site.to_numpy())
    sr=ctx.dev.fs_family.to_numpy();source=ctx.dev.g_site.to_numpy();form=ctx.y
    for f in FEATURES:
        X=ctx.X[f];Z,_=transform(X);nn=nearest10(Z)
        same=float(np.mean(source[:,None]==source[nn]))
        for k in [4,8,12]:
            km=KMeans(n_clusters=k,n_init=10,random_state=SEED).fit(Z);lab=km.labels_
            subset=lab[ctx.idx]
            compose.append({'feature':f,'k':k,'form_AMI':adjusted_mutual_info_score(form,subset),
                            'source_AMI':adjusted_mutual_info_score(ctx.sites,subset),
                            'fs_AMI':adjusted_mutual_info_score(sr[ctx.idx],subset),
                            'same_source_NN10':same,'n_labeled':len(form)})
            if k==8:base=lab.copy()
        labels.extend({'feature':f,'cut_id':i,'cluster':int(c),'source':s,'fs_family':ff}
                      for i,c,s,ff in zip(ctx.dev.cut_id,base,source,sr))
        # Refit representation and clustering after removing every source.
        for no,s in enumerate(us):
            keep=source!=s;Zs,_=transform(X[keep])
            ls=KMeans(n_clusters=8,n_init=10,random_state=SEED).fit_predict(Zs)
            ari=adjusted_rand_score(base[keep],ls)
            jac,mapping=cluster_jaccard(base[keep],ls)
            # Same candidate pool in both neighbour sets: this isolates geometric change.
            n0=nearest10(Z[keep]);n1=nearest10(Zs)
            js=np.array([len(set(a)&set(b))/len(set(a)|set(b)) for a,b in zip(n0,n1)])
            # Source-balanced query mean prevents one large source dominating.
            sources_remaining=source[keep]
            balanced_j=float(np.mean([np.mean(js[sources_remaining==ss]) for ss in np.unique(sources_remaining)]))
            rows.append({'feature':f,'removed_source':s,'removed_n':int(np.sum(~keep)),
                         'retained_n':int(np.sum(keep)),'cluster_ARI':ari,
                         'cluster_mean_Jaccard':jac,'NN10_Jaccard_micro':float(np.mean(js)),
                         'NN10_Jaccard_source_macro':balanced_j})
            for old,new,value in mapping:
                cluster_detail.append({'feature':f,'removed_source':s,'baseline_cluster':old,
                                       'matched_cluster':new,'Jaccard':value})
            if (no+1)%10==0:log(f'Analysis 5 {f}: {no+1}/{len(us)} source removals')
        log(f'Analysis 5 {f} done; baseline same-source NN10={same:.3f}')
        # Checkpoints make each completed representation recoverable.
        save_table(pd.DataFrame(rows),'05_source_removal_stability.csv')
        save_table(pd.DataFrame(compose),'05_cluster_composition.csv')
        save_table(pd.DataFrame(labels),'05_baseline_cluster_labels.csv')
        save_table(pd.DataFrame(cluster_detail),'05_cluster_jaccard_detail.csv')
    d=pd.DataFrame(rows);s=[]
    for f,g in d.groupby('feature',sort=False):
        row={'feature':f,'source_removals':len(g)}
        for col in ['cluster_ARI','cluster_mean_Jaccard','NN10_Jaccard_source_macro']:
            row[col+'_median']=float(g[col].median());row[col+'_q10']=float(g[col].quantile(.1))
            row[col+'_q90']=float(g[col].quantile(.9));row[col+'_min']=float(g[col].min())
            row[col+'_max']=float(g[col].max())
        s.append(row)
    save_table(pd.DataFrame(s),'05_stability_summary.csv')
    log('Analysis 5 complete: seven representations, each with 29 source perturbations.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['initial','ablations','stability','sourcefit','all'],default='all')
    args=ap.parse_args();TAB.mkdir(parents=True,exist_ok=True)
    ctx=Context()
    if args.stage in ['initial','all']:initial(ctx)
    if args.stage in ['ablations','all']:ablations(ctx)
    if args.stage in ['stability','all']:stability(ctx)
    if args.stage in ['sourcefit','all']:sourcefit_retrieval(ctx)

if __name__=='__main__':main()
