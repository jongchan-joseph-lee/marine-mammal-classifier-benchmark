"""Validate the actual saved cohort, coverage, predictions and five analyses."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['meta','mfcc16k','physv0','vggish','ast_zero','ast_tile']
FORMS=['click','growl','moan','pulse_train','pulsed_call','song','whistle']
DIMS=dict(meta=2,mfcc16k=80,physv0=51,vggish=128,ast_zero=768,ast_tile=768)
GROUPS=['Mysticeti','Odontoceti','Pinnipedia']
COMMON=['click','pulse_train','pulsed_call','whistle']

def independent_ba(frame,classes):
    recalls=[(frame.loc[frame.true.eq(c),'pred']==c).mean() for c in classes]
    return float(np.mean(recalls))

def metric_error(actual,saved):
    # A fixed-class score can be undefined if the evaluation scope lacks a
    # class. Accept paired NaNs explicitly; never hide a one-sided NaN in max().
    if np.isnan(actual) and np.isnan(saved):return 0.
    if not np.isfinite(actual) or not np.isfinite(saved):return float('inf')
    return abs(actual-saved)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-incomplete',action='store_true');args=ap.parse_args()
    out=ROOT/'output';records=[]
    m=pd.read_csv(out/'provenance/main_full_metadata.csv',dtype={'cut_id':str},low_memory=False).fillna('')
    ids=m.cut_id.to_numpy(dtype=str);single=m[m.type_single.isin(FORMS)];dev=m[m.split.eq('dev')]
    def check(name,passed,details):records.append(dict(check=name,passed=bool(passed),details=details))
    check('main metadata',len(m)==10694 and m.cut_id.is_unique,dict(n=len(m),unique=m.cut_id.nunique()))
    check('development/test source separation',set(dev.g_site).isdisjoint(set(m[m.split.eq('test')].g_site)),
          dict(n_dev=len(dev),n_test=int(m.split.eq('test').sum()),dev_sources=dev.g_site.nunique()))
    for col,subset in [('fold_all',m),('fold',dev)]:
        check(col+' assigns each source to one fold',subset.groupby('g_site')[col].nunique().max()==1,
              dict(sources=subset.g_site.nunique()))
    core=pd.read_csv(ROOT.parent/'deep_review/output/tables/canonical_core_metadata.csv',dtype={'cut_id':str},low_memory=False).fillna('')
    core=core[core.cut_id.isin(ids)].set_index('cut_id');now=m.set_index('cut_id').loc[core.index]
    preserved=['group3','g_site','type_single','split','fold','fs_family','fs_hz']
    mismatches={c:int((core[c].astype(str)!=now[c].astype(str)).sum()) for c in preserved}
    check('frozen core labels and split unchanged',len(core)==3539 and sum(mismatches.values())==0,
          dict(main_core_rows=len(core),mismatches=mismatches))
    hashes=m.groupby('md5_b64').agg(n=('cut_id','size'),sources=('g_site','nunique'),splits=('split','nunique'))
    check('no byte-identical FLAC crosses evaluation sources',hashes.sources.le(1).all() and hashes.splits.le(1).all(),
          dict(duplicate_hashes=int(hashes.n.gt(1).sum()),cross_source_hashes=int(hashes.sources.gt(1).sum()),development_test_hashes=int(hashes.splits.gt(1).sum())))
    audio=json.loads((out/'provenance/audio_validation.json').read_text())
    check('all original audio validated',audio.get('failed')==0 and audio.get('verified')==10694,audio)
    score_path=out/'five_analyses/tables/01_task_protocol_scores.csv'
    scores=pd.read_csv(score_path) if score_path.exists() else pd.DataFrame()
    matched_path=out/'five_analyses/tables/01_matched_four_form_scores.csv'
    matched=pd.read_csv(matched_path) if matched_path.exists() else pd.DataFrame()
    for f in FEATURES:
        p=out/'features'/(f+'.npz')
        if not p.exists():check(f+' full features',False,'file absent')
        else:
            with np.load(p,allow_pickle=False) as z:
                X=z['X'];aligned=np.array_equal(z['cut_id'].astype(str),ids)
                finite=bool(np.isfinite(X).all())
                ok=aligned and X.shape==(10694,DIMS[f]) and not np.isinf(X).any() and (finite or f=='physv0')
                check(f+' full features',ok,dict(shape=list(X.shape),IDs_exactly_aligned=aligned,
                      all_missing_rows=int(np.isnan(X).all(1).sum()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
                if f!='meta':
                    original=ROOT.parent/('deep_review/raw_new/work/features' if f in ['mfcc16k','physv0'] else 'deep_review/raw/features')/(f+'.npz')
                    with np.load(original,allow_pickle=False) as old:
                        oi=pd.Index(old['cut_id'].astype(str));ix=oi.get_indexer(ids);keep=ix>=0
                        exact=np.array_equal(X[keep],old['X'][ix[keep]],equal_nan=True)
                    check(f+' frozen core rows reused exactly',int(keep.sum())==3539 and exact,
                          dict(main_core_rows=int(keep.sum()),bitwise_equal_including_nan=exact))
        s=out/'stage1'/(f+'_structure.npz')
        if s.exists():
            with np.load(s,allow_pickle=False) as z:
                ok=np.array_equal(z['cut_id'].astype(str),ids) and z['coords'].shape==(10694,3) and np.isfinite(z['coords']).all()
                nn=z['NN10'];ok=ok and nn.shape==(10694,10) and ((nn>=0)&(nn<10694)).all() and not (nn==np.arange(10694)[:,None]).any()
                check(f+' unsupervised structure',ok,dict(coordinates=list(z['coords'].shape),neighbours=list(nn.shape)))
        else:check(f+' unsupervised structure',False,'file absent')
        pred=out/'five_analyses/predictions'/(f+'.csv')
        if pred.exists():
            d=pd.read_csv(pred,dtype={'cut_id':str});expected=3*(10694+len(single))
            expected_units={('group3',p) for p in ['allcv','filecv','devcv','test']}|{('type_single',p) for p in ['allcv','filecv','devcv','test']}
            check(f+' prediction coverage',len(d)==expected and set(d.groupby(['task','protocol']).groups)==expected_units,
                  dict(n=len(d),expected=expected))
            coverage=True;metric_errors=[];metadata_ok=True
            for task in ['group3','type_single']:
                relevant=m if task=='group3' else single
                classes=GROUPS if task=='group3' else FORMS
                for protocol in ['allcv','filecv','devcv','test']:
                    q=d[d.task.eq(task)&d.protocol.eq(protocol)]
                    target=relevant if protocol in ['allcv','filecv'] else relevant[relevant.split.eq('dev' if protocol=='devcv' else 'test')]
                    coverage &= q.cut_id.is_unique and set(q.cut_id)==set(target.cut_id)
                    if len(q) and q.cut_id.is_unique:
                        truth=target.set_index('cut_id').reindex(q.cut_id)
                        metadata_ok &= np.array_equal(q['true'].to_numpy(),truth[task].to_numpy())
                        metadata_ok &= np.array_equal(q.g_site.to_numpy(),truth.g_site.to_numpy())
                        metadata_ok &= q.pred.isin(classes).all()
                    r=scores[scores.feature.eq(f)&scores.task.eq(task)&scores.protocol.eq(protocol)]
                    value=independent_ba(q,classes)
                    metric_errors.append(metric_error(value,r.iloc[0].BA) if len(r)==1 else float('inf'))
            check(f+' unique IDs and true labels per evaluation',coverage and metadata_ok,
                  dict(no_duplicate_or_missing_evaluation_cuts=bool(coverage),metadata_labels_and_sources_match=bool(metadata_ok)))
            check(f+' reported accuracy matches saved predictions',max(metric_errors)<1e-12,
                  dict(max_BA_error=float(max(metric_errors))))
            four_errors=[]
            for scope in ['all','20k','40k','80k']:
                q=d[d.task.eq('type_single')&d.protocol.eq('allcv')&d['true'].isin(COMMON)]
                if scope!='all':q=q[q.fs_family.eq(scope)]
                r=matched[matched.feature.eq(f)&matched.scope.eq(scope)]
                four_errors.append(metric_error(independent_ba(q,COMMON),r.iloc[0].BA) if len(r)==1 else float('inf'))
            check(f+' four-form scores match saved predictions',max(four_errors)<1e-12,
                  dict(max_BA_error=float(max(four_errors))))
            folds=pd.read_csv(pred.with_name(f+'_fold_validation.csv')).fillna('')
            safe=folds[~folds.protocol.eq('filecv')]
            check(f+' evaluation source separation',safe.source_overlap.eq(0).all(),dict(source_safe_folds=len(safe),overlap_max=int(safe.source_overlap.max())))
        else:check(f+' prediction coverage',False,'file absent')
        for prefix in ['03_retrieval_summary','03_source_excluded_fit_summary']:
            p=out/'five_analyses'/f/'tables'/(prefix+'.csv')
            check(f+' '+prefix,p.exists() and p.stat().st_size>100,'saved summary')
        qp=out/'five_analyses'/f/'tables/03_source_excluded_fit_per_query.csv'
        if qp.exists():
            q=pd.read_csv(qp,dtype={'cut_id':str});q=q[q.feature.eq(f)]
            sr=pd.read_csv(out/'five_analyses'/f/'tables/03_source_excluded_fit_summary.csv')
            errors=[];unique=not q.duplicated(['feature','condition','cut_id']).any();neighbours_safe=True
            byid=m.set_index('cut_id');neighbour_cols=[c for c in q if c.startswith('neighbor_')]
            for condition,dq in q.groupby('condition'):
                r=sr[sr.feature.eq(f)&sr.condition.eq(condition)]
                value=dq.groupby('form').P5.mean().mean()
                errors.append(abs(value-r.iloc[0].P5_macro) if len(r)==1 else float('inf'))
                for col in neighbour_cols:
                    valid=dq[col].notna();nid=dq.loc[valid,col].astype(str)
                    neighbours_safe &= nid.isin(byid.index).all()
                    neighbours_safe &= np.array_equal(byid.loc[dq.loc[valid,'cut_id'],'g_site'].to_numpy(),dq.loc[valid,'source'].to_numpy())
                    if nid.isin(byid.index).all():neighbours_safe &= (byid.loc[nid,'g_site'].to_numpy()!=dq.loc[valid,'source'].to_numpy()).all()
            check(f+' strict retrieval matches queries and excludes their sources',unique and neighbours_safe and max(errors)<1e-12,
                  dict(unique_queries=unique,all_neighbour_sources_differ=bool(neighbours_safe),max_P5_error=float(max(errors))))
        else:check(f+' strict retrieval matches queries and excludes their sources',False,'queries absent')
        stab=out/'five_analyses'/f/'tables/05_stability_summary.csv'
        cp=list((out/'five_analyses'/f/'checkpoints').glob('*_stability.json'))
        removals=[json.loads(p.read_text())['row']['removed_source'] for p in cp]
        ok=stab.exists() and set(removals)==set(dev.g_site.unique()) and len(removals)==29
        if stab.exists():ok=ok and int(pd.read_csv(stab).iloc[0].source_removals)==29
        check(f+' source removal stability',ok,dict(saved_removals=len(removals),expected=29))
        p=out/'stage2'/(f+'_probe_comparison.csv')
        ok=p.exists() and len(pd.read_csv(p))==44
        check(f+' sampling performance',ok,'expected 44 records across four cohorts and two tasks')
        p=out/'stage2'/(f+'_structure_comparison.csv')
        check(f+' sampling structure',p.exists() and len(pd.read_csv(p))==7,'seven cohorts/weighting conditions')
    for name,expected in [('04_ablation_probes',15),('02_source_concentration',16),('01_task_protocol_scores',48)]:
        p=out/'five_analyses/tables'/(name+'.csv')
        check(name,p.exists() and len(pd.read_csv(p))==expected,dict(expected_rows=expected))
    for f in ['ast_zero','ast_tile']:
        p=out/'provenance'/(f+'_cache_equivalence.json')
        q=json.loads(p.read_text()) if p.exists() else []
        check(f+' frozen vector compatibility',len(q)>=12 and all(r['passed'] for r in q),dict(calibration_cuts=len(q)))
    shape_dir=out/'five_analyses/physical_strict'
    p=shape_dir/'validation.json'
    shape=json.loads(p.read_text()) if p.exists() else {}
    digests=shape.get('output_sha256',{})
    tied_to_outputs=len(digests)==3 and all(
        (shape_dir/'tables'/name).exists() and
        hashlib.sha256((shape_dir/'tables'/name).read_bytes()).hexdigest()==digest
        for name,digest in digests.items())
    check('exploratory strict shape retrieval',
          shape.get('complete') is True and shape.get('dimensions')==13 and
          shape.get('feature_indices')==list(range(38,51)) and
          shape.get('baseline_identical_to_main_strict_retrieval') is True and
          shape.get('all_neighbor_sources_differ') is True and
          shape.get('matched_neighbors_same_fs_and_within_factor_two_duration') is True and
          shape.get('query_counts')=={'meta / cross_source':2986,'meta / cross_source_matched':2911,
                                   'shape / cross_source':2986,'shape / cross_source_matched':2911} and
          tied_to_outputs,
          dict(exploratory_followup=True,validated_outputs_match_checksums=bool(tied_to_outputs),
               query_counts=shape.get('query_counts',{})))
    report=dict(complete=all(r['passed'] for r in records),eligible_main=10694,features=FEATURES,
                original_test_sources_previously_seen=True,label_basis='metadata-note weak labels',
                efficientnet_b3='old main-core only; exact original preprocessing unavailable',checks=records)
    target=out/'provenance/final_validation.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'complete':report['complete'],'passed':sum(r['passed'] for r in records),'checks':len(records),
                      'missing':[r['check'] for r in records if not r['passed']]},ensure_ascii=False),flush=True)
    if not report['complete'] and not args.allow_incomplete:raise SystemExit(2)

if __name__=='__main__':main()
