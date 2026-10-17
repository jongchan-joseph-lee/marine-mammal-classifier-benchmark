"""Shared full-cohort definitions and exact, memory-bounded neighbours."""
from pathlib import Path
import importlib.util
import json
import hashlib
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
OUT=ROOT/'output'
SEED=20261004
FEATURES=['meta','mfcc16k','physv0','vggish','ast_zero','ast_tile']
FORMS=['click','growl','moan','pulse_train','pulsed_call','song','whistle']
COMMON=['click','pulse_train','pulsed_call','whistle']
spec=importlib.util.spec_from_file_location('original_review',BASE/'deep_review/scripts/run_review.py')
rr=importlib.util.module_from_spec(spec);spec.loader.exec_module(rr)


def log(s): print(f'[{time.strftime("%H:%M:%S")}] {s}',flush=True)


def metadata():
    m=pd.read_csv(OUT/'provenance/main_full_metadata.csv',dtype={'cut_id':str},low_memory=False).fillna('')
    assert len(m)==10694 and m.cut_id.is_unique
    for c in ['fold','fold_all','fold_region','fs_hz']:
        m[c]=pd.to_numeric(m[c],errors='raise').astype(int)
    dev=m[m.split.eq('dev')];test=m[m.split.eq('test')]
    assert len(dev)==8592 and len(test)==2102
    assert set(dev.g_site).isdisjoint(set(test.g_site))
    for col in ['fold_all','fold']:
        d=m if col=='fold_all' else dev
        assert d.groupby('g_site')[col].nunique().max()==1
    return m


def features(names=FEATURES,wait=False):
    m=metadata();ids=m.cut_id.to_numpy(dtype=str);arrays={};feature_names={};infos={}
    for f in names:
        p=OUT/'features'/(f+'.npz')
        while wait and not p.exists(): time.sleep(2)
        if not p.exists(): raise FileNotFoundError(f'Full-cohort feature incomplete: {f}')
        z=np.load(p,allow_pickle=False)
        assert np.array_equal(z['cut_id'].astype(str),ids)
        assert z['X'].shape[0]==len(m) and not np.isinf(z['X']).any()
        arrays[f]=z['X'];feature_names[f]=z['names'].astype(str)
        infos[f]=json.loads(str(z['info'].item()))
    return arrays,feature_names,infos


def stable_topk(D,k):
    """Exact distance/index ordering, including ties at the kth boundary."""
    ix=np.argpartition(D,kth=k-1,axis=1)[:,:k]
    distances=np.take_along_axis(D,ix,axis=1)
    thresholds=distances.max(1)
    tie_rows=np.flatnonzero(np.sum(D<=thresholds[:,None],axis=1)>k)
    for r in tie_rows:
        left=np.flatnonzero(D[r]<thresholds[r])
        equal=np.flatnonzero(D[r]==thresholds[r])
        ix[r]=np.r_[left,equal[:k-len(left)]]
    selected=np.take_along_axis(D,ix,axis=1)
    order=np.lexsort((ix,selected),axis=1)
    return np.take_along_axis(ix,order,axis=1)


def nearest10(Z,block=512):
    n=len(Z);answer=np.empty((n,10),dtype=np.int32)
    for start in range(0,n,block):
        end=min(n,start+block)
        D=cdist(Z[start:end],Z,'sqeuclidean')
        D[np.arange(end-start),np.arange(start,end)]=np.inf
        answer[start:end]=stable_topk(D,10)
    return answer


def verify_neighbours():
    rng=np.random.default_rng(SEED)
    for Z in [rng.normal(size=(173,12)),np.zeros((173,2)),np.r_[np.zeros((80,2)),rng.normal(size=(93,2))]]:
        D=cdist(Z,Z,'sqeuclidean');np.fill_diagonal(D,np.inf)
        expected=np.argsort(D,axis=1,kind='stable')[:,:10]
        assert np.array_equal(expected,nearest10(Z,block=31))
    return True


def configure_original(out,names):
    rr.OUT=out;rr.TAB=out/'tables';rr.FEATURES=list(names)
    rr.TAB.mkdir(parents=True,exist_ok=True);(out/'methods').mkdir(parents=True,exist_ok=True)
    rr.nearest10=nearest10


class FullContext:
    def __init__(self,names):
        self.core=metadata();self.dev=self.core[self.core.split.eq('dev')].reset_index(drop=True)
        self.typ=self.dev[self.dev.type_single.isin(FORMS)].reset_index(drop=True)
        self.idx=pd.Index(self.dev.cut_id).get_indexer(self.typ.cut_id)
        X,self.names,self.infos=features(names)
        di=pd.Index(self.core.cut_id).get_indexer(self.dev.cut_id)
        self.X={f:a[di] for f,a in X.items()}
        self.y=self.typ.type_single.to_numpy();self.sites=self.typ.g_site.to_numpy()
        self.folds=self.dev.fold.to_numpy();self.tf=self.typ.fold.to_numpy()
        self.pred=None

    def original_predictions(self,task,protocol,feature,ids=None):
        # Retains the original helper API name. These are NEW full-cohort
        # predictions, never the old 3539-cut predictions.
        t=self.pred.query('task==@task and protocol==@protocol and feature==@feature').set_index('cut_id')
        if ids is not None:t=t.loc[ids]
        return t


def write_scope():
    m=metadata()
    scope=dict(n_inventory=13693,n_main=len(m),n_dev=int(m.split.eq('dev').sum()),n_test=int(m.split.eq('test').sum()),
               sources=m.g_site.nunique(),species=m.species.nunique(),full_features_expected=FEATURES,
               form_metrics_classes=FORMS,description_forms_include_bark=True,
               excluded_bark_from_comparative_probe_n=int(m.type_single.eq('bark').sum()),
               n_single_seven_forms=int(m.type_single.isin(FORMS).sum()),
               n_no_single_form=int(m.type_single.eq('').sum()),
               fixed_source_split=True,maximum_analysis_seconds=120,partially_used_long_cuts=int((m.dur_s>120).sum()),
               old_core_arrays_reused=True,cuda_available=False,
               source_unit='g_site metadata proxy; not verified individual/device/session',
               evaluation='Retrospective reanalysis; previous test-source results were already seen.',
               effb3=dict(status='core_only',n_main_cached=3539,
                          reason='Frozen vectors available; exact original image-preprocessing code is not present. Do not fabricate an extension.'))
    (OUT/'provenance/study_scope.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2))
    return scope
