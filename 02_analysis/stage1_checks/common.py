import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
FD='/home/claude/pkg_in/work/features'
D=pd.read_csv('/home/claude/pkg_in/data/bench_manifest.csv',low_memory=False); D['cut_id']=D.cut_id.astype(str); D=D.set_index('cut_id')
IDS=np.load(f'{FD}/physv0.npz',allow_pickle=True)['cut_id'].astype(str)
M=D.loc[IDS].copy()
FEATS=['meta','physv0','mfcc16k','vggish','effb3','ast_tile']
def load(f):
    z=np.load(f'{FD}/{f}.npz',allow_pickle=True); assert (z['cut_id'].astype(str)==IDS).all(); return z['X'].astype(float), [str(n) for n in z['names']]
X={f:load(f)[0] for f in FEATS}
PHYS_NAMES=load('physv0')[1]
def clip8(a): return np.clip(a,-8,8)
def model(dim, pca=True):
    steps=[SimpleImputer(strategy='median'),StandardScaler(),FunctionTransformer(clip8)]
    if pca: steps.append(PCA(n_components=min(64,dim)))
    steps.append(LogisticRegression(C=1.0,class_weight='balanced',max_iter=3000))
    return make_pipeline(*steps)
def cv_predict(Xm, y, folds, pca=True):
    pred=np.array([None]*len(y),dtype=object)
    for f in np.unique(folds):
        tr,te=folds!=f,folds==f
        m=model(Xm.shape[1],pca); m.fit(Xm[tr],y[tr]); pred[te]=m.predict(Xm[te])
    return pred
def ba(y,p): return balanced_accuracy_score(y,p)
def boot(y, pa, pb, groups, n=1000, seed=0):
    """source bootstrap: BA of a, and diff a-b CI"""
    rng=np.random.default_rng(seed); g=np.unique(groups); idx={k:np.flatnonzero(groups==k) for k in g}
    A=[];Dd=[]
    for _ in range(n):
        s=np.concatenate([idx[k] for k in rng.choice(g,len(g),replace=True)])
        try:
            a=ba(y[s],pa[s]); A.append(a)
            if pb is not None: Dd.append(a-ba(y[s],pb[s]))
        except Exception: pass
    out=dict(ba=ba(y,pa),lo=np.percentile(A,2.5),hi=np.percentile(A,97.5))
    if pb is not None: out.update(d=ba(y,pa)-ba(y,pb),dlo=np.percentile(Dd,2.5),dhi=np.percentile(Dd,97.5))
    return out
TASKS={'species':'task_species','group3':'task_group3','type7':'task_type_single'}
def task_mask(t):
    c=M[TASKS[t]]
    m=c.notna()&(c.astype(str)!='')&(c.astype(str)!='nan')
    if t=='type7': m&=c.isin(['whistle','moan','click','pulse_train','pulsed_call','growl','song'])
    return m.values
