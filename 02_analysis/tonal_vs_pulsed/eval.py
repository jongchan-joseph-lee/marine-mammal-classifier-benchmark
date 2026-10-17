import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
F = pd.read_csv('tp_feats.csv', index_col=0)
d = pd.read_csv('/home/claude/pkg_in/data/bench_manifest.csv', low_memory=False).set_index('cut_id')
F = F.join(d[['task_type_single', 'fs_family', 'group3', 'fold_all', 'g_site', 'dur_s']])
z = np.load('/home/claude/pkg_in/full_cohort/features/physv0.npz', allow_pickle=True)
P = pd.DataFrame(z['X'], index=z['cut_id'].astype(str), columns=list(z['names']))
T = {'whistle': 'tonal', 'moan': 'tonal', 'pulsed_call': 'pulsed', 'growl': 'pulsed'}
F['g'] = F.task_type_single.map(T)
NEW = ['mod_peak_db', 'mod_peak_hz', 'mod_depth', 'cpp_db', 'cpp_q', 'env_kurt_log']
def au(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    s = roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]); return s, ('pulsed↑' if s >= .5 else 'pulsed↓')
def show(title, A, B):
    print(f'\n## {title}  (n={len(A)} vs {len(B)})')
    for c in NEW:
        s, dirn = au(A[c].values, B[c].values); print(f'  {c:13s} AUROC {max(s,1-s):.3f} {dirn}  median {np.nanmedian(A[c]):.3g} vs {np.nanmedian(B[c]):.3g}')
tn, pu = F[F.g == 'tonal'], F[F.g == 'pulsed']
show('tonal vs pulsed (all)', tn, pu)
show('whistle vs pulsed call', F[F.task_type_single == 'whistle'], F[F.task_type_single == 'pulsed_call'])
show('moan vs growl', F[F.task_type_single == 'moan'], F[F.task_type_single == 'growl'])
show('moan vs pulsed call', F[F.task_type_single == 'moan'], F[F.task_type_single == 'pulsed_call'])
for fam in ['20k', '40k']:
    show(f'tonal vs pulsed within {fam}', tn[tn.fs_family == fam], pu[pu.fs_family == fam])
show('impulsive vs rest (sanity)', F[F.g.notna()], F[F.task_type_single.isin(['click', 'pulse_train'])])
# source-disjoint linear probe (same as deck)
S = F[F.g.notna()].copy(); y = (S.g == 'pulsed').astype(int).values; folds = S.fold_all.values
meta = np.c_[np.log(S.fs.values), np.log(S.dur_s.values)]
phys = P.loc[S.index].values; new = S[NEW].values
def probe(X):
    pred = np.zeros(len(y))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if len(np.unique(y[tr])) < 2: continue
        k = min(64, X.shape[1])
        clf = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), FunctionTransformer(lambda a: np.clip(a, -8, 8)),
                            PCA(n_components=min(k, tr.sum() - 1)), LogisticRegression(C=1, class_weight='balanced', max_iter=2000))
        clf.fit(X[tr], y[tr]); pred[te] = clf.predict(X[te])
    return balanced_accuracy_score(y, pred)
print('\n## source-disjoint 7-fold linear probe, tonal vs pulsed BA')
for name, X in [('meta baseline', meta), ('phys51', phys), ('new 6', new), ('phys51 + new 6', np.c_[phys, new])]:
    print(f'  {name:15s} {probe(X):.3f}')
print('\n## within 40 kHz family, source-disjoint probe BA')
keep = (S.fs_family == '40k').values
y0, f0 = y.copy(), folds.copy()
y, folds = y0[keep], f0[keep]
for name, X in [('meta baseline', meta[keep]), ('phys51', phys[keep]), ('new 6', new[keep]), ('phys51 + new 6', np.c_[phys, new][keep])]:
    print(f'  {name:15s} {probe(X):.3f}  (n={keep.sum()}, pulsed={y.sum()}, sources={S.g_site[keep].nunique()})')
print('pulsed species in 40k:', S[keep & (S.g=="pulsed").values].index.map(lambda c: d.loc[c,"species_en"]).value_counts().head(5).to_dict())
