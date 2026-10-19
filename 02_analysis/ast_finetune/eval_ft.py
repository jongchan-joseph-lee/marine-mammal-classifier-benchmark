import json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, accuracy_score
d = pd.read_csv('labels_3539.csv', dtype={'cut_id': str})
CL = ['Mysticeti', 'Odontoceti', 'Pinnipedia']; y = d.tx.map(CL.index).to_numpy(); sites = d.site.to_numpy()
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(d)), y, d.site))
Z = np.load('z_frozen.npz'); X = np.stack([Z[c] for c in d.cut_id])
P0 = np.zeros((len(d), 3))
for tr, ev in folds:
    sc = StandardScaler().fit(X[tr]); pca = PCA(min(64, len(tr) - 1), random_state=0).fit(np.clip(sc.transform(X[tr]), -8, 8))
    f = lambda A: pca.transform(np.clip(sc.transform(A), -8, 8))
    clf = LogisticRegression(max_iter=3000, class_weight='balanced').fit(f(X[tr]), y[tr])
    P0[ev] = clf.predict_proba(f(X[ev]))
def load(prefix):
    P = np.full((len(d), 3), np.nan)
    for k in range(5):
        a = np.load(f'{prefix}_fold{k}.npy'); P[a[:, 0].astype(int)] = a[:, 1:]
    return P.argmax(1)
p0 = P0.argmax(1); p1 = load('oof_ft'); p1e = load('oof_ft1')
ba = lambda yt, yp: balanced_accuracy_score(yt, yp)
res = dict(n=len(d), n_sources=int(len(set(sites))), frozen=ba(y, p0), finetune=ba(y, p1), finetune1=ba(y, p1e),
           per_fold=[dict(frozen=ba(y[ev], p0[ev]), finetune=ba(y[ev], p1[ev]), finetune1=ba(y[ev], p1e[ev]), n=len(ev), n_src=int(len(set(sites[ev])))) for tr, ev in folds],
           recall_frozen=recall_score(y, p0, average=None).tolist(), recall_ft=recall_score(y, p1, average=None).tolist(),
           recall_ft1=recall_score(y, p1e, average=None).tolist(), classes=CL)
rng = np.random.default_rng(0); us = np.unique(sites); idx = {s: np.flatnonzero(sites == s) for s in us}
B0, B1, D, B2, D2, DA, DR = [], [], [], [], [], [], []
for _ in range(2000):
    take = np.concatenate([idx[s] for s in rng.choice(us, len(us))])
    if len(set(y[take])) < 3: continue
    b0, b1, b2 = ba(y[take], p0[take]), ba(y[take], p1[take]), ba(y[take], p1e[take]); B0.append(b0); B1.append(b1); D.append(b1 - b0); B2.append(b2); D2.append(b2 - b0)
    DA.append(accuracy_score(y[take], p1[take]) - accuracy_score(y[take], p0[take]))
    DR.append(recall_score(y[take], p1[take], average=None, labels=[0, 1, 2]) - recall_score(y[take], p0[take], average=None, labels=[0, 1, 2]))
q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
res.update(ci_frozen=q(B0), ci_ft=q(B1), diff=res['finetune'] - res['frozen'], ci_diff=q(D), ci_ft1=q(B2), diff1=res['finetune1'] - res['frozen'], ci_diff1=q(D2))
DR = np.array(DR)
a0, a1 = accuracy_score(y, p0), accuracy_score(y, p1)
res.update(acc_frozen=a0, acc_ft=a1, d_acc=[a1 - a0] + q(DA),
           d_rM=[float(DR[:, 0].mean())] + q(DR[:, 0]), d_rO=[float(DR[:, 1].mean())] + q(DR[:, 1]), d_rP=[float(DR[:, 2].mean())] + q(DR[:, 2]))
json.dump(res, open('ast_finetune_results.json', 'w'), indent=2, ensure_ascii=False); print(json.dumps(res, indent=1, ensure_ascii=False))
