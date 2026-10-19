"""Pinniped recall per recording source, frozen probe vs fine-tuned (3 epoch). Run after train_ft.py / eval_ft.py."""
import json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
d = pd.read_csv('labels_3539.csv', dtype={'cut_id': str}); CL = ['Mysticeti', 'Odontoceti', 'Pinnipedia']; y = d.tx.map(CL.index).to_numpy()
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(d)), y, d.site))
Z = np.load('z_frozen.npz'); X = np.stack([Z[c] for c in d.cut_id]); p0 = np.zeros(len(d), int)
for tr, ev in folds:
    sc = StandardScaler().fit(X[tr]); pca = PCA(64, random_state=0).fit(np.clip(sc.transform(X[tr]), -8, 8)); f = lambda A: pca.transform(np.clip(sc.transform(A), -8, 8))
    p0[ev] = LogisticRegression(max_iter=3000, class_weight='balanced').fit(f(X[tr]), y[tr]).predict(f(X[ev]))
P = np.full((len(d), 3), np.nan)
for k in range(5):
    a = np.load(f'oof_ft_fold{k}.npy'); P[a[:, 0].astype(int)] = a[:, 1:]
p3 = P.argmax(1); m = y == 2
rows = []
for s in d[m].site.value_counts().index:
    i = np.flatnonzero((d.site == s).to_numpy() & m)
    rows.append((s, len(i), '/'.join(sorted(d.sp[i].unique())), float((p0[i] == 2).mean()), float((p3[i] == 2).mean())))
json.dump(dict(rows=rows), open('ast_finetune_pinniped_by_source.json', 'w'), ensure_ascii=False, indent=1)
for r in rows: print(r)
