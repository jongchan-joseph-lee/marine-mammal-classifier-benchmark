import numpy as np, pandas as pd, json, time, warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
import umap
d = pd.read_csv('/home/claude/pkg_in/data/bench_manifest.csv', low_memory=False).set_index('cut_id')
REPS = ['physv0', 'mfcc16k', 'vggish', 'effb3', 'ast_tile']
ids = np.load('/home/claude/pkg_in/work/features/physv0.npz', allow_pickle=True)['cut_id'].astype(str)
m = d.loc[ids]; sp = m.species.values; tx = m.group3.values; split = m.split.values
print('split', pd.Series(split).value_counts().to_dict(), flush=True)
out = {'ids': ids.tolist(), 'maps': {}, 'score': {}}
def knn_loo(Y, lab, k=15):
    clf = KNeighborsClassifier(n_neighbors=k + 1).fit(Y, lab)
    ind = clf.kneighbors(Y, return_distance=False)[:, 1:]
    pred = np.array([pd.Series(lab[r]).mode().iloc[0] for r in ind])
    return balanced_accuracy_score(lab, pred)
def q(Y):  # quantize to int16 range 0..9999 per axis
    lo, hi = Y.min(0), Y.max(0); return np.round((Y - lo) / (hi - lo + 1e-12) * 9999).astype(int).T.tolist()
for rep in REPS:
    z = np.load(f'/home/claude/pkg_in/work/features/{rep}.npz', allow_pickle=True)
    assert (z['cut_id'].astype(str) == ids).all()
    X = np.clip(StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(z['X'].astype(float))), -8, 8)
    X50 = PCA(n_components=min(50, X.shape[1]), random_state=0).fit_transform(X)
    res = {}
    t = time.time()
    res['pca'] = X50[:, :2]
    res['tsne'] = TSNE(n_components=2, perplexity=30, init='pca', random_state=0).fit_transform(X50)
    res['umap'] = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X50)
    spc = pd.Categorical(sp).codes
    res['umap_sup'] = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X50, y=spc)
    dev = split == 'dev'; te = ~dev
    u = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit(X50[dev], y=spc[dev])
    Y = np.zeros((len(ids), 2)); Y[dev] = u.embedding_; Y[te] = u.transform(X50[te]); res['umap_sup_dev'] = Y
    sc = {}
    for k, Y in res.items():
        sc[k] = {'species': round(knn_loo(Y, sp), 3), 'taxon': round(knn_loo(Y, tx), 3)}
    # dev->test generalisation on supervised dev map: kNN fit on dev, predict test (species present in dev)
    Y = res['umap_sup_dev']; ok = te & np.isin(sp, np.unique(sp[dev]))
    clf = KNeighborsClassifier(n_neighbors=15).fit(Y[dev], sp[dev])
    sc['umap_sup_dev']['test_species'] = round(balanced_accuracy_score(sp[ok], clf.predict(Y[ok])), 3)
    clf = KNeighborsClassifier(n_neighbors=15).fit(Y[dev], sp[dev]); sc['umap_sup_dev']['dev_species'] = round(knn_loo(Y[dev], sp[dev]), 3)
    Yu = res['umap']; clf = KNeighborsClassifier(n_neighbors=15).fit(Yu[dev], sp[dev]); sc['umap']['test_species_from_dev'] = round(balanced_accuracy_score(sp[ok], clf.predict(Yu[ok])), 3)
    out['maps'][rep] = {k: q(v) for k, v in res.items()}; out['score'][rep] = sc
    print(rep, round(time.time() - t), sc, flush=True)
json.dump(out, open('maps.json', 'w'))
print('done')
