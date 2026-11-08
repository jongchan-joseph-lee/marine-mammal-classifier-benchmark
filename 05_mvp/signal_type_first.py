"""Signal-type-first taxon classifier (MVP).
신호 유형 우선 분류군 분류기 — 최소 동작판.

Pipeline (단계)
  1. measured physical attributes per clip (클립별 물리량; results/attributes_3539.csv)
  2. signal-type model: tonal / impulsive / pulsed from shape attributes only (모양 물리량 → 신호 유형 확률)
  3. taxon model: signal-type probabilities + frequency band (f50, observed band) → taxon probability
     (신호 유형 확률 + 주파수 대역 → 분류군 확률)
  4. abstain: low-confidence clips are routed to review (저신뢰 클립은 검토로 넘김)

Evaluation (평가)
  recording-source-disjoint CV: sites split into 7 folds, repeated over 20 random site partitions;
  site-bootstrap 95% CI on the first partition. Baselines use the same folds.

Usage:  python 05_mvp/signal_type_first.py   →  results/mvp_signal_type_first.json
"""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = pathlib.Path(__file__).resolve().parents[1]
A = pd.read_csv(ROOT / "results/attributes_3539.csv", low_memory=False).rename(columns={"Unnamed: 0": "cut_id"})

SHAPE = ["imp_frac", "imp_kurt", "rep_ici_ms", "rep_prr_hz", "rep_prr_prom", "harm", "harm_spacing_hz",
         "fpk_over_spacing", "tonal_frac", "contour_cont", "fm_oct_s", "env_crest"]
BAND = ["f50_khz", "band_obs_khz"]
GROUP = {"whistle": "tonal", "moan": "tonal", "click": "impulsive", "pulse_train": "impulsive",
         "pulsed_call": "pulsed", "growl": "pulsed"}
TYPES = ["tonal", "impulsive", "pulsed"]
TAXA = ["Mysticeti", "Odontoceti", "Pinnipedia"]
K, REPEATS, BOOT = 7, 20, 2000

A["type3"] = A["type"].map(GROUP)
for c in BAND: A["log_" + c] = np.log10(A[c].clip(lower=1e-3))
A["log_fs"] = np.log10(A["fs"])
y = A["tx"].to_numpy()
site = A["site"].to_numpy()


def lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))


def folds_for(seed, sites, k=K):
    rng = np.random.default_rng(seed)
    u = rng.permutation(np.unique(sites))
    f = {s: i % k for i, s in enumerate(u)}
    return np.array([f[s] for s in sites])


def type_probs(tr, te):
    """stage 1: fit on training clips with a single signal-type label; out-of-fold probs for training rows."""
    lab = A["type3"].notna().to_numpy()
    X = A[SHAPE].to_numpy(float)
    m = lr().fit(X[tr & lab], A.loc[tr & lab, "type3"])
    p_te = m.predict_proba(X[te])
    # out-of-fold probabilities inside the training sites (no leakage into stage 2)
    p_tr = np.zeros((tr.sum(), len(m.classes_)))
    inner = folds_for(123, site[tr], k=5)
    idx = np.where(tr)[0]
    for j in range(5):
        a, b = idx[inner != j], idx[inner == j]
        mm = lr().fit(X[a][lab[a]], A["type3"].to_numpy()[a][lab[a]])
        p_tr[inner == j] = mm.predict_proba(X[b])
    order = [list(m.classes_).index(t) for t in TYPES]
    return p_tr[:, order], p_te[:, order]


def run(fold):
    P = {k: np.zeros((len(A), 3)) for k in ["meta_fs", "band_only", "flat_physical", "mvp", "mvp_shape"]}
    Xs = {"meta_fs": A[["log_fs"]].to_numpy(float),
          "band_only": A[["log_f50_khz", "log_band_obs_khz"]].to_numpy(float),
          "flat_physical": A[SHAPE + ["log_f50_khz", "log_band_obs_khz"]].to_numpy(float)}
    for f in range(K):
        te, tr = fold == f, fold != f
        for k, X in Xs.items():
            m = lr().fit(X[tr], y[tr]); P[k][te] = m.predict_proba(X[te])[:, [list(m.classes_).index(t) for t in TAXA]]
        pt_tr, pt_te = type_probs(tr, te)
        band = A[["log_f50_khz", "log_band_obs_khz"]].to_numpy(float)
        Z_tr, Z_te = np.hstack([pt_tr, band[tr]]), np.hstack([pt_te, band[te]])
        m = lr().fit(Z_tr, y[tr]); P["mvp"][te] = m.predict_proba(Z_te)[:, [list(m.classes_).index(t) for t in TAXA]]
        S = A[SHAPE].to_numpy(float)
        Z_tr, Z_te = np.hstack([pt_tr, band[tr], S[tr]]), np.hstack([pt_te, band[te], S[te]])
        m = lr().fit(Z_tr, y[tr]); P["mvp_shape"][te] = m.predict_proba(Z_te)[:, [list(m.classes_).index(t) for t in TAXA]]
    return P


def bal_acc(yt, pred):
    return float(np.mean([np.mean(pred[yt == c] == c) for c in TAXA if (yt == c).any()]))


def boot_ci(yt, pred, sites, seed=0):
    rng = np.random.default_rng(seed); us = np.unique(sites); idx = {s: np.where(sites == s)[0] for s in us}
    v = [bal_acc(yt[i], pred[i]) for i in (np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(BOOT))]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


out = {"n_clips": int(len(A)), "n_sites": int(len(np.unique(site))), "folds": K, "repeats": REPEATS,
       "classes": {t: int((y == t).sum()) for t in TAXA}, "models": {}}
reps = {}
for r in range(REPEATS):
    P = run(folds_for(r, site))
    for k, p in P.items():
        reps.setdefault(k, []).append(bal_acc(y, np.array(TAXA)[p.argmax(1)]))
    if r == 0:
        P0 = P
for k, p in P0.items():
    pred = np.array(TAXA)[p.argmax(1)]
    conf = p.max(1)
    cov = {}
    for q in [1.0, 0.9, 0.8, 0.7, 0.5]:
        thr = np.quantile(conf, 1 - q) if q < 1 else -1
        keep = conf >= thr
        cov[str(q)] = {"ba": bal_acc(y[keep], pred[keep]), "n": int(keep.sum())}
    out["models"][k] = {"ba_partition0": bal_acc(y, pred), "ci95_partition0": boot_ci(y, pred, site),
                        "ba_mean_over_partitions": float(np.mean(reps[k])),
                        "ba_range_over_partitions": [float(np.min(reps[k])), float(np.max(reps[k]))],
                        "recall": {t: float(np.mean(pred[y == t] == t)) for t in TAXA},
                        "coverage": cov}
# paired site-bootstrap difference against the sampling-rate-only baseline (표본 출처 단위 짝 부트스트랩)
rng = np.random.default_rng(1); us = np.unique(site); idx = {s: np.where(site == s)[0] for s in us}
pm = np.array(TAXA)[P0["meta_fs"].argmax(1)]
for k, p in P0.items():
    pr = np.array(TAXA)[p.argmax(1)]; d = []
    for _ in range(BOOT):
        i = np.concatenate([idx[s] for s in rng.choice(us, len(us))]); d.append(bal_acc(y[i], pr[i]) - bal_acc(y[i], pm[i]))
    out["models"][k]["diff_vs_meta_fs"] = {"d": bal_acc(y, pr) - bal_acc(y, pm), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
# sampling-rate control: clips recorded at fs >= 40 kHz (Odontoceti vs Pinnipedia; 표본률 통제)
hi = (A["fs"] >= 40000).to_numpy() & np.isin(y, ["Odontoceti", "Pinnipedia"])
out["fs40k_subset"] = {"n": int(hi.sum()), "sites": int(len(np.unique(site[hi]))), "classes": {t: int((y[hi] == t).sum()) for t in ["Odontoceti", "Pinnipedia"]}, "ba": {}}
for k, p in P0.items():
    q = p[hi][:, [1, 2]]; pr = np.array(["Odontoceti", "Pinnipedia"])[q.argmax(1)]
    out["fs40k_subset"]["ba"][k] = float(np.mean([np.mean(pr[y[hi] == c] == c) for c in ["Odontoceti", "Pinnipedia"]]))
# stage-1 check: signal-type accuracy under the same folds (partition 0)
lab = A["type3"].notna().to_numpy(); fold = folds_for(0, site); pt = np.zeros((len(A), 3))
for f in range(K):
    te, tr = fold == f, fold != f
    _, pt[te] = type_probs(tr, te)
tp = np.array(TYPES)[pt.argmax(1)]
yt = A["type3"].to_numpy()
out["stage1_signal_type_ba"] = float(np.mean([np.mean(tp[lab & (yt == t)] == t) for t in TYPES]))
out["stage1_n_labelled"] = int(lab.sum())
(ROOT / "results/mvp_signal_type_first.json").write_text(json.dumps(out, indent=2))
for k, v in out["models"].items():
    print(f"{k:14s} BA {v['ba_mean_over_partitions']:.3f} (range {v['ba_range_over_partitions'][0]:.3f}-{v['ba_range_over_partitions'][1]:.3f})  "
          f"p0 {v['ba_partition0']:.3f} CI {v['ci95_partition0'][0]:.2f}-{v['ci95_partition0'][1]:.2f}  "
          f"recall {' '.join(f'{t[:4]}:{x:.2f}' for t,x in v['recall'].items())}  cov70 {v['coverage']['0.7']['ba']:.3f}")
print("diff vs meta", {k: (round(v["diff_vs_meta_fs"]["d"],3), [round(x,3) for x in v["diff_vs_meta_fs"]["ci95"]]) for k,v in out["models"].items()})
print("fs>=40k", out["fs40k_subset"])
print("stage1 signal-type BA", round(out["stage1_signal_type_ba"], 3), "n", out["stage1_n_labelled"])
