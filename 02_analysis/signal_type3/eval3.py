#!/usr/bin/env python3
"""eval3.py — 발성 유형을 물리적 신호 유형 3개(tonal / impulsive / pulsed)로 다시 묶어 평가.
   평가 규약은 Bench v1 의 b07 과 같다: 출처 단위 7겹(fold_all), 겹 안에서 중앙값 대치 → 표준화 → ±8 자름 → PCA ≤64
   → 로지스틱 회귀(C=1, class_weight=balanced). 구간은 출처 단위 부트스트랩 2,000회."""
import sys, json
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
PKG = sys.argv[1]
FEATS = ["meta", "physv0", "mfcc16k", "vggish", "ast_zero", "ast_tile", "effb3"]
GROUP = {"whistle": "tonal", "moan": "tonal", "click": "impulsive", "pulse_train": "impulsive", "pulsed_call": "pulsed", "growl": "pulsed"}
rng = np.random.default_rng(0)
m = pd.read_csv(f"{PKG}/data/core_3539_metadata.csv", dtype={"cut_id": str}, low_memory=False)
X = {}
for f in FEATS:
    z = np.load(f"{PKG}/work/features/{f}.npz", allow_pickle=False)
    assert (z["cut_id"] == m.cut_id.to_numpy()).all()
    X[f] = np.asarray(z["X"], dtype=np.float64)

def fit_predict(Xtr, ytr, Xte):
    med = np.nanmedian(Xtr, 0); med = np.where(np.isfinite(med), med, 0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med); Xte = np.where(np.isfinite(Xte), Xte, med)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-12] = 1
    Xtr = np.clip((Xtr - mu) / sd, -8, 8); Xte = np.clip((Xte - mu) / sd, -8, 8)
    k = min(64, Xtr.shape[1], Xtr.shape[0] - 1)
    p = PCA(k, random_state=0).fit(Xtr)
    return LogisticRegression(C=1, class_weight="balanced", max_iter=3000).fit(p.transform(Xtr), ytr).predict(p.transform(Xte))

def ba(y, p, classes):
    r = [np.mean(p[y == c] == c) for c in classes if (y == c).any()]
    return float(np.mean(r))

def run(rows, y, folds):
    """rows: 표본 안의 행 번호, y: 라벨, folds: 겹 번호 → 특징별 예측"""
    out = {}
    for f in FEATS:
        pred = np.empty(len(rows), dtype=object)
        for k in np.unique(folds):
            te = folds == k
            if len(np.unique(y[~te])) < 2: pred[te] = y[~te][0]; continue
            pred[te] = fit_predict(X[f][rows[~te]], y[~te], X[f][rows[te]])
        out[f] = pred
    return out

def summarize(y, preds, site, mask=None, B=2000):
    classes = np.unique(y)
    if mask is None: mask = np.ones(len(y), bool)
    y, site = y[mask], site[mask]; preds = {f: p[mask] for f, p in preds.items()}
    sites = np.unique(site); idx = {s: np.where(site == s)[0] for s in sites}
    boot = {f: [] for f in preds}
    for _ in range(B):
        ii = np.concatenate([idx[s] for s in rng.choice(sites, len(sites))])
        for f in preds: boot[f].append(ba(y[ii], preds[f][ii], classes))
    res = {}
    for f in preds:
        b, bm = np.array(boot[f]), np.array(boot["meta"])
        res[f] = dict(ba=ba(y, preds[f], classes), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                      d=ba(y, preds[f], classes) - ba(y, preds["meta"], classes),
                      dlo=float(np.percentile(b - bm, 2.5)), dhi=float(np.percentile(b - bm, 97.5)),
                      recall={c: float(np.mean(preds[f][y == c] == c)) for c in classes})
    return dict(n=int(len(y)), sites=int(len(sites)), classes={c: int((y == c).sum()) for c in classes}, chance=1 / len(classes), feat=res)

def show(title, s):
    print(f"\n## {title}: {s['n']}컷, 출처 {s['sites']}, 범주 {s['classes']}, 우연 {s['chance']:.2f}")
    for f, r in s["feat"].items():
        print(f"  {f:9s} BA {r['ba']:.3f} [{r['lo']:.2f},{r['hi']:.2f}]  기준선 대비 {r['d']:+.3f} [{r['dlo']:+.2f},{r['dhi']:+.2f}]  " + " ".join(f"{c}:{v:.2f}" for c, v in r["recall"].items()))

OUT = {}
t = m[m.type_single.notna()].copy(); t["g3"] = t.type_single.map(GROUP)
site_all = m.g_site.to_numpy()
# (0) 구현 확인: 기존 7유형, 출처 단위 7겹 (b07 의 allcv 값: meta 0.411, physv0 0.410, ast_tile 0.411, effb3 0.478)
t7 = t[t.type_single != "bark"]
p7 = run(t7.index.to_numpy(), t7.type_single.to_numpy(), t7.fold_all.to_numpy()); OUT["7_site"] = summarize(t7.type_single.to_numpy(), p7, t7.g_site.to_numpy()); show("7유형 · 출처 단위 분할 (구현 확인)", OUT["7_site"])
# (1) 3묶음, 출처 단위 분할
t3 = t[t.g3.notna()]; r3, y3, s3 = t3.index.to_numpy(), t3.g3.to_numpy(), t3.g_site.to_numpy()
p3 = run(r3, y3, t3.fold_all.to_numpy()); OUT["3_site"] = summarize(y3, p3, s3); show("3묶음 · 출처 단위 분할", OUT["3_site"])
# (2) 3묶음, 무작위 분할 (문헌 방식)
fr = np.zeros(len(t3), int)
for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(r3, y3)): fr[te] = k
pr = run(r3, y3, fr); OUT["3_random"] = summarize(y3, pr, s3); show("3묶음 · 무작위 분할", OUT["3_random"])
# (3) 같은 샘플링률대 안에서 (학습은 전체, 평가만 부분집합)
for fam in ["20k", "40k", "80k"]:
    mk = (t3.fs_family == fam).to_numpy()
    if mk.sum() > 50 and len(np.unique(s3[mk])) >= 4:
        OUT[f"3_site_fs{fam}"] = summarize(y3, p3, s3, mk); show(f"3묶음 · 출처 단위 · fs {fam} 계열만", OUT[f"3_site_fs{fam}"])
# (4) 묶으면 무엇을 잃는가: 묶음 안의 두 유형을 가르는 과제 (출처 단위)
for a, b in [("whistle", "moan"), ("click", "pulse_train"), ("pulsed_call", "growl")]:
    tt = t[t.type_single.isin([a, b])]
    pp = run(tt.index.to_numpy(), tt.type_single.to_numpy(), tt.fold_all.to_numpy())
    OUT[f"pair_{a}_{b}"] = summarize(tt.type_single.to_numpy(), pp, tt.g_site.to_numpy()); show(f"묶음 안 구분: {a} 대 {b}", OUT[f"pair_{a}_{b}"])
# 유형 × 분류군, 유형 × fs
print(pd.crosstab(t3.g3, t3.group3)); print(pd.crosstab(t3.g3, t3.fs_family)); print(t3.groupby("g3").g_site.nunique())
json.dump(OUT, open("regroup3.json", "w"), ensure_ascii=False, indent=1)
