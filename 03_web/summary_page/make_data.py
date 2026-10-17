#!/usr/bin/env python3
"""make_data.py — 요약 그림 6장의 숫자를 한 파일(summary.json)로 모은다. 새 학습은 하지 않는다(그림 C 의 값은 regroup/eval3.py 결과)."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
PKG, REF, RG = "/home/claude/pkg_in", "/home/claude/ref_v1/eval/reference_v1", "/home/claude/regroup/regroup3.json"
FE = ["meta", "physv0", "mfcc16k", "vggish", "ast_zero", "ast_tile", "effb3"]
out = {}
# A. 자료 구조 (평가 대상 10,694컷)
man = pd.read_csv(f"{PKG}/data/bench_manifest.csv", dtype={"cut_id": str}, low_memory=False)
main = pd.read_csv(f"{PKG}/data/main_10694_metadata.csv", dtype={"cut_id": str}, low_memory=False)
core = pd.read_csv(f"{PKG}/data/core_3539_metadata.csv", dtype={"cut_id": str}, low_memory=False)
ns = main.groupby("species_ko").g_site.nunique()
fam = ["5k", "10k", "20k", "40k", "80k", "160k"]
tx = [("Odontoceti", "이빨고래류"), ("Mysticeti", "수염고래류"), ("Pinnipedia", "기각류")]
out["A"] = dict(n_all=len(man), n_main=len(main), n_core=len(core), n_species_all=int(man.species.nunique()), n_species=int(main.species.nunique()),
                n_sites=int(main.g_site.nunique()),
                site_hist={k: int(v) for k, v in zip(["1", "2", "3", "4", "5+"], [(ns == 1).sum(), (ns == 2).sum(), (ns == 3).sum(), (ns == 4).sum(), (ns >= 5).sum()])},
                fs=dict(fam=fam, rows=[dict(name=ko, n=int((main.group3 == en).sum()), share=[float(((main.group3 == en) & (main.fs_family == f)).sum() / (main.group3 == en).sum()) for f in fam]) for en, ko in tx]),
                below16k={ko: float(((main.group3 == en) & (main.fs_hz < 16000)).sum() / (main.group3 == en).sum()) for en, ko in tx})
# B. 누수와 교란 (표본 3,539컷, 선형 분류기, 균형정확도)
r = pd.read_csv(f"{REF}/results.csv"); r = r[(r.subset == "all") & (r.metric == "balanced_acc") & r.feature.isin(FE)]
out["B"] = []
for task, name in [("species", "종 (10종)"), ("type_single", "발성 유형 (7유형)"), ("group3", "분류군 (3개)")]:
    d = {"task": name}
    for prot, key in [("filecv", "random"), ("allcv", "site")]:
        q = r[(r.task == task) & (r.protocol == prot)].set_index("feature")
        d[key] = {f: float(q.loc[f, "value"]) for f in FE}; d["chance"] = float(q.chance.iloc[0]); d["n"] = int(q.n_eval.iloc[0]); d["sites"] = int(q.n_sites.iloc[0])
    out["B"].append(d)
# C. 굵은 유형 3개
g = json.load(open(RG)); out["C"] = {k: g[k] for k in ["3_random", "3_site", "3_site_fs40k", "3_site_fs80k", "7_site"]}
out["C_pairs"] = {k: g[k] for k in g if k.startswith("pair_")}
# D. 물리량 하나의 분포
z = np.load(f"{PKG}/work/features/physv0.npz"); X = pd.DataFrame(z["X"], columns=[str(n) for n in z["names"]])
G = {"whistle": "tonal", "moan": "tonal", "click": "impulsive", "pulse_train": "impulsive", "pulsed_call": "pulsed", "growl": "pulsed"}
def hist(col, groups, lab, bins, tf=lambda v: v):
    res = []
    for name, mask in groups:
        v = tf(X.loc[mask[mask].index, col].to_numpy()); v = v[np.isfinite(v)]
        h, _ = np.histogram(v, bins=bins); res.append(dict(name=name, n=int(len(v)), h=(h / max(h.sum(), 1)).tolist(), med=float(np.median(v))))
    a = X.loc[groups[0][1][groups[0][1]].index, col]; b = X.loc[groups[1][1][groups[1][1]].index, col]
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]; v = np.r_[a, b]; ok = np.isfinite(v); au = roc_auc_score(y[ok], v[ok])
    return dict(label=lab, bins=[float(b) for b in bins], groups=res, auroc=float(max(au, 1 - au)))
ts, g3 = core.type_single, core.type_single.map(G)
out["D"] = [hist("tonal_conc", [("tonal", g3 == "tonal"), ("impulsive", g3 == "impulsive"), ("pulsed", g3 == "pulsed")], "tonal_conc", np.linspace(0, 1, 26)),
            hist("log_sig_f50", [("whistle", ts == "whistle"), ("moan", ts == "moan")], "log_hz", np.linspace(2, 5, 31)),
            hist("tr_log_ici_med", [("click", ts == "click"), ("pulse_train", ts == "pulse_train")], "log_s", np.linspace(-3.2, 0, 27))]
print(X.tonal_conc.describe().round(3).to_dict())
# E. 유형 × 대역 → 분류군
t = core[g3.notna()].copy(); t["type"] = g3[g3.notna()]; t["f50"] = 10 ** X.loc[t.index, "log_sig_f50"]
t["band"] = pd.cut(t.f50, [0, 1000, 8000, 1e7], labels=["1 kHz 미만", "1–8 kHz", "8 kHz 초과"])
out["E"] = dict(unmeasured=int(t.band.isna().sum()), cells=[])
for ty in ["tonal", "impulsive", "pulsed"]:
    for b in ["1 kHz 미만", "1–8 kHz", "8 kHz 초과"]:
        q = t[(t.type == ty) & (t.band == b)]
        out["E"]["cells"].append(dict(type=ty, band=b, n=int(len(q)), share=[float((q.group3 == en).sum() / max(len(q), 1)) for en, _ in tx]))
out["E"]["type_only"] = [dict(type=ty, n=int((t.type == ty).sum()), share=[float(((t.type == ty) & (t.group3 == en)).sum() / (t.type == ty).sum()) for en, _ in tx]) for ty in ["tonal", "impulsive", "pulsed"]]
json.dump(out, open("summary.json", "w"), ensure_ascii=False)
print(json.dumps(out["A"], ensure_ascii=False)); print(json.dumps(out["B"], ensure_ascii=False)[:900]); print([(d["label"], round(d["auroc"], 2), [(x["name"], x["n"], round(x["med"], 2)) for x in d["groups"]]) for d in out["D"]]); print(out["E"])
