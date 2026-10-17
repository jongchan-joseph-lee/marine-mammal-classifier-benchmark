#!/usr/bin/env python3
"""
build_maps.py — Watkins cluster-map builder (전체 컷 / 표본 / 종 지도 UMAP)

입력 (Input)
  --cuts      cuts.csv : 컷 목록표 (한 줄 = 한 컷). 필요한 열:
              cut_id, species, taxon, site, fs, dur_s, status, type_single, in_sample, split
              status  = eval(평가 대상) / hold(보류) / excluded(제외)
              split   = dev / test / 빈칸  (출처 분리 분할, source-disjoint split)
  --features  폴더. <이름>.npz 마다 cut_id, X (Bench v1 의 b07 입력 형식과 같다)
출력 (Output)
  --out       data.json  (assemble.py 가 HTML 에 넣는다)

계산 (What is computed)
  1) 전처리 Preprocess : 중앙값 대치 → 표준화 → ±8 자름 → PCA 최대 64차원  (Bench v1 / GPT 와 같은 순서)
  2) 투영 Projection   : PCA 2D, UMAP 2D, t-SNE 2D            ← 그림용. 거리 해석에 쓰지 않는다
  3) 군집 Clustering   : KMeans, HDBSCAN  ← 2D 그림이 아니라 전처리된 고차원 공간에서 계산
  4) 군집·이웃이 무엇을 따라가는가 : AMI, 가장 가까운 10컷의 일치 비율(NN10)
  5) 종 지도 UMAP (supervised UMAP) 과 그 대조 그림 3종 (표본에서만)
"""
import argparse, base64, json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import umap

warnings.filterwarnings("ignore")
SEED = 0
CFG = dict(pca_max=64, clip=8.0, umap_neighbors=30, umap_min_dist=0.1,
           tsne_perplexity=30, kmeans_k=8, kmeans_n_init=10, hdbscan_min_cluster=40, hdbscan_dims=16,
           nn_k=10, sup_target_weight=0.5, sup_knn=15, seed=SEED)

REP_INFO = {  # 표시 이름과 설명. 폴더에 있는 것만 쓴다.
    "physv0":   ("물리 특징 51개", "원래 쓰던 특징. 대역 에너지·시간 구조·음조성·충격성·반복성. 원 샘플링률 그대로"),
    "mfcc16k":  ("MFCC 80", "16 kHz 로 내린 뒤 계산. 8 kHz 위는 들어가지 않는다"),
    "vggish":   ("VGGish 128", "AudioSet 사전학습 모델. 16 kHz 입력"),
    "ast_zero": ("AST 768 · 0 채움", "AudioSet 사전학습 모델. 앞 10.24초, 짧으면 0 으로 채움"),
    "ast_tile": ("AST 768 · 되풀이 채움", "짧으면 되풀이, 길면 여러 창의 평균"),
    "effb3":    ("EfficientNet-B3 1536", "ImageNet 사전학습 모델. 표본 3,539컷에만 있다"),
    "meta":     ("녹음 조건만 (대조군)", "샘플링률과 컷 길이 두 값. 소리를 전혀 쓰지 않는다"),
}
REP_ALIAS = {"phys": "physv0", "mfcc": "mfcc16k"}
REP_ORDER = ["physv0", "mfcc16k", "vggish", "ast_zero", "ast_tile", "effb3", "meta"]
FS_FAM = [5, 10, 20, 40, 80, 160]  # kHz. 목록표에 fsfam 열이 없을 때만 쓴다
FS_ORDER = ["LF(<5k)", "5k", "10k", "20k", "40k", "80k", "160k"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def preprocess(X, fit_rows=None):
    """중앙값 대치 → 표준화 → 자름 → PCA. fit_rows 가 있으면 그 행으로만 맞춘다(평가 행 누출 방지)."""
    X = np.asarray(X, dtype=np.float64)
    F = X if fit_rows is None else X[fit_rows]
    med = np.nanmedian(F, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med)
    F = X if fit_rows is None else X[fit_rows]
    mu, sd = F.mean(0), F.std(0)
    sd[sd < 1e-12] = 1.0
    Z = np.clip((X - mu) / sd, -CFG["clip"], CFG["clip"])
    F = Z if fit_rows is None else Z[fit_rows]
    k = min(CFG["pca_max"], Z.shape[1], F.shape[0] - 1)
    p = PCA(n_components=k, random_state=SEED).fit(F)
    return p.transform(Z), p.explained_variance_ratio_


def q16(Y):
    """2D 좌표를 0..65535 로 양자화해 base64 로."""
    Y = np.asarray(Y, dtype=np.float64)
    lo, hi = np.percentile(Y, 0.2, axis=0), np.percentile(Y, 99.8, axis=0)
    pad = (hi - lo) * 0.04 + 1e-9
    lo, hi = lo - pad, hi + pad
    Q = np.clip((Y - lo) / (hi - lo), 0, 1)
    return base64.b64encode((Q * 65535).round().astype("<u2").tobytes()).decode()


def b64(a, dt):
    return base64.b64encode(np.asarray(a).astype(dt).tobytes()).decode()


def projections(P):
    out = {"pca": q16(P[:, :2])}
    out["umap"] = q16(umap.UMAP(n_neighbors=CFG["umap_neighbors"], min_dist=CFG["umap_min_dist"],
                                random_state=SEED).fit_transform(P))
    out["tsne"] = q16(TSNE(perplexity=CFG["tsne_perplexity"], init="pca", learning_rate="auto",
                           random_state=SEED, n_jobs=-1).fit_transform(P))
    return out


def structure(P, lab):
    """군집과 이웃이 어떤 라벨을 따라가는가. lab = {이름: 정수 배열(-1 = 라벨 없음)}"""
    km = KMeans(CFG["kmeans_k"], n_init=CFG["kmeans_n_init"], random_state=SEED).fit_predict(P)
    hd = HDBSCAN(min_cluster_size=CFG["hdbscan_min_cluster"]).fit_predict(P[:, :CFG["hdbscan_dims"]])
    nn = NearestNeighbors(n_neighbors=CFG["nn_k"] + 1).fit(P)
    idx = nn.kneighbors(P, return_distance=False)[:, 1:]
    m = {"ami_km": {}, "ami_hdb": {}, "nn": {}, "base": {},
         "hdb_n": int(hd.max() + 1), "hdb_noise": float((hd < 0).mean())}
    for name, y in lab.items():
        ok = y >= 0
        if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
            continue
        m["ami_km"][name] = float(ami(y[ok], km[ok]))
        m["ami_hdb"][name] = float(ami(y[ok], hd[ok]))
        both = ok[:, None] & ok[idx]  # 라벨이 있는 컷과 라벨이 있는 이웃만 센다 (형태 라벨은 일부 컷에만 있다)
        m["nn"][name] = float(((y[idx] == y[:, None]) & both).sum() / both.sum())
        p = np.bincount(y[ok]) / ok.sum()
        m["base"][name] = float((p ** 2).sum())  # 무작위로 골랐을 때 같은 라벨일 확률
    # 다른 출처의 컷 가운데서만 가장 가까운 10컷을 고른다: 같은 테이프·같은 출처의 닮은 컷을 빼고 봤을 때도 같은 라벨인가
    site = lab["site"]
    n, k = len(P), CFG["nn_k"]
    xi = np.zeros((n, k), dtype=np.int64)
    sq = (P ** 2).sum(1)
    for s0 in range(0, n, 1000):
        d = sq[s0:s0 + 1000, None] - 2 * P[s0:s0 + 1000] @ P.T + sq[None, :]
        d[site[s0:s0 + 1000, None] == site[None, :]] = np.inf
        xi[s0:s0 + 1000] = np.argpartition(d, k, axis=1)[:, :k]
    m["nn_x"], m["base_x"] = {}, {}
    nsite = np.bincount(site)
    for name, y in lab.items():
        ok = y >= 0
        if name == "site" or name not in m["nn"]:
            continue
        both = ok[:, None] & ok[xi]
        m["nn_x"][name] = float(((y[xi] == y[:, None]) & both).sum() / both.sum())
        # 무작위 기대값: 다른 출처의 (라벨 있는) 컷 가운데 같은 라벨의 비율을 컷마다 구해 평균
        tab = pd.crosstab(pd.Series(y[ok]), pd.Series(site[ok])).reindex(columns=range(len(nsite)), fill_value=0).to_numpy()
        yi, si_ = pd.factorize(y[ok], sort=True)[0], site[ok]
        same_other = tab.sum(1)[yi] - tab[yi, si_]
        all_other = tab.sum() - tab.sum(0)[si_]
        m["base_x"][name] = float(np.mean(same_other / np.maximum(all_other, 1)))
    return km, hd, m, idx


def knn_ba(Ytr, ytr, Yte=None, yte=None):
    """2D 그림에서 가까운 점끼리 같은 종인가 (균형정확도). Yte 가 없으면 자기 자신을 뺀 이웃으로 잰다."""
    k = CFG["sup_knn"]
    if Yte is None:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Ytr)
        idx = nn.kneighbors(Ytr, return_distance=False)[:, 1:]
        votes = ytr[idx]
        pred = np.array([np.bincount(v).argmax() for v in votes])
        return float(balanced_accuracy_score(ytr, pred))
    clf = KNeighborsClassifier(k).fit(Ytr, ytr)
    return float(balanced_accuracy_score(yte, clf.predict(Yte)))


def supervised_block(X, sp, split, rng):
    """종 지도 UMAP 과 대조 그림. X = 표본의 원 특징, sp = 종 번호, split = 1 개발 / 2 시험(처음 보는 출처)."""
    kw = dict(n_neighbors=CFG["umap_neighbors"], min_dist=CFG["umap_min_dist"], random_state=SEED)
    sup = dict(target_metric="categorical", target_weight=CFG["sup_target_weight"])
    P, _ = preprocess(X)
    out, met = {}, {}
    # (A) 비지도: 라벨을 쓰지 않는다
    Ya = umap.UMAP(**kw).fit_transform(P)
    out["unsup"], met["unsup"] = q16(Ya), {"all": knn_ba(Ya, sp)}
    # (B) 종 지도: 모든 컷의 종 라벨을 알려 준다 (문헌 그림에서 흔한 방식)
    Yb = umap.UMAP(**kw, **sup).fit_transform(P, y=sp)
    out["sup_all"], met["sup_all"] = q16(Yb), {"all": knn_ba(Yb, sp)}
    # (C) 종 지도를 개발 출처로만 만들고, 처음 보는 출처의 컷을 그 지도에 얹는다
    tr, te = split == 1, split == 2
    Pc, _ = preprocess(X, fit_rows=np.where(tr)[0])
    mc = umap.UMAP(**kw, **sup).fit(Pc[tr], y=sp[tr])
    Yc = np.zeros((len(sp), 2))
    Yc[tr] = mc.embedding_
    Yc[te] = mc.transform(Pc[te])
    seen = te & np.isin(sp, np.unique(sp[tr]))  # 학습에 있던 종만 채점할 수 있다
    out["sup_holdout"] = q16(Yc)
    met["sup_holdout"] = {"train": knn_ba(Yc[tr], sp[tr]),
                          "heldout": knn_ba(Yc[tr], sp[tr], Yc[seen], sp[seen]) if seen.sum() > 20 else None,
                          "n_train": int(tr.sum()), "n_heldout": int(te.sum()), "n_scored": int(seen.sum()),
                          "chance": 1.0 / len(np.unique(sp[seen])) if seen.sum() else None}
    # 같은 조건의 비지도 지도에 얹었을 때 (비교값. 그림은 싣지 않는다)
    mu = umap.UMAP(**kw).fit(Pc[tr])
    met["unsup_holdout"] = {"train": knn_ba(mu.embedding_, sp[tr]),
                            "heldout": knn_ba(mu.embedding_, sp[tr], mu.transform(Pc[seen]), sp[seen]) if seen.sum() > 20 else None}
    # (D) 라벨을 섞어서 종 지도: 소리와 무관한 가짜 라벨로도 섬이 생기는가
    fake = rng.permutation(sp)
    Yd = umap.UMAP(**kw, **sup).fit_transform(P, y=fake)
    out["shuffled"] = q16(Yd)
    out["shuffled_label"] = b64(fake, "<u1")
    met["shuffled"] = {"fake": knn_ba(Yd, fake), "real": knn_ba(Yd, sp)}
    return out, met


def codes(s, order=None):
    s = s.fillna("").astype(str)
    cats = order if order is not None else [c for c in s.value_counts().index if c != ""]
    m = {c: i for i, c in enumerate(cats)}
    return s.map(m).fillna(-1).astype(int).to_numpy(), cats


def load_rep(folder, key, id2row):
    """<key>.npz 를 읽어 (목록표의 행 번호, X) 로 돌려준다. 컷은 cut_id 로만 잇는다. 모든 값이 빈 행은 뺀다."""
    if folder is None:
        return None
    for name in [key] + [k for k, v in REP_ALIAS.items() if v == key]:
        f = Path(folder) / f"{name}.npz"
        if f.exists():
            break
    else:
        return None
    z = np.load(f, allow_pickle=False)
    ids = [str(c) for c in z["cut_id"]]
    assert len(set(ids)) == len(ids), f"{f}: cut_id 중복"
    keep = np.array([i for i, c in enumerate(ids) if c in id2row])
    rows = np.array([id2row[ids[i]] for i in keep])
    X = np.asarray(z["X"], dtype=np.float64)[keep]
    X[~np.isfinite(X)] = np.nan
    empty = np.isnan(X).all(1)
    o = np.argsort(rows[~empty])
    return rows[~empty][o], X[~empty][o], int(empty.sum()), int(np.isnan(X[~empty]).any(1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", required=True)
    ap.add_argument("--features", required=True, help="전체 컷의 특징 폴더")
    ap.add_argument("--features-sample", default=None, help="표본에만 있는 표현의 폴더 (예: effb3)")
    ap.add_argument("--gpt-cache", default=None, help="GPT 가 저장한 cached_umap3d 폴더 (대조용)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="real", choices=["real", "mock"])
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    cuts = pd.read_csv(a.cuts, dtype={"cut_id": str}, keep_default_na=False)
    n = len(cuts)
    assert cuts.cut_id.is_unique
    cuts["in_sample"] = cuts["in_sample"].astype(int)
    if "fsfam" not in cuts:
        fam = np.array(FS_FAM)[np.abs(np.log(cuts["fs"].to_numpy()[:, None] / 1000.0) - np.log(FS_FAM)).argmin(1)]
        cuts["fsfam"] = [f"{f}k" for f in fam]
    if "feat" not in cuts:
        cuts["feat"] = "main"
    sp, sp_c = codes(cuts["species"])
    tx, tx_c = codes(cuts["taxon"])
    si, si_c = codes(cuts["site"])
    ff, ff_c = codes(cuts["fsfam"], [f for f in FS_ORDER if (cuts["fsfam"] == f).any()])
    st, st_c = codes(cuts["status"], ["eval", "hold", "excluded"])
    ty, ty_c = codes(cuts["type_single"])
    wh, wh_c = codes(cuts["status_reason"])
    ft, ft_c = codes(cuts["feat"], ["main", "lf87", "none"])
    spl = cuts["split"].map({"dev": 1, "test": 2}).fillna(0).astype(int).to_numpy()
    smp = cuts["in_sample"].to_numpy()
    lab = {"taxon": tx, "species": sp, "site": si, "fsfam": ff, "type": ty}
    id2row = {c: i for i, c in enumerate(cuts["cut_id"])}

    data = {"kind": a.kind, "built": time.strftime("%Y-%m-%d"), "n": n, "cfg": CFG,
            "cuts": {"id": cuts["cut_id"].tolist(), "sp": sp.tolist(), "tx": tx.tolist(), "site": si.tolist(),
                     "fsf": ff.tolist(), "st": st.tolist(), "ty": ty.tolist(), "smp": smp.tolist(), "why": wh.tolist(),
                     "feat": ft.tolist(), "split": spl.tolist(), "fs": cuts["fs"].astype(float).round().astype(int).tolist(),
                     "dur": cuts["dur_s"].astype(float).round(2).tolist()},
            "cats": {"sp": sp_c, "tx": tx_c, "site": si_c, "fsf": ff_c, "st": st_c, "ty": ty_c, "why": wh_c},
            "reps": [], "full": {}, "sample": {}, "sup": {}, "check": {}}

    def block(P, ev, rows, m, km, hd):
        b = projections(P)
        b.update(rows=b64(rows, "<u2"), km=b64(km, "<u1"), hdb=b64(hd + 1, "<u1"), metrics=m,
                 pca_var=[float(ev[0]), float(ev[1])], pca_dims=int(P.shape[1]))
        return b

    for key in REP_ORDER:
        full = load_rep(a.features, key, id2row)
        samp = load_rep(a.features_sample, key, id2row) if full is None else None
        if full is None and samp is None:
            continue
        info = {"key": key, "label": REP_INFO[key][0], "desc": REP_INFO[key][1]}
        if full is not None:
            rows, X, n_empty, n_part = full
            log(f"[{key}] 전체 {len(rows)}컷, 차원 {X.shape[1]}, 모든 값이 빈 컷 {n_empty} (지도에서 뺌), 일부 값이 빈 컷 {n_part} (중앙값으로 채움)")
            P, ev = preprocess(X)
            km, hd, m, idx = structure(P, {k: v[rows] for k, v in lab.items()})
            data["full"][key] = block(P, ev, rows, m, km, hd)
            log(f"  전체 완료: NN10 {m['nn']}")
            # GPT 가 저장한 군집·이웃과 대조 (같은 컷, 같은 전처리 순서로 따로 다시 계산한 값과의 일치)
            g = Path(a.gpt_cache) / f"{key}.npz" if a.gpt_cache else None
            if g is not None and g.exists():
                z = np.load(g, allow_pickle=False)
                gid = np.array([str(c) for c in z["cut_id"]])
                pos = {c: i for i, c in enumerate(cuts["cut_id"].to_numpy()[rows])}
                sel = np.array([i for i, c in enumerate(gid) if c in pos])
                mine = np.array([pos[gid[i]] for i in sel])
                grow = np.array([pos.get(c, -1) for c in gid])  # GPT 행 번호 → 내 행 번호
                their = grow[z["NN10"][sel]]
                ov = np.mean([len(set(idx[mine[j]]) & set(their[j])) / 10 for j in range(len(sel))])
                from sklearn.metrics import adjusted_rand_score
                data["check"][key] = {"n": int(len(sel)), "nn10_overlap": float(ov),
                                      "kmeans_ari": float(adjusted_rand_score(z["cluster8"][sel], km[mine]))}
                log(f"  GPT 대조: {data['check'][key]}")
            sm = smp[rows] == 1
            rows_s, X_s = rows[sm], X[sm]
            info.update(dim=int(X.shape[1]), n_full=int(len(rows)), n_empty=n_empty, n_partial=n_part)
        else:
            rows_s, X_s, n_empty, n_part = samp
            keep = smp[rows_s] == 1
            rows_s, X_s = rows_s[keep], X_s[keep]
            info.update(dim=int(X_s.shape[1]), n_full=0, n_empty=n_empty, n_partial=n_part)
            log(f"[{key}] 표본에만 있음 {len(rows_s)}컷, 차원 {X_s.shape[1]}")
        info["n_sample"] = int(len(rows_s))
        data["reps"].append(info)
        if len(rows_s) > 200:
            Ps, evs = preprocess(X_s)
            km, hd, m, _ = structure(Ps, {k: v[rows_s] for k, v in lab.items()})
            data["sample"][key] = block(Ps, evs, rows_s, m, km, hd)
            log("  표본 완료")
            if key != "meta":
                o2, m2 = supervised_block(X_s, sp[rows_s], spl[rows_s], rng)
                o2.update(rows=b64(rows_s, "<u2"), metrics=m2)
                data["sup"][key] = o2
                log(f"  종 지도 완료: {json.dumps(m2, ensure_ascii=False)}")

    Path(a.out).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    log("저장", a.out, f"{Path(a.out).stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    sys.exit(main())
