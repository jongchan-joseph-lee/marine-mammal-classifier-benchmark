# -*- coding: utf-8 -*-
"""7단계. 특징(또는 임베딩)을 같은 규약으로 평가한다. docs/protocol.md 의 규약을 코드로 옮긴 것이다.

입력   work/features/<이름>.npz (cut_id, X)   — 어떤 방법이 만든 것이든 이 형식이면 된다
       data/bench_core.csv                    — 과제 라벨, 분할, 해역 묶음
출력   work/eval/<run>/results.csv            지표 (과제 × 규약 × 특징 × 부분집합), 95 % 구간
       work/eval/<run>/per_class.csv          범주별 재현율과 표본 수
       work/eval/<run>/paired.csv             같은 평가 컷에서의 방법 간 차이와 95 % 구간
       work/eval/<run>/diagnostics.csv        진단 지표 (통과 기준이 아님)
       work/eval/<run>/predictions.csv        컷마다 예측 (실패 사례를 찾아 듣는 데 쓴다)
       work/eval/<run>/confusion_*.csv        혼동 행렬
       work/eval/<run>/report.md              요약

과제   group3        분류군 3종 (수염고래류·이빨고래류·기각류)
       type_single   노트가 대상 동물의 소리로 한 가지 형태만 적은 컷의 형태 (약한 라벨. 청취로 확인한 정답이 아니다)
       type_multi    형태별 있음/없음 (다중 라벨, 범주별 AUROC). 형태를 모르는 소리나 다른 종의 소리가 적힌 컷은 뺀다
                     --type-labels tags 를 주면 두 과제를 v1 의 라벨(공개본의 낱말 태그)로 돌린다. 분할은 같으므로 라벨의 효과만 볼 수 있다
       species       해역이 개발 2개·시험 1개 이상인 종의 종 분류 (종 수준이 어디까지 되는지 재는 진단 과제)
규약   test          개발 해역 전체로 학습 → 잠금 시험 해역으로 평가 (보고용)
       devcv         개발 해역 5겹, 해역 단위 (개발 중 비교용)
       allcv         전 해역 7겹, 해역 단위 (잠금을 쓰지 않는 참고값. 독립 해역 수가 가장 많다)
       regioncv      구역 단위 6겹 (같은 넓은 구역은 모든 해를 통틀어 한 겹. 가장 엄격한 참고값)
       filecv        파일 단위 무작위 5겹 (문헌에서 흔한 방식. 같은 해역이 학습과 평가에 함께 들어간다. 견주기 위한 값)
       lso           종을 통째로 빼고 분류군을 맞히기 (group3 만. 그 종이 있던 해역도 학습에서 뺀다)

분류기  겹 안에서 맞춘 중앙값 대치 → 표준화 → 주성분(최대 64개) → 로지스틱 회귀(C=1, 범주 가중 균형).
        조정하는 값이 없다. 방법마다 분류기를 따로 다듬지 않는다.
구간    평가 컷을 해역 단위로 다시 뽑는 부트스트랩 2,000회의 가운데 95 %.
        해역이 바뀔 때의 흔들림만 담는다. 학습 자료가 바뀔 때의 흔들림은 담지 않는다.
대조군  meta(fs 와 길이만), 무작위 특징의 분포(잠금 시험 200회, 교차검증 40회. 대괄호는 그 분포의 가운데 95 %).

쓰는 법  python src/b07_evaluate.py --features meta,mfcc16k,physv0 [--run 이름] [--protocols test,devcv,allcv,regioncv,filecv,lso]
                                    [--tasks group3,type_single,type_multi,species] [--gold work/gold/gold_labels.csv]
                                    [--type-labels note|tags] [--exclude-classes song]
  --exclude-classes  형태 과제에서 뺄 범주(쉼표로 구분). type_single 에서는 그 범주의 컷이 학습과 평가에서 모두 빠지고,
                     type_multi 에서는 그 범주의 AUROC 만 빠진다. 범주 구성이 다른 두 실행을 같은 범주로 맞춰 견줄 때 쓴다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from wb_common import DATA_DIR, GROUPS3, SEED, TYPE_CLASSES, csv_arg, log, work_dir

N_BOOT = 2000
N_RANDOM = 200          # 잠금 시험(test)에서의 무작위 특징 반복 수
N_RANDOM_CV = 40        # 교차검증 규약에서의 반복 수 (겹마다 다시 맞추므로 줄였다)
RANDOM_DIM = 32
MAX_PCA = 64
MIN_TRAIN, MIN_EVAL = 15, 8
MIN_SUBSET_SITES = 4    # 부분집합(지역, fs 계열)의 값을 내는 최소 평가 해역 수
REGION_SCOPES = {"arctic", "korea", "arctic+korea"}
RANDOM_LABEL = "(무작위 특징)"


# ----------------------------------------------------------------------------------------------
def make_clf():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)


def fit_transform_fold(Xtr: np.ndarray, Xev: np.ndarray):
    """대치·표준화·주성분을 학습 쪽으로만 맞추고 양쪽에 적용한다."""
    from sklearn.decomposition import PCA
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med)
    Xev = np.where(np.isfinite(Xev), Xev, med)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    # 표준화한 값을 ±8 로 자른다. 학습 쪽에서 거의 일정하던 특징이 평가 쪽에서 튀는 것을 막는다.
    Xtr, Xev = np.clip((Xtr - mu) / sd, -8, 8), np.clip((Xev - mu) / sd, -8, 8)
    k = min(MAX_PCA, Xtr.shape[1], Xtr.shape[0] - 1)
    if k < Xtr.shape[1]:
        pca = PCA(n_components=k, random_state=SEED).fit(Xtr)
        Xtr, Xev = pca.transform(Xtr), pca.transform(Xev)
    return Xtr, Xev


def balanced_acc(y_true: np.ndarray, y_pred: np.ndarray, classes: list) -> float:
    rec = [np.mean(y_pred[y_true == c] == c) for c in classes if (y_true == c).any()]
    return float(np.mean(rec)) if rec else np.nan


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: list) -> float:
    f = []
    for c in classes:
        if not (y_true == c).any():
            continue
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f)) if f else np.nan


def site_bootstrap(sites: np.ndarray, stat, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """해역을 복원추출해 stat(행 번호 배열) 의 가운데 95 % 구간을 구한다."""
    uniq = np.unique(sites)
    rows = {s: np.flatnonzero(sites == s) for s in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        v = stat(np.concatenate([rows[s] for s in pick]))
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def balanced_site_bootstrap(sites, y_true, pred_a, classes, n_boot=N_BOOT, seed=SEED, pred_b=None):
    """Same site resamples and statistic, using additive class counts for speed."""
    unique, group = np.unique(sites, return_inverse=True)
    rng = np.random.default_rng(seed)
    weights = np.stack([np.bincount(rng.choice(len(unique), size=len(unique), replace=True),
                                  minlength=len(unique)) for _ in range(n_boot)])
    totals = np.zeros((len(unique), len(classes)), dtype=np.float64)
    correct_a = np.zeros_like(totals)
    correct_b = np.zeros_like(totals) if pred_b is not None else None
    for j, cls in enumerate(classes):
        belongs = y_true == cls
        totals[:, j] = np.bincount(group[belongs], minlength=len(unique))
        good = belongs & (pred_a == cls)
        correct_a[:, j] = np.bincount(group[good], minlength=len(unique))
        if correct_b is not None:
            good_b = belongs & (pred_b == cls)
            correct_b[:, j] = np.bincount(group[good_b], minlength=len(unique))
    denominators = weights @ totals
    def values(correct):
        numerators = weights @ correct
        recall = np.divide(numerators, denominators, out=np.zeros_like(numerators), where=denominators > 0)
        present = (denominators > 0).sum(1)
        return np.divide(recall.sum(1), present, out=np.full(len(weights), np.nan), where=present > 0)
    result = values(correct_a)
    if correct_b is not None:
        result -= values(correct_b)
    result = result[np.isfinite(result)]
    if len(result) < n_boot * .5:
        return np.nan, np.nan
    return float(np.percentile(result, 2.5)), float(np.percentile(result, 97.5))


# ----------------------------------------------------------------------------------------------
def protocol_folds(df: pd.DataFrame, protocol: str, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """(학습 행, 평가 행) 목록. df 는 과제에 쓰는 행만 든 표(0..n-1 로 번호를 다시 매긴 것)."""
    split = df.split.to_numpy()
    if protocol == "test":
        return [(np.flatnonzero(split == "dev"), np.flatnonzero(split == "test"))]
    if protocol in ("devcv", "allcv", "regioncv"):
        colname = {"devcv": "fold", "allcv": "fold_all", "regioncv": "fold_region"}[protocol]
        col = df[colname].to_numpy().astype(int)
        use = np.flatnonzero((split == "dev") if protocol == "devcv" else np.ones(len(df), bool))
        out = []
        for k in sorted(set(col[use])):
            if k < 0:
                continue
            out.append((use[col[use] != k], use[col[use] == k]))
        return out
    if protocol == "filecv":
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return [(tr, ev) for tr, ev in skf.split(np.zeros(len(y)), y)]
    if protocol == "lso":
        sp = df.species.to_numpy()
        site = df.g_site.to_numpy()
        out = []
        for s in sorted(set(sp)):
            ev = np.flatnonzero(sp == s)
            if len(ev) < MIN_EVAL:
                continue
            bad_sites = set(site[ev])
            tr = np.flatnonzero((sp != s) & ~np.isin(site, list(bad_sites)))
            out.append((tr, ev))
        return out
    raise ValueError(protocol)


def keep_classes(df: pd.DataFrame, y: np.ndarray, protocol: str) -> list:
    """학습과 평가 양쪽에 충분히 있는 범주만 남긴다. 빠진 범주는 '평가 불가'로 보고한다."""
    split = df.split.to_numpy()
    site = df.g_site.to_numpy()
    out = []
    for c in sorted(set(y)):
        if protocol == "test":
            ok = (np.sum((y == c) & (split == "dev")) >= MIN_TRAIN) and (np.sum((y == c) & (split == "test")) >= MIN_EVAL)
        elif protocol == "devcv":
            sel = (y == c) & (split == "dev")
            ok = sel.sum() >= MIN_TRAIN + MIN_EVAL and len(set(site[sel])) >= 2
        elif protocol == "lso":
            ok = True
        else:
            sel = y == c
            ok = sel.sum() >= MIN_TRAIN + MIN_EVAL and (protocol == "filecv" or len(set(site[sel])) >= 2)
        if ok:
            out.append(c)
    return out


def run_multiclass(X: np.ndarray, y: np.ndarray, folds) -> tuple[np.ndarray, np.ndarray]:
    """평가 행마다 예측 라벨과 최대 확률. 평가되지 않은 행은 빈 문자열."""
    pred = np.array([""] * len(y), dtype=object)
    pmax = np.full(len(y), np.nan)
    for tr, ev in folds:
        if len(tr) == 0 or len(ev) == 0 or len(set(y[tr])) < 2:
            continue
        Xtr, Xev = fit_transform_fold(X[tr], X[ev])
        clf = make_clf()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xtr, y[tr])
            proba = clf.predict_proba(Xev)
        pred[ev] = clf.classes_[proba.argmax(1)]
        pmax[ev] = proba.max(1)
    return pred, pmax


def run_multilabel(X: np.ndarray, Y: np.ndarray, folds) -> np.ndarray:
    """(n, 범주 수) 점수. 평가되지 않은 칸은 NaN."""
    S = np.full(Y.shape, np.nan)
    for tr, ev in folds:
        if len(tr) == 0 or len(ev) == 0:
            continue
        Xtr, Xev = fit_transform_fold(X[tr], X[ev])
        for j in range(Y.shape[1]):
            if Y[tr, j].sum() < 5 or (1 - Y[tr, j]).sum() < 5:
                continue
            clf = make_clf()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(Xtr, Y[tr, j])
                S[ev, j] = clf.predict_proba(Xev)[:, 1]
    return S


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """순위 기반 AUROC. 한쪽 범주가 없으면 NaN."""
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# ----------------------------------------------------------------------------------------------
def load_features(names: list[str], directory=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for n in names:
        p = (Path(directory) if directory else work_dir("features")) / f"{n}.npz"
        if not p.exists():
            raise SystemExit(f"특징 파일이 없다: {p}")
        z = np.load(p, allow_pickle=False)
        ids = z["cut_id"].astype(str)
        X = z["X"].astype(np.float64)
        if len(ids) != len(X) or len(set(ids)) != len(ids):
            raise SystemExit(f"{n}: cut_id 와 X 의 행이 맞지 않거나 cut_id 가 겹친다")
        out[n] = (ids, X)
    return out


def subsets_for(task: str, df: pd.DataFrame) -> dict[str, np.ndarray]:
    sub = {"all": np.ones(len(df), bool)}
    reg = df.region_scope.isin(REGION_SCOPES).to_numpy()
    if reg.any() and not reg.all():
        sub["region"] = reg                                    # 북극권·한반도 대상 종만
    if task == "group3":
        fam = df.fs_family.isin(["10k", "20k"]).to_numpy()
        if fam.any():
            sub["fs10k-20k"] = fam                             # 세 분류군이 함께 있는 fs 계열
    if task in ("species", "type_single"):
        for name in ["10k", "20k", "40k", "80k", "160k"]:      # fs 계열 안에서의 구분 (학습은 전체로 한다)
            fam = (df.fs_family == name).to_numpy()
            if fam.sum() >= 40:
                sub[f"fs{name}"] = fam
    return sub


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, nargs="+", help="특징 이름들. 쉼표로 잇거나 띄어 쓴다 (meta,mfcc16k 또는 meta mfcc16k)")
    ap.add_argument("--run", default="")
    ap.add_argument("--tasks", nargs="+", default="group3,type_single,type_multi,species")
    ap.add_argument("--protocols", nargs="+", default="test,devcv,allcv,regioncv,filecv,lso")
    ap.add_argument("--manifest", default=str(DATA_DIR / "bench_core.csv"))
    ap.add_argument("--feature-dir", default="", help="특징 입력 사본 폴더. 생략하면 기존 work/features")
    ap.add_argument("--pool", default="main")
    ap.add_argument("--gold", default="", help="골드 라벨 CSV (review 화면에서 내보낸 것). 주면 형태 과제를 골드 라벨로도 평가한다")
    ap.add_argument("--type-labels", default="note", choices=["note", "tags"],
                    help="형태 과제의 라벨. note = 노트를 읽어 만든 것(v1.1, 기본값), tags = 공개본의 낱말 태그(v1)")
    ap.add_argument("--exclude-classes", nargs="+", default="", help="형태 과제에서 뺄 범주 (쉼표로 잇거나 띄어 쓴다. 예: song)")
    ap.add_argument("--no-random", action="store_true")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    feats, tasks, protocols, excl = csv_arg(args.features), csv_arg(args.tasks), csv_arg(args.protocols), csv_arg(args.exclude_classes)
    if [c for c in excl if c not in TYPE_CLASSES]:
        raise SystemExit(f"--exclude-classes 에 모르는 범주가 있다: {[c for c in excl if c not in TYPE_CLASSES]}. 쓸 수 있는 이름: {TYPE_CLASSES}")
    run = args.run or time.strftime("run_%Y%m%d_%H%M")
    odir = work_dir("eval", run)
    t0 = time.time()

    core = pd.read_csv(args.manifest, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    core = core[core.pool == args.pool].reset_index(drop=True)
    if args.type_labels == "tags":
        miss = [c for c in ("task_type_single_tags", "task_type_multi_tags") if c not in core.columns]
        if miss:
            raise SystemExit(f"목록표에 v1 라벨 열이 없다: {miss}")
        core["task_type_single"], core["task_type_multi"] = core.task_type_single_tags, core.task_type_multi_tags
        log("형태 과제의 라벨: 공개본의 낱말 태그 (v1)")
    F = load_features(feats, args.feature_dir or None)
    # 견주는 특징이 모두 가진 컷만 쓴다 (같은 컷에서 견주기 위해). 빠진 수는 보고한다.
    common = set(core.cut_id)
    cov = {}
    for n, (ids, _) in F.items():
        cov[n] = len(set(ids) & set(core.cut_id))
        common &= set(ids)
    n_pool = len(core)
    core = core[core.cut_id.isin(common)].reset_index(drop=True)
    n_dropped = n_pool - len(core)
    log(f"핵심 표본(pool={args.pool}) {n_pool}컷 가운데 모든 특징이 있는 컷: {len(core)}  | 특징별: {cov}")
    if n_dropped:
        # 한 특징에 없는 컷은 다른 특징의 평가에서도 빠진다 (예: mfcc_band4k 는 fs 8 kHz 미만 컷이 없다).
        log(f"주의: {n_dropped}컷이 모든 특징의 평가에서 빠졌다. 가장 적게 가진 특징: {min(cov, key=cov.get)}. "
            f"전체 핵심 표본의 값이 필요하면 그 특징을 빼고 따로 돌린다.")
    if len(core) < 50:
        raise SystemExit("평가할 컷이 너무 적다")
    Xall = {}
    for n, (ids, X) in F.items():
        pos = {c: i for i, c in enumerate(ids)}
        Xall[n] = X[[pos[c] for c in core.cut_id]]

    gold = None
    if args.gold:
        gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        miss = [c for c in ("cut_id", "dominant_type") if c not in gold.columns]
        if miss:
            raise SystemExit(f"골드 CSV 에 필요한 열이 없다: {miss}. review 화면의 '표기 CSV 저장'으로 만든 파일을 그대로 준다")
        n_gold = len(gold)
        gold = gold.drop_duplicates("cut_id", keep="last")
        gold = gold[gold.cut_id.isin(set(core.cut_id))]
        log(f"골드 라벨 {n_gold}줄 가운데 평가에 쓰는 핵심 표본에 든 것 {len(gold)}개 (시험 해역 {int(gold.cut_id.isin(set(core[core.split == 'test'].cut_id)).sum())}개)")

    res, per_class, paired, preds = [], [], [], []

    for task in tasks:
        # ---- 과제에 쓰는 행과 라벨 ------------------------------------------------------
        if task == "group3":
            df = core[core.task_group3 != ""].reset_index(drop=True)
            y_all = df.task_group3.to_numpy()
        elif task == "type_single":
            df = core[(core.task_type_single != "") & ~core.task_type_single.isin(excl)].reset_index(drop=True)
            y_all = df.task_type_single.to_numpy()
        elif task == "species":
            df = core[core.task_species != ""].reset_index(drop=True)
            y_all = df.task_species.to_numpy()
        elif task == "type_multi":
            df = core[core.task_type_multi != ""].reset_index(drop=True)
            y_all = None
        else:
            raise SystemExit(f"모르는 과제: {task}")
        if len(df) < 40:
            log(f"[{task}] 컷이 {len(df)}개뿐이라 건너뛴다")
            continue
        rows_in_core = {c: i for i, c in enumerate(core.cut_id)}
        take = np.array([rows_in_core[c] for c in df.cut_id])
        site = df.g_site.to_numpy()

        for protocol in protocols:
            if protocol == "lso" and task != "group3":
                continue
            # ---------------- 다중 라벨 --------------------------------------------------
            if task == "type_multi":
                classes = [c for c in TYPE_CLASSES if c not in excl]
                Y = np.array([[1.0 if c in s.split(";") else 0.0 for c in classes] for s in df.task_type_multi])
                folds = protocol_folds(df, protocol, df.task_type_multi.to_numpy()) if protocol != "filecv" else None
                if protocol == "filecv":
                    from sklearn.model_selection import KFold
                    folds = list(KFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(df))))
                scores = {}
                for n in feats:
                    scores[n] = run_multilabel(Xall[n][take], Y, folds)
                ev_rows = np.flatnonzero(np.isfinite(next(iter(scores.values()))).any(1))
                good = [j for j, c in enumerate(classes)
                        if Y[ev_rows, j].sum() >= MIN_EVAL and (1 - Y[ev_rows, j]).sum() >= MIN_EVAL
                        and all(np.isfinite(scores[n][ev_rows, j]).all() for n in feats)]
                if not good:
                    continue
                for n in feats:
                    S = scores[n]
                    per = {classes[j]: auroc(Y[ev_rows, j], S[ev_rows, j]) for j in good}
                    stat = lambda idx, S=S: float(np.nanmean([auroc(Y[ev_rows[idx], j], S[ev_rows[idx], j]) for j in good]))
                    lo, hi = site_bootstrap(site[ev_rows], stat, args.boot)
                    res.append(dict(task=task, protocol=protocol, feature=n, subset="all", metric="macro_auroc",
                                    value=float(np.mean(list(per.values()))), ci_lo=lo, ci_hi=hi, chance=0.5,
                                    n_eval=len(ev_rows), n_sites=len(set(site[ev_rows])), n_classes=len(good),
                                    classes=";".join(classes[j] for j in good),
                                    dropped=";".join(c for j, c in enumerate(classes) if j not in good)))
                    for c, v in per.items():
                        j = classes.index(c)
                        per_class.append(dict(task=task, protocol=protocol, feature=n, subset="all", cls=c, metric="auroc",
                                              value=v, n_pos=int(Y[ev_rows, j].sum()), n_eval=len(ev_rows),
                                              n_sites=len(set(site[ev_rows][Y[ev_rows, j] > 0]))))
                continue

            # ---------------- 단일 라벨 --------------------------------------------------
            classes = keep_classes(df, y_all, protocol)
            dropped = sorted(set(y_all) - set(classes))
            if len(classes) < 2:
                log(f"[{task}/{protocol}] 평가할 수 있는 범주가 {len(classes)}개라 건너뛴다 (빠진 범주: {dropped})")
                continue
            # 교차검증 규약에서는 두 겹 이상에 걸쳐 있는 범주만 평가할 수 있다. 한 겹에만 있는 범주는
            # 그 겹을 뺄 때 학습 자료가 하나도 없어 재현율이 0 으로 나온다 (특징의 한계가 아니라 자료의 구조다).
            fold_col = {"devcv": "fold", "allcv": "fold_all", "regioncv": "fold_region"}.get(protocol)
            if fold_col:
                fc = df[fold_col].to_numpy().astype(int)
                in_scope = (df.split.to_numpy() == "dev") if protocol == "devcv" else np.ones(len(df), bool)
                classes = [c for c in classes if len(set(fc[(y_all == c) & in_scope & (fc >= 0)])) >= 2]
                dropped = sorted(set(y_all) - set(classes))
                if len(classes) < 2:
                    log(f"[{task}/{protocol}] 두 겹 이상에 걸친 범주가 {len(classes)}개라 건너뛴다")
                    continue
            keep = np.isin(y_all, classes)
            d2 = df[keep].reset_index(drop=True)
            y = y_all[keep]
            tk = take[keep]
            st = site[keep]
            folds = protocol_folds(d2, protocol, y)
            out_pred = {}
            for n in feats:
                out_pred[n] = run_multiclass(Xall[n][tk], y, folds)
            ev_rows = np.flatnonzero(next(iter(out_pred.values()))[0] != "")
            if len(ev_rows) == 0:
                continue
            ev_classes = [c for c in classes if (y[ev_rows] == c).any()]
            subs = subsets_for(task, d2)
            for n in feats:
                pred, pmax = out_pred[n]
                for sname, smask in subs.items():
                    rows = ev_rows[smask[ev_rows]]
                    cls_here = [c for c in ev_classes if (y[rows] == c).sum() >= MIN_EVAL]
                    if len(rows) < 2 * MIN_EVAL or len(cls_here) < 2:
                        continue
                    if sname != "all" and len(set(st[rows])) < MIN_SUBSET_SITES:
                        continue                               # 해역이 너무 적은 부분집합은 값을 내지 않는다
                    ba = balanced_acc(y[rows], pred[rows], cls_here)
                    lo, hi = balanced_site_bootstrap(st[rows], y[rows], pred[rows], cls_here, args.boot)
                    res.append(dict(task=task, protocol=protocol, feature=n, subset=sname, metric="balanced_acc",
                                    value=ba, ci_lo=lo, ci_hi=hi, chance=1 / len(cls_here), n_eval=len(rows),
                                    n_sites=len(set(st[rows])), n_classes=len(cls_here), classes=";".join(cls_here),
                                    dropped=";".join(dropped)))
                    res.append(dict(task=task, protocol=protocol, feature=n, subset=sname, metric="macro_f1",
                                    value=macro_f1(y[rows], pred[rows], cls_here), ci_lo=np.nan, ci_hi=np.nan,
                                    chance=np.nan, n_eval=len(rows), n_sites=len(set(st[rows])),
                                    n_classes=len(cls_here), classes=";".join(cls_here), dropped=";".join(dropped)))
                    for c in cls_here:
                        sel = rows[y[rows] == c]
                        per_class.append(dict(task=task, protocol=protocol, feature=n, subset=sname, cls=c, metric="recall",
                                              value=float(np.mean(pred[sel] == c)), n_pos=len(sel), n_eval=len(rows),
                                              n_sites=len(set(st[sel]))))
                # 혼동 행렬과 예측
                conf = pd.crosstab(pd.Series(y[ev_rows], name="true"), pd.Series(pred[ev_rows], name="pred"))
                conf.to_csv(odir / f"confusion_{task}_{protocol}_{n}.csv", encoding="utf-8-sig")
                preds.append(pd.DataFrame(dict(task=task, protocol=protocol, feature=n, cut_id=d2.cut_id.to_numpy()[ev_rows],
                                               g_site=st[ev_rows], true=y[ev_rows], pred=pred[ev_rows],
                                               p_max=np.round(pmax[ev_rows], 4))))
            # 방법 간 직접 비교: 같은 평가 컷, 같은 해역 재추출
            for i, a in enumerate(feats):
                for b in feats[i + 1:]:
                    pa, pb = out_pred[a][0], out_pred[b][0]
                    diff = balanced_acc(y[ev_rows], pa[ev_rows], ev_classes) - balanced_acc(y[ev_rows], pb[ev_rows], ev_classes)
                    lo, hi = balanced_site_bootstrap(st[ev_rows], y[ev_rows], pa[ev_rows], ev_classes,
                                                    args.boot, pred_b=pb[ev_rows])
                    verdict = ("A 가 높다" if lo > 0 else "B 가 높다" if hi < 0 else "어느 쪽이 나은지 입증하지 못함")
                    paired.append(dict(task=task, protocol=protocol, A=a, B=b, diff_balanced_acc=diff, ci_lo=lo, ci_hi=hi,
                                       verdict=verdict, n_eval=len(ev_rows), n_sites=len(set(st[ev_rows]))))
            # 무작위 특징 대조군 (해역 단위 규약에서만)
            if not args.no_random and protocol in ("test", "devcv", "allcv", "regioncv"):
                vals = []
                n_rand = N_RANDOM if protocol == "test" else N_RANDOM_CV
                # 난수는 (과제, 규약)마다 따로 정한다. 과제나 규약을 일부만 돌려도 이 줄의 값이 달라지지 않는다.
                rng = np.random.default_rng([SEED, zlib.crc32(f"{task}/{protocol}".encode("utf-8"))])
                for _ in range(n_rand):
                    Xr = rng.standard_normal((len(y), RANDOM_DIM))
                    pr, _pm = run_multiclass(Xr, y, folds)
                    vals.append(balanced_acc(y[ev_rows], pr[ev_rows], ev_classes))
                res.append(dict(task=task, protocol=protocol, feature=RANDOM_LABEL, subset="all", n_random=n_rand,
                                metric="balanced_acc", value=float(np.mean(vals)), ci_lo=float(np.percentile(vals, 2.5)),
                                ci_hi=float(np.percentile(vals, 97.5)), chance=1 / len(ev_classes), n_eval=len(ev_rows),
                                n_sites=len(set(st[ev_rows])), n_classes=len(ev_classes), classes=";".join(ev_classes),
                                dropped=";".join(dropped)))
            log(f"[{task}/{protocol}] 범주 {len(ev_classes)}개, 평가 컷 {len(ev_rows)}, 해역 {len(set(st[ev_rows]))}  ({time.time() - t0:.0f}초)")

        # ---- 골드 라벨로 형태 과제 평가 (있을 때만) ----------------------------------------
        if gold is not None and task == "type_single" and "test" in protocols:
            g = gold[(gold.get("usable", "") != "n") & (gold.dominant_type.isin(TYPE_CLASSES))]
            gmap = dict(zip(g.cut_id, g.dominant_type))
            d_tr = df[df.split == "dev"]
            d_ev = core[(core.split == "test") & core.cut_id.isin(gmap)]
            if len(d_ev) >= 2 * MIN_EVAL:
                ytr = d_tr.task_type_single.to_numpy()
                yev = d_ev.cut_id.map(gmap).to_numpy()
                tr_idx = np.array([rows_in_core[c] for c in d_tr.cut_id])
                ev_idx = np.array([rows_in_core[c] for c in d_ev.cut_id])
                cls = [c for c in sorted(set(ytr)) if (ytr == c).sum() >= MIN_TRAIN and (yev == c).sum() >= MIN_EVAL]
                if len(cls) < 2:
                    log(f"[type_gold/test] 학습 {MIN_TRAIN}컷·평가 {MIN_EVAL}컷 이상인 형태가 {len(cls)}개라 건너뛴다")
                if len(cls) >= 2:
                    ktr, kev = np.isin(ytr, cls), np.isin(yev, cls)
                    for n in feats:
                        Xtr, Xev = fit_transform_fold(Xall[n][tr_idx[ktr]], Xall[n][ev_idx[kev]])
                        clf = make_clf().fit(Xtr, ytr[ktr])
                        pr = clf.predict(Xev)
                        st_g = d_ev.g_site.to_numpy()[kev]
                        ba = balanced_acc(yev[kev], pr, cls)
                        lo, hi = site_bootstrap(st_g, lambda idx: balanced_acc(yev[kev][idx], pr[idx], cls), args.boot)
                        res.append(dict(task="type_gold", protocol="test", feature=n, subset="all", metric="balanced_acc",
                                        value=ba, ci_lo=lo, ci_hi=hi, chance=1 / len(cls), n_eval=int(kev.sum()),
                                        n_sites=len(set(st_g)), n_classes=len(cls), classes=";".join(cls), dropped=""))
                    log(f"[type_gold/test] 범주 {len(cls)}개, 평가 컷 {int(kev.sum())}  (학습은 개발 해역의 약한 단일 라벨, 평가는 시험 해역의 골드 대표 형태)")
            else:
                log(f"[type_gold/test] 시험 해역의 골드 표기(쓸 수 있고 대표 형태가 있는 것)가 {len(d_ev)}컷뿐이라 건너뛴다 (최소 {2 * MIN_EVAL}컷)")

    # ---- 진단 지표 --------------------------------------------------------------------------
    diag = []
    site_all = core.g_site.to_numpy()
    sp_all = core.species.to_numpy()
    logfs = np.log10(core.fs_hz.astype(float).to_numpy())
    logduration = np.log10(np.maximum(core.dur_s.astype(float).to_numpy(), 1e-9))
    for n in feats:
        X = Xall[n]
        Z, _ = fit_transform_fold(X, X[:1])                      # 진단용: 전체로 맞춘다 (평가 지표가 아니다)
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
        S = Zn @ Zn.T
        np.fill_diagonal(S, -np.inf)
        nn = S.argmax(1)
        same_site = float(np.mean(site_all[nn] == site_all))
        cnt = pd.Series(site_all).value_counts()
        base_site = float(((cnt * (cnt - 1)).sum()) / (len(core) * (len(core) - 1)))
        S2 = S.copy()
        S2[site_all[:, None] == site_all[None, :]] = -np.inf     # 다른 해역에서만 찾는다
        nn2 = S2.argmax(1)
        same_sp = float(np.mean(sp_all[nn2] == sp_all))
        other = site_all[:, None] != site_all[None, :]
        base_sp = float(np.mean([(sp_all[other[i]] == sp_all[i]).mean() for i in range(len(core))]))
        # fs 복원: 해역 겹으로 능형 회귀
        from sklearn.linear_model import Ridge
        fold_all = core.fold_all.astype(int).to_numpy()
        pr = np.full(len(core), np.nan)
        pr_duration = np.full(len(core), np.nan)
        for k in sorted(set(fold_all)):
            if k < 0:
                continue
            tr, ev = np.flatnonzero(fold_all != k), np.flatnonzero(fold_all == k)
            Xtr, Xev = fit_transform_fold(X[tr], X[ev])
            pr[ev] = Ridge(alpha=1.0).fit(Xtr, logfs[tr]).predict(Xev)
            pr_duration[ev] = Ridge(alpha=1.0).fit(Xtr, logduration[tr]).predict(Xev)
        ok = np.isfinite(pr)
        r2 = 1 - np.sum((logfs[ok] - pr[ok]) ** 2) / np.sum((logfs[ok] - logfs[ok].mean()) ** 2)
        row = dict(feature=n, n=len(core), nn_same_site=same_site, nn_same_site_base=base_site,
                   nn_other_site_same_species=same_sp, nn_other_site_same_species_base=base_sp, r2_log_fs=float(r2))
        ok_duration = np.isfinite(pr_duration)
        denom = np.sum((logduration[ok_duration] - logduration[ok_duration].mean()) ** 2)
        row["r2_log_duration"] = float(1 - np.sum((logduration[ok_duration] - pr_duration[ok_duration]) ** 2) / denom) if denom > 0 else np.nan
        # 비지도 군집 (k-평균, k = 범주 수). 같은 군집 결과를 종·분류군 라벨, 해역, fs 계열과 각각 견준다.
        # 라벨과의 일치도가 해역·fs 계열과의 일치도보다 낮으면, 군집이 녹음 조건을 따라 생긴 것이다.
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_mutual_info_score as ami
        for lab_name, lab in [("species", core.species.to_numpy()), ("group3", core.group3.to_numpy()),
                              ("type", core.task_type_single.to_numpy())]:
            use = lab != ""
            k = len(set(lab[use]))
            if use.sum() < 50 or k < 2:
                continue
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Z[use])
            row[f"kmeans_{lab_name}_k"] = k
            row[f"kmeans_{lab_name}_ami_label"] = float(ami(lab[use], km))
            row[f"kmeans_{lab_name}_ami_site"] = float(ami(site_all[use], km))
            row[f"kmeans_{lab_name}_ami_fsfam"] = float(ami(core.fs_family.to_numpy()[use], km))
        diag.append(row)

    # ---- 저장 -------------------------------------------------------------------------------
    R = pd.DataFrame(res)
    R.to_csv(odir / "results.csv", index=False, encoding="utf-8-sig")
    PC = pd.DataFrame(per_class)
    PC.to_csv(odir / "per_class.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired, columns=["task", "protocol", "A", "B", "diff_balanced_acc", "ci_lo", "ci_hi", "verdict", "n_eval",
                                  "n_sites"]).to_csv(odir / "paired.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diag).to_csv(odir / "diagnostics.csv", index=False, encoding="utf-8-sig")
    if preds:
        pd.concat(preds).to_csv(odir / "predictions.csv", index=False, encoding="utf-8-sig")
    info = dict(run=run, features=feats, tasks=tasks, protocols=protocols, n_core_pool=n_pool, n_core_used=len(core),
                n_core_dropped=n_dropped, coverage=cov,
                type_labels=args.type_labels, exclude_classes=excl,
                pool=args.pool, n_boot=args.boot, n_random_test=0 if args.no_random else N_RANDOM,
                n_random_cv=0 if args.no_random else N_RANDOM_CV, seconds=round(time.time() - t0, 1),
                manifest=Path(args.manifest).name)
    (odir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 요약 보고 ----------------------------------------------------------------------------
    lines = [f"# 평가 요약: {run}", "",
             f"- 평가에 쓴 핵심 표본: {len(core)}컷 (pool={args.pool}, 전체 {n_pool}컷). 특징별로 가진 컷 수: {cov}",
             "- 형태 과제의 라벨: " + ("노트를 읽어 만든 약한 라벨 (v1.1)" if args.type_labels == "note" else "공개본의 낱말 태그 (v1 의 라벨)") + ". 청취로 확인한 정답이 아니다.",
             "- 값은 균형정확도(범주별 재현율의 평균) 또는 범주별 AUROC 의 평균. 대괄호는 평가 해역을 다시 뽑은 95 % 구간.",
             "- `filecv` 는 같은 해역이 학습과 평가에 함께 들어가는 방식이다. `test`·`devcv`·`allcv` 와의 차이가 출처 효과의 크기다.",
             "- 구간이 넓으면 독립 해역 수가 적다는 뜻이다. 구간이 겹치는 두 방법의 순위는 정해지지 않은 것으로 읽는다.",
             f"- `{RANDOM_LABEL}` 행은 정보가 없는 특징으로 같은 절차를 되풀이한 값의 평균과 가운데 95 % 다 (test 200회, 교차검증 40회).", ""]
    if excl:
        lines.insert(4, f"- 형태 과제에서 뺀 범주(--exclude-classes): {', '.join(excl)}. 범주 구성이 기본 실행과 다르므로 값을 한 표에 섞지 않는다.")
    if n_dropped:
        lines.insert(3, f"- **{n_dropped}컷은 모든 특징의 평가에서 빠졌다.** 견주는 특징이 모두 가진 컷만 쓰기 때문이다 "
                        f"(가장 적게 가진 특징: {min(cov, key=cov.get)}). 이 표의 값은 전체 핵심 표본의 값과 다르다.")
    if len(R):
        main_m = R[R.metric.isin(["balanced_acc", "macro_auroc"])]
        for (task, subset), a in main_m.groupby(["task", "subset"], sort=False):
            lines += [f"## {task} ({subset})", "", "| 특징 | " + " | ".join(protocols) + " |",
                      "|---|" + "---|" * len(protocols)]
            for f in feats + [RANDOM_LABEL]:
                cells = []
                for p in protocols:
                    r = a[(a.feature == f) & (a.protocol == p)]
                    if len(r):
                        r = r.iloc[0]
                        ci = f" [{r.ci_lo:.2f}, {r.ci_hi:.2f}]" if np.isfinite(r.ci_lo) else ""
                        cells.append(f"{r.value:.3f}{ci}")
                    else:
                        cells.append("")
                if any(cells):
                    lines.append(f"| {f} | " + " | ".join(cells) + " |")
            r0 = a.iloc[0]
            notes = []
            for p in protocols:
                r = a[a.protocol == p]
                if len(r):
                    r = r.iloc[0]
                    ch = f"우연 {r.chance:.2f}, " if np.isfinite(r.chance) else ""
                    notes.append(f"{p}: {ch}범주 {int(r.n_classes)}개, 컷 {int(r.n_eval)}, 해역 {int(r.n_sites)}")
            lines += ["", "- " + " / ".join(notes)]
            if r0.dropped:
                lines.append(f"- 표본이 모자라 평가에서 빠진 범주({a.protocol.iloc[0]} 기준): {r0.dropped}")
            # 평가 컷이 한 해역에서만 나온 범주는 그 범주의 값이 한 해역의 값이다 (구간에 해역 간 흔들림이 담기지 않는다)
            one = []
            if len(PC) and "n_sites" in PC.columns:
                for p in protocols:
                    q = PC[(PC.task == task) & (PC.subset == subset) & (PC.protocol == p) & (PC.feature == feats[0]) & (PC.n_sites <= 1)]
                    if len(q):
                        one.append(f"{p}: {', '.join(q.cls)}")
            if one:
                lines.append("- 평가 컷이 해역 1개에서만 나온 범주 (그 범주의 값은 한 해역의 값이다): " + " / ".join(one))
            lines.append("")
    if paired:
        lines += ["## 방법 간 차이 (같은 평가 컷, 균형정확도 A − B)", "", "| 과제 | 규약 | A | B | 차이 | 95 % 구간 | 읽는 법 |",
                  "|---|---|---|---|---|---|---|"]
        for r in paired:
            lines.append(f"| {r['task']} | {r['protocol']} | {r['A']} | {r['B']} | {r['diff_balanced_acc']:+.3f} | "
                         f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['verdict']} |")
        lines.append("")
    lines += ["## 진단 지표 (통과 기준이 아니다)", "",
              "| 특징 | 가장 가까운 컷이 같은 해역 (기준율) | 다른 해역에서 찾은 가장 가까운 컷이 같은 종 (기준율) | fs 복원 R² |", "|---|---|---|---|"]
    for d in diag:
        lines.append(f"| {d['feature']} | {d['nn_same_site']:.3f} ({d['nn_same_site_base']:.3f}) | "
                     f"{d['nn_other_site_same_species']:.3f} ({d['nn_other_site_same_species_base']:.3f}) | {d['r2_log_fs']:.3f} |")
    if diag and any(k.startswith("kmeans_") for k in diag[0]):
        lines += ["", "### 비지도 군집 (k-평균, k = 범주 수)의 일치도 AMI. 우연이면 0", "",
                  "| 특징 | 기준 | k | 라벨과 | 해역과 | fs 계열과 |", "|---|---|---|---|---|---|"]
        for d in diag:
            for lab_name in ["group3", "type", "species"]:
                if f"kmeans_{lab_name}_k" in d:
                    lines.append(f"| {d['feature']} | {lab_name} | {d[f'kmeans_{lab_name}_k']} | "
                                 f"{d[f'kmeans_{lab_name}_ami_label']:.3f} | {d[f'kmeans_{lab_name}_ami_site']:.3f} | "
                                 f"{d[f'kmeans_{lab_name}_ami_fsfam']:.3f} |")
    (odir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n".join(lines))
    log(f"\n저장: {odir}  ({time.time() - t0:.0f}초)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
