"""Development-only feature-family ablations, condition audit and review candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from wb_common import log, work_dir

GROUPS = [
    ("eventband", "신호 대역·활동 분포", ["log_sig_f05", "log_sig_f50", "log_sig_f95", "sig_bw_oct",
                                      "sig_bins_frac", "rise_peak_db", "tf_active_frac"]),
    ("spectrum", "1/3 옥타브 대역", []),
    ("temporal", "사건의 시간 구조", ["duty", "log1p_events_per_s", "log_event_dur_med", "log_event_dur_max"]),
    ("tonal", "음조·주파수 변화", ["tonal_conc", "log1p_fm_oct_per_s", "pk_range_oct"]),
    ("impulse", "충격음·클릭 간격", ["log_tr_kurt", "log1p_tr_clicks_per_s", "tr_log_ici_med", "tr_ici_cv"]),
    ("repeat", "반복성", ["mod_lag_s", "mod_strength"]),
]
BASE = ["meta", "mfcc16k", "physv0"]
REFERENCE_PRIORITY = ["ast_zero_dev", "ast_zero", "ast_wavzero_dev", "ast_wavzero", "ast_tile_dev", "ast_tile"]


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(safe_json(value), ensure_ascii=False, indent=2), encoding="utf-8")


def cached_references(folder):
    ast = next((n for n in REFERENCE_PRIORITY if (folder / (n + ".npz")).exists()), None)
    vggish = next((n for n in ("vggish_dev", "vggish") if (folder / (n + ".npz")).exists()), None)
    return [n for n in (ast, vggish) if n]


def atomic_npz(path, **data):
    temp = path.with_suffix(".npz.tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **data)
    temp.replace(path)


def prepare(manifest, destination, include_references=True):
    destination.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    dev = original[(original.pool == "main") & (original.split == "dev")].reset_index(drop=True)
    if "label_origin" in dev:
        dev = dev[dev.label_origin != "human_excluded"].reset_index(drop=True)
    if len(dev) < 50:
        raise ValueError("특징 비교에 필요한 개발 자료가 50컷 미만입니다.")
    dev.to_csv(destination / "study_manifest.csv", index=False, encoding="utf-8-sig")
    source_dir = work_dir("features")
    feature_dir = destination / "features"
    feature_dir.mkdir(exist_ok=True)
    source = source_dir / "physv0.npz"
    with np.load(source, allow_pickle=False) as data:
        ids = data["cut_id"].astype(str)
        values = data["X"].astype(np.float32)
        names = data["names"].astype(str).tolist()
    if values.ndim != 2 or values.shape != (len(ids), len(names)) or len(set(ids)) != len(ids):
        raise ValueError("physv0 특징 파일의 행·열·번호를 확인해 주세요.")
    if len(set(names)) != len(names):
        raise ValueError("physv0 특징 이름이 중복되어 있습니다.")
    groups, assigned = [], set()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    # Derived files contain only development rows, including for unsupervised diagnostics.
    keep_rows = np.isin(ids, dev.cut_id.to_numpy())
    ids, values = ids[keep_rows], values[keep_rows]
    for key, label, requested in GROUPS:
        requested = [n for n in names if n.startswith("rise3oct_")] if key == "spectrum" else requested
        if not requested or any(n not in names for n in requested):
            raise ValueError(f"physv0의 {label} 정의가 현재 구현과 다릅니다.")
        if assigned & set(requested):
            raise ValueError("물리 특징군의 정의가 겹칩니다.")
        assigned.update(requested)
        columns = [i for i, n in enumerate(names) if n not in requested]
        variant = "phys_drop_" + key
        atomic_npz(feature_dir / (variant + ".npz"), cut_id=ids, X=values[:, columns],
                   names=np.array([names[i] for i in columns]),
                   info=json.dumps(dict(source="physv0", source_sha256=source_hash, removed=requested,
                                        split="dev", study="leave-one-family-out"), ensure_ascii=False))
        groups.append(dict(key=key, label=label, feature=variant, removed=len(requested),
                           retained=len(columns), names=requested))
    if assigned != set(names):
        raise ValueError("physv0에 분류되지 않은 특징이 있습니다. 특징군 정의를 갱신해 주세요.")
    references = cached_references(source_dir) if include_references else []
    source_hashes = {}
    for name in BASE + references:
        path = source_dir / (name + ".npz")
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            source_ids = data["cut_id"].astype(str)
            matrix = data["X"]
            if matrix.ndim != 2 or len(source_ids) != len(matrix) or len(set(source_ids)) != len(source_ids):
                raise ValueError(f"{name} 특징의 번호·행을 확인해 주세요.")
            selected = np.isin(source_ids, dev.cut_id.to_numpy())
            atomic_npz(feature_dir / (name + ".npz"), cut_id=source_ids[selected], X=matrix[selected],
                       names=data["names"] if "names" in data else np.array([f"e{i}" for i in range(matrix.shape[1])]),
                       info=data["info"] if "info" in data else "")
    features = BASE + [g["feature"] for g in groups] + references
    common = set(dev.cut_id)
    coverage = {}
    hashes = {}
    for name in features:
        p = feature_dir / (name + ".npz")
        with np.load(p, allow_pickle=False) as n:
            present = set(n["cut_id"].astype(str))
        coverage[name] = len(present & set(dev.cut_id))
        common &= present
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    if len(common) < 50:
        raise ValueError("모든 특징에 공통으로 있는 개발 자료가 50컷 미만입니다. 레퍼런스 특징의 계산 범위를 확인해 주세요.")
    # Correlations are descriptive: a connection with fs can also reflect a biological difference.
    from scipy.stats import spearmanr
    meta = dev.set_index("cut_id").reindex(ids)
    logfs = np.log10(meta.fs_hz.astype(float).to_numpy())
    logdur = np.log10(np.maximum(meta.dur_s.astype(float).to_numpy(), 1e-9))
    audit = []
    for i, name in enumerate(names):
        finite = np.isfinite(values[:, i])
        row = dict(feature=name, group=next(g["label"] for g in groups if name in g["names"]),
                   n_dev=len(ids), finite_fraction=float(finite.mean()),
                   unique_values=len(np.unique(values[finite, i])))
        for label, target in (("rho_log_fs", logfs), ("rho_log_duration", logdur)):
            ok = finite & np.isfinite(target)
            row[label] = float(spearmanr(values[ok, i], target[ok]).statistic) if (ok.sum() >= 10
                         and row["unique_values"] > 1 and np.ptp(target[ok]) > 0) else np.nan
        audit.append(row)
    pd.DataFrame(audit).to_csv(destination / "feature_audit.csv", index=False, encoding="utf-8-sig")
    plan = dict(features=features, groups=groups, references=references, source_sha256=source_hash,
                feature_sha256=hashes, source_feature_sha256=source_hashes, n_dev=len(dev), n_common=len(common), coverage=coverage,
                held_out_test_used=False, human_dev=int((dev.get("label_origin", "") == "human_review").sum())
                if "label_origin" in dev else 0,
                note="동일한 개발 표본·해역 겹·분류기로 비교. 라벨 미확인 컷에는 약한 라벨이 남음.")
    write_json(destination / "study_plan.json", plan)
    log(f"특징군 비교 준비: 개발 {len(dev)}컷, 공통 {len(common)}컷, 특징 조건 {len(features)}개")
    return plan


def summarize(plan_path, result_dir):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    results = pd.read_csv(result_dir / "results.csv", keep_default_na=False)
    paired = pd.read_csv(result_dir / "paired.csv", keep_default_na=False)
    main = results[(results.subset == "all") & (results.protocol == "devcv")
                   & results.metric.isin(["balanced_acc", "macro_auroc"])]
    effects = []
    for _, full in main[main.feature == "physv0"].iterrows():
        for group in plan["groups"]:
            drop = main[(main.task == full.task) & (main.metric == full.metric) & (main.feature == group["feature"])]
            if drop.empty:
                continue
            lower, upper = None, None
            compare = paired[(paired.task == full.task) & (paired.protocol == "devcv")
                             & (((paired.A == "physv0") & (paired.B == group["feature"]))
                                | ((paired.B == "physv0") & (paired.A == group["feature"])))]
            if not compare.empty:
                p = compare.iloc[0]
                if p.ci_lo != "" and p.ci_hi != "":
                    lower, upper = float(p.ci_lo), float(p.ci_hi)
                    if p.A != "physv0":
                        lower, upper = -upper, -lower
            difference = float(full.value) - float(drop.iloc[0].value)
            conclusion = "점수 차이만 확인" if lower is None else "차이 불확실"
            if lower is not None and math.isfinite(lower) and lower > 0:
                conclusion = "빼면 성능 저하"
            elif upper is not None and math.isfinite(upper) and upper < 0:
                conclusion = "빼면 성능 상승"
            if lower is None:
                conclusion = "점수 차이만 확인"
            effects.append(dict(task=full.task, metric=full.metric, group=group["label"], feature=group["feature"],
                                removed=group["removed"], full=float(full.value), without=float(drop.iloc[0].value),
                                contribution=difference, ci_lo=lower, ci_hi=upper, conclusion=conclusion))
    pd.DataFrame(effects, columns=["task", "metric", "group", "feature", "removed", "full", "without",
                                  "contribution", "ci_lo", "ci_hi", "conclusion"]).to_csv(
                                      result_dir / "feature_ablation.csv", index=False, encoding="utf-8-sig")
    manifest = pd.read_csv(plan_path.parent / "study_manifest.csv", dtype=str, keep_default_na=False).set_index("cut_id")
    candidates = []
    prediction_path = result_dir / "predictions.csv"
    if prediction_path.exists():
        pred = pd.read_csv(prediction_path, dtype={"cut_id": str})
        pred = pred[(pred.task == "type_single") & (pred.protocol == "devcv")
                    & pred.feature.isin(["physv0", "mfcc16k", *plan["references"]])]
        for cut_id, rows in pred.groupby("cut_id", sort=False):
            if cut_id not in manifest.index or manifest.at[cut_id, "split"] != "dev":
                continue
            errors = rows[rows["true"] != rows["pred"]]
            disagree = rows["pred"].nunique() > 1
            if errors.empty and not disagree:
                continue
            candidates.append(dict(cut_id=cut_id, current_label=rows.iloc[0]["true"],
                                   label_origin=manifest.at[cut_id, "label_origin"] if "label_origin" in manifest else "weak_note",
                                   n_mismatch=len(errors), model_disagreement=bool(disagree),
                                   max_mismatch_score=float(errors.p_max.max()) if len(errors) else 0.0,
                                   predictions=";".join(f"{r.feature}:{r.pred}" for r in rows.itertuples())))
    candidates.sort(key=lambda c: (c["label_origin"] == "weak_note", c["n_mismatch"], c["max_mismatch_score"]), reverse=True)
    pd.DataFrame(candidates, columns=["cut_id", "current_label", "label_origin", "n_mismatch", "model_disagreement",
                                      "max_mismatch_score", "predictions"]).to_csv(
                                          result_dir / "review_candidates.csv", index=False, encoding="utf-8-sig")
    diagnostics = pd.read_csv(result_dir / "diagnostics.csv", keep_default_na=False).to_dict("records")
    study = dict(plan=plan, effects=effects, diagnostics=diagnostics, candidate_count=len(candidates),
                 reference_evaluated=plan["references"], label_scope="development: weak + completed human corrections")
    write_json(result_dir / "study_summary.json", study)
    lines = ["# 다음 단계: 특징군 비교", "", f"- 개발 자료 {plan['n_dev']}컷, 모든 특징에 공통인 자료 {plan['n_common']}컷.",
             f"- 완료한 사람 표기로 형태 라벨을 수정한 개발 자료: {plan['human_dev']}컷.",
             "- 시험 자료는 이 비교와 녹음 조건·군집 진단에 사용하지 않았습니다.",
             "- 물리 특징은 원래 샘플링에서 계산한 기존 값을 사용합니다. 특징군을 하나씩 제거하고 같은 분류기로 평가합니다.",
             "- AST/VGGish가 있는 경우 동결 임베딩에 같은 분류기를 적용합니다. 네트워크 자체의 재학습이나 멀티태스크 학습이 아닙니다.",
             "- 확인하지 않은 컷에는 약한 라벨이 남습니다. 30개 육안 확인만으로 전체 정확도를 확정할 수 없습니다.", "",
             "| 과제 | 제거한 특징군 | 전체 − 제거 점수 | 95% 차이 구간 | 해석 |", "|---|---|---:|---|---|"]
    for row in effects:
        ci = "점수 차이만" if row["ci_lo"] is None else f"{row['ci_lo']:+.3f} ~ {row['ci_hi']:+.3f}"
        lines.append(f"| {row['task']} | {row['group']} | {row['contribution']:+.3f} | {ci} | {row['conclusion']} |")
    lines += ["", "양수는 해당 특징군을 뺐을 때 성능이 내려갔다는 뜻입니다. 개별 특징의 인과적 기여나 최적 조합을 뜻하지 않습니다.",
              "다중 형태 AUROC는 차이의 점 추정만 표시합니다. 균형정확도의 차이 구간은 같은 평가 해역을 함께 재추출한 값입니다.",
              "", f"다시 들을 개발 자료 후보: {len(candidates)}컷 (`review_candidates.csv`). 자동 후보와 현재 라벨의 불일치이며 실제 오분류라고 확정한 것은 아닙니다.",
              "", "fs 복원 R², 같은 해역 최근접 이웃 비율, 군집의 라벨·해역·fs 계열 일치도는 진단 지표입니다.",
              "`feature_audit.csv`의 주파수·길이 상관은 생물학적 차이도 반영할 수 있으므로 녹음 편향의 증명으로 사용하지 않습니다.",
              "", "다음 작업은 불일치 후보의 소리 확인 → 라벨 수정 → 같은 비교 재실행 → 반복해서 약한 특징군을 보완하는 순서입니다."]
    (result_dir / "feature_study_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"특징군 비교 요약: 차이 {len(effects)}줄, 재검사 후보 {len(candidates)}컷")
    return study


def main():
    parser = argparse.ArgumentParser(description="개발 자료에서 특징군 제거 비교")
    parser.add_argument("action", choices=["prepare", "summarize"])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--no-references", action="store_true")
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.manifest:
            parser.error("prepare에는 --manifest가 필요합니다")
        prepare(args.manifest, args.out, not args.no_references)
    else:
        if not args.plan:
            parser.error("summarize에는 --plan이 필요합니다")
        summarize(args.plan, args.out)


if __name__ == "__main__":
    main()
