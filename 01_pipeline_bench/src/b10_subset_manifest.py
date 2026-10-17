# -*- coding: utf-8 -*-
"""보조. 임의의 컷 목록으로 평가용 목록표를 만든다 (예: GPT 가 뽑은 1,005컷 표본을 그대로 진단할 때).

핵심 표본과 달리 상한을 다시 적용하지 않는다. 해역 단위 겹(fold_all)을 이 목록 안에서 새로 나눈다.
잠금 시험셋의 구분은 쓰지 않으므로 b07 은 --protocols allcv,filecv 로 돌린다.

쓰는 법  python src/b10_subset_manifest.py --ids <컷 번호 목록 또는 cut_id 열이 있는 CSV> --out work/subset_x.csv [--keep-held]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from b02_make_benchmark import assign_folds
from wb_common import DATA_DIR, log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--keep-held", action="store_true", help="사육·공중·다종·불량 표시가 붙은 컷도 넣는다")
    ap.add_argument("--min-sites", type=int, default=2, help="종 과제에 넣을 최소 해역 수")
    args = ap.parse_args()

    p = Path(args.ids)
    if p.suffix.lower() == ".csv":
        ids = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig").cut_id.tolist()
    else:
        ids = p.read_text(encoding="utf-8").split()
    m = pd.read_csv(DATA_DIR / "bench_manifest.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    s = m[m.cut_id.isin(set(ids))].copy()
    n0 = len(s)
    if not args.keep_held:
        s = s[(s.hold_reason == "") & s.pool.isin(["main", "lf"])]
    log(f"목록 {len(ids)}개 가운데 목록표에 있는 것 {n0}개, 보류 표시를 빼고 {len(s)}개")
    s["type_weak_n"] = s.type_weak_n.astype(int)
    s["type_tags_n"] = s.type_tags_n.astype(int)
    fold = assign_folds(s, args.folds, "subset")
    s["fold_all"] = s.g_site.map(fold).astype(int)
    s["fold"] = s.fold_all
    s["region_key"] = [r if r else "site:" + g for r, g in zip(s.region, s.g_site)]
    rf = assign_folds(s, min(args.folds, s.region_key.nunique()), "subset_region", key="region_key")
    s["fold_region"] = s.region_key.map(rf).astype(int)
    s["split"] = "dev"
    s["pool"] = "main"
    s["task_group3"] = np.where(s.group3 != "Other", s.group3, "")
    s["task_type_single"] = s.type_single                                              # 노트 기반 라벨 (b02 와 같은 규칙)
    s["task_type_multi"] = np.where((s.type_weak != "") & (s.type_incomplete == "") & (s.note_other_sound == ""), s.type_weak, "")
    s["task_type_single_tags"] = np.where(s.type_tags_n == 1, s.type_tags, "")         # v1 의 라벨 (b07 --type-labels tags)
    s["task_type_multi_tags"] = np.where(s.type_tags_n >= 1, s.type_tags, "")
    sites = s.groupby("species").g_site.nunique()
    ok = set(sites[sites >= args.min_sites].index)
    s["task_species"] = np.where(s.species.isin(ok), s.species, "")
    cols = ["cut_id", "file_name", "species", "species_ko", "group3", "region_scope", "fs_hz", "fs_family", "dur_s",
            "pool", "split", "fold", "fold_all", "fold_region", "region", "g_tape", "g_session", "g_campaign", "g_site", "type_weak", "type_weak_n",
            "task_group3", "task_type_single", "task_type_multi", "task_species", "gold", "gold_order", "location", "year", "note",
            "type_single", "type_terms", "type_unplaced", "type_incomplete", "note_nonanimal", "note_other_species", "note_flags",
            "call_tags_raw", "type_tags", "type_tags_n", "task_type_single_tags", "task_type_multi_tags"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    s[cols].to_csv(args.out, index=False, encoding="utf-8-sig")
    t = s.groupby("species").agg(n=("cut_id", "size"), tapes=("g_tape", "nunique"), sites=("g_site", "nunique"),
                                 folds=("fold_all", "nunique")).reset_index()
    log(t.to_string(index=False))
    log(f"해역 {s.g_site.nunique()}개, 겹 {args.folds}개 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
