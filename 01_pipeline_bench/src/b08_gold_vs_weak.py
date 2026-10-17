# -*- coding: utf-8 -*-
"""8단계. 연구자의 골드 표기와 메타데이터의 약한 라벨을 견준다. 음원은 읽지 않는다.

'노트에서 만든 라벨을 어디까지 믿을 수 있는가'에 답하는 표를 만든다. 약한 라벨은 두 가지를 함께 견준다.
  note      노트를 구절 단위로 읽어 만든 라벨 (v1.1 의 과제 라벨. type_weak, type_single)
  tags_v1   공개본의 낱말 태그를 옮긴 라벨 (v1 의 과제 라벨. type_tags)

  1) 형태별로: 약한 라벨이 있다고 한 컷 가운데 골드도 있다고 한 비율(정밀도), 골드가 있다고 한 컷 가운데 약한 라벨도 있던 비율(재현율)
  2) 대표 형태와 약한 단일 라벨의 혼동표
  3) 노트의 낱말마다: 그 낱말이 적힌 컷에서 골드가 고른 형태와 물리적 모양 (대응표 note_term_map.csv 의 해석이 맞는지 본다)
  4) 분류군·종별 사용 가능 비율, 방해음의 빈도
  5) 서로 어긋난 컷의 목록 (b09 --list 로 바로 다시 열 수 있다)

쓰는 법  python src/b08_gold_vs_weak.py --gold <review 화면에서 저장한 CSV> [--gold2 <날짜를 두고 다시 표기한 CSV>]
        연습 화면과 시험 화면에서 따로 저장한 파일은 쉼표로 잇거나 띄어 써서 함께 준다:  --gold 연습.csv,시험.csv   또는   --gold 연습.csv 시험.csv
출력    work/gold/weak_vs_gold_summary.csv, weak_vs_gold_by_type.csv, weak_vs_gold_confusion_note.csv, weak_vs_gold_confusion_tags_v1.csv,
        term_vs_gold.csv, gold_usable_by_species.csv, gold_disagreements.csv, (있으면) gold_repeat_agreement.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import pandas as pd

from wb_common import DATA_DIR, TYPE_CLASSES, csv_arg, log, work_dir
from wb_notes import load_term_map, map_row


def as_set(s: str) -> set:
    return {t for t in str(s).split(";") if t}


def load_gold(paths: list, known: set) -> pd.DataFrame:
    """review 화면이 저장한 표기 CSV 를 읽는다. 쉼표로 이은 여러 파일은 차례로 붙인다.
    열이 모자라면 멈추고, 겹친 컷은 뒤의 줄을 쓰고, 모르는 컷은 뺀다."""
    need = ["cut_id", "target_present", "types", "dominant_type", "interference", "usable", "confidence", "memo", "meta_seen"]
    parts = []
    path = ", ".join(paths)
    for one in paths:
        a = pd.read_csv(one, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        miss = [c for c in need if c not in a.columns]
        if miss:
            raise SystemExit(f"골드 CSV 에 필요한 열이 없다: {miss}. review 화면의 '표기 CSV 저장'으로 만든 파일을 그대로 준다 ({one})")
        parts.append(a)
    if not parts:
        raise SystemExit("--gold 에 파일이 없다")
    g = pd.concat(parts, ignore_index=True)
    if "attrs" not in g.columns:
        g["attrs"] = ""
    if g.cut_id.duplicated().any():
        log(f"같은 컷이 여러 번 든 줄 {int(g.cut_id.duplicated().sum())}개: 뒤에 나온 줄을 쓴다 ({path})")
        g = g.drop_duplicates("cut_id", keep="last")
    unknown = ~g.cut_id.isin(known)
    if unknown.any():
        log(f"목록표에 없는 컷 {int(unknown.sum())}개는 뺀다 ({path})")
        g = g[~unknown]
    return g.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, nargs="+", help="표기 CSV. 여러 개면 쉼표로 잇거나 띄어 쓴다")
    ap.add_argument("--gold2", nargs="+", default="", help="같은 컷을 다시 표기한 CSV. 주면 표기의 일관성을 잰다")
    args = ap.parse_args()
    m = pd.read_csv(DATA_DIR / "bench_manifest.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    g = load_gold(csv_arg(args.gold), set(m.cut_id))
    d = g.merge(m[["cut_id", "species", "species_ko", "group3", "fs_family", "split", "g_site", "type_weak", "type_single", "type_terms",
                   "type_unplaced", "note_nonanimal", "note_other_species", "note_flags", "type_tags", "call_tags_raw", "note"]],
                on="cut_id", how="left", validate="one_to_one")
    out = work_dir("gold")
    log(f"골드 표기 {len(d)}컷 (시험 {int((d.split == 'test').sum())}, 개발 {int((d.split == 'dev').sum())})")
    use = d[d.usable != "n"].copy()
    use["gold_set"] = use.types.map(lambda s: ";".join(t for t in TYPE_CLASSES if t in as_set(s)))
    # 약한 라벨 두 가지: (이름, 형태 집합 열, 단일 형태를 주는 함수)
    sources = [("note", "type_weak", lambda r: r.type_single),
               ("tags_v1", "type_tags", lambda r: r.type_tags if len(as_set(r.type_tags)) == 1 else "")]

    rows, summ = [], []
    for name, col, single_of in sources:
        for t in TYPE_CLASSES:
            w = use[col].map(lambda s: t in as_set(s))
            gl = use.types.map(lambda s: t in as_set(s))
            both = int((w & gl).sum())
            rows.append(dict(source=name, type=t, weak_yes=int(w.sum()), gold_yes=int(gl.sum()), both=both,
                             weak_precision=round(both / w.sum(), 3) if w.sum() else "",
                             weak_recall=round(both / gl.sum(), 3) if gl.sum() else ""))
        single = use.assign(weak_single=[single_of(r) for r in use.itertuples(index=False)])
        single = single[(single.weak_single != "") & (single.dominant_type != "")]
        conf = pd.crosstab(single.weak_single, single.dominant_type.rename("gold_dominant"))
        conf.to_csv(out / f"weak_vs_gold_confusion_{name}.csv", encoding="utf-8-sig")
        same_set = [as_set(a) == as_set(b) for a, b in zip(use[col], use.gold_set)]
        has = use[col] != ""
        summ.append(dict(source=name, n_usable=len(use), n_with_weak=int(has.sum()),
                         set_equal=round(float(pd.Series(same_set)[has.to_numpy()].mean()), 3) if has.any() else "",
                         n_single=len(single),
                         single_agree=round(float((single.weak_single == single.dominant_type).mean()), 3) if len(single) else ""))
        if len(single):
            log(f"\n[{name}] 약한 단일 라벨과 골드 대표 형태가 같은 비율: {summ[-1]['single_agree']} (n={len(single)})\n" + conf.to_string())
    by_type = pd.DataFrame(rows)
    by_type.to_csv(out / "weak_vs_gold_by_type.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summ).to_csv(out / "weak_vs_gold_summary.csv", index=False, encoding="utf-8-sig")
    log("\n형태별 (쓸 수 있다고 한 컷만)\n" + by_type.to_string(index=False))
    log("\n요약\n" + pd.DataFrame(summ).to_string(index=False))

    # 노트의 낱말마다 골드가 고른 형태. 대응표의 해석(basis = reading)을 확인하는 표다.
    table = load_term_map()
    acc = {}
    for r in use.itertuples(index=False):
        gs = as_set(r.types)
        for k in as_set(r.type_terms):
            typ, basis = map_row(k, r.group3, table, r.species)
            a = acc.setdefault((k, r.group3, typ, basis), dict(n=0, gold_has_mapped=0, dom=Counter(), attrs=Counter()))
            a["n"] += 1
            a["gold_has_mapped"] += typ in gs
            a["dom"][r.dominant_type or "(없음)"] += 1
            a["attrs"].update(as_set(r.attrs))
    tv = pd.DataFrame([dict(term_key=k, group3=grp, mapped_type=typ, basis=basis, n=a["n"],
                            gold_has_mapped=(a["gold_has_mapped"] if typ in TYPE_CLASSES else ""),
                            gold_dominant=" ".join(f"{x}:{c}" for x, c in a["dom"].most_common()),
                            gold_attrs=" ".join(f"{x}:{c}" for x, c in a["attrs"].most_common()))
                       for (k, grp, typ, basis), a in acc.items()])
    if len(tv):
        tv = tv.sort_values(["basis", "n"], ascending=[True, False])
        tv.to_csv(out / "term_vs_gold.csv", index=False, encoding="utf-8-sig")
        log("\n노트의 낱말마다 골드가 고른 대표 형태 (해석이 들어간 낱말부터)\n" + tv.to_string(index=False))

    us = d.groupby(["group3", "species_ko"]).agg(
        n=("cut_id", "size"), usable=("usable", lambda s: int((s == "y").sum())),
        excluded=("usable", lambda s: int((s == "n").sum())), hold=("usable", lambda s: int((s == "hold").sum())),
        no_target=("target_present", lambda s: int((s == "n").sum())),
        meta_seen=("meta_seen", lambda s: int((s == "y").sum()))).reset_index()
    us.to_csv(out / "gold_usable_by_species.csv", index=False, encoding="utf-8-sig")
    log("\n종별 사용 가능 여부\n" + us.to_string(index=False))
    intf = pd.Series([t for s in d.interference for t in as_set(s)]).value_counts()
    log("\n방해음 빈도: " + str(intf.to_dict()))

    dis = use[[as_set(a) != as_set(b) for a, b in zip(use.type_weak, use.gold_set)]]
    dis[["cut_id", "species_ko", "type_weak", "type_terms", "type_unplaced", "note_nonanimal", "type_tags", "types", "dominant_type",
         "confidence", "memo", "note"]].to_csv(out / "gold_disagreements.csv", index=False, encoding="utf-8-sig")
    log(f"\n노트 기반 라벨과 골드의 형태 집합이 다른 컷: {len(dis)} / {len(use)} → {out / 'gold_disagreements.csv'}")

    if args.gold2:
        g2 = load_gold(csv_arg(args.gold2), set(m.cut_id))
        j = g.merge(g2, on="cut_id", suffixes=("_1", "_2"))
        rep = dict(n=len(j),
                   dominant_same=round(float((j.dominant_type_1 == j.dominant_type_2).mean()), 3) if len(j) else "",
                   types_same=round(float((j.types_1.map(as_set) == j.types_2.map(as_set)).mean()), 3) if len(j) else "",
                   usable_same=round(float((j.usable_1 == j.usable_2).mean()), 3) if len(j) else "")
        pd.DataFrame([rep]).to_csv(out / "gold_repeat_agreement.csv", index=False, encoding="utf-8-sig")
        log(f"\n다시 표기한 {len(j)}컷의 일치: {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
