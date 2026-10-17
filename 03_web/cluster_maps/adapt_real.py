#!/usr/bin/env python3
"""adapt_real.py — Watkins_Claude_Input_Package 의 목록표를 build_maps.py 가 읽는 cuts.csv 로 옮긴다.
   컷은 cut_id 로만 잇는다(행 번호로 잇지 않는다). usage: python3 -I adapt_real.py <package_dir> <out_dir>"""
import sys
from pathlib import Path
import pandas as pd
pkg, out = Path(sys.argv[1]), Path(sys.argv[2])
m = pd.read_csv(pkg / "data/bench_manifest.csv", dtype={"cut_id": str}, low_memory=False)
fa = pd.read_csv(pkg / "data/feature_availability_13693.csv", dtype={"cut_id": str}).set_index("cut_id")
main = set(pd.read_csv(pkg / "data/main_10694_metadata.csv", dtype={"cut_id": str}, usecols=["cut_id"]).cut_id)
core = set(pd.read_csv(pkg / "data/core_3539_metadata.csv", dtype={"cut_id": str}, usecols=["cut_id"]).cut_id)
assert m.cut_id.is_unique and len(m) == 13693 and len(main) == 10694 and len(core) == 3539 and core <= main
KO = {"captive": "사육", "multi_species": "다종 표기", "air": "공중 녹음", "bad_note": "불량 메모"}
def status(r):
    if r.cut_id in main: return "eval", ""
    if r.pool == "main": return "hold", " + ".join(KO.get(k, k) for k in str(r.hold_reason).split(";"))
    return "excluded", {"excluded_user": "참고래 (연구자 결정)", "lf": "저주파 별도 풀 (fs 5 kHz 미만)"}.get(r.pool, "기타 (해달)")
s = [status(r) for r in m.itertuples()]
c = pd.DataFrame({"cut_id": m.cut_id, "species": m.species_ko.fillna(m.species), "taxon": m.group3_ko, "site": m.g_site,
                  "fs": m.fs_hz, "dur_s": m.dur_s, "status": [x[0] for x in s], "status_reason": [x[1] for x in s],
                  "type_single": m.type_single.fillna(""), "in_sample": m.cut_id.isin(core).astype(int),
                  "split": m.split.where(m.cut_id.isin(main), ""), "fsfam": m.fs_family,
                  "feat": fa.loc[m.cut_id, "feature_scope"].map({"main_10694": "main", "aux_lf87": "lf87"}).fillna("none").to_numpy()})
assert (c.status == "eval").sum() == 10694 and (c.status == "hold").sum() == 2315
c.to_csv(out / "cuts.csv", index=False)
print(c.status.value_counts().to_dict(), c.feat.value_counts().to_dict(), "sample", int(c.in_sample.sum()),
      "split(sample)", c[c.in_sample == 1].split.value_counts().to_dict(), "species(eval)", c[c.status == "eval"].species.nunique(), "sites(eval)", c[c.status == "eval"].site.nunique())
print(pd.crosstab(c.status_reason, c.feat))
