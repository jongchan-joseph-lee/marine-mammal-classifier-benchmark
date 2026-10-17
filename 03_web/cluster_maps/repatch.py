#!/usr/bin/env python3
"""repatch.py — 투영 좌표는 그대로 두고 군집·이웃 수치만 다시 계산해 data.json 에 넣는다 (build_maps.py 의 같은 함수를 쓴다)."""
import sys, json
import numpy as np, pandas as pd
import build_maps as B
cuts_csv, feat, feat_s, path = sys.argv[1:5]
cuts = pd.read_csv(cuts_csv, dtype={"cut_id": str}, keep_default_na=False)
id2row = {c: i for i, c in enumerate(cuts.cut_id)}
d = json.load(open(path))
C = d["cuts"]
lab = {"taxon": np.array(C["tx"]), "species": np.array(C["sp"]), "site": np.array(C["site"]), "fsfam": np.array(C["fsf"]), "type": np.array(C["ty"])}
smp = np.array(C["smp"])
for r in d["reps"]:
    key = r["key"]
    full = B.load_rep(feat, key, id2row)
    for scope in ["full", "sample"]:
        if key not in d[scope]:
            continue
        if scope == "full":
            rows, X = full[0], full[1]
        elif full is not None:
            rows, X = full[0][smp[full[0]] == 1], full[1][smp[full[0]] == 1]
        else:
            s = B.load_rep(feat_s, key, id2row); rows, X = s[0][smp[s[0]] == 1], s[1][smp[s[0]] == 1]
        P, _ = B.preprocess(X)
        km, hd, m, _ = B.structure(P, {k: v[rows] for k, v in lab.items()})
        old = d[scope][key]["metrics"]
        assert abs(old["nn"]["species"] - m["nn"]["species"]) < 1e-9 and d[scope][key]["km"] == B.b64(km, "<u1"), (key, scope)
        d[scope][key]["metrics"] = m
        B.log(key, scope, "다른 출처 이웃:", {k: round(v, 3) for k, v in m["nn_x"].items()}, "기대값", {k: round(v, 3) for k, v in m["base_x"].items()})
open(path, "w", encoding="utf-8").write(json.dumps(d, ensure_ascii=False, separators=(",", ":")))
