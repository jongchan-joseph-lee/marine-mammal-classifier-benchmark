#!/usr/bin/env python3
"""
make_mock.py — 모의 자료(synthetic data) 생성기. 화면과 계산 흐름을 확인하는 용도다.
컷 수와 구성만 실제(13,693 → 10,694 → 3,539)에 맞췄고, 특징값은 난수로 만든 것이다.
종·출처 이름도 일부러 가짜 이름을 쓴다. 여기서 나온 그림과 수치는 Watkins 자료의 결과가 아니다.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

out = Path(sys.argv[1]); (out / "features").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)

# 평가 대상 10,694 = 이빨 8,820 + 수염 1,385 + 기각 489, 36종·49출처
taxa = [("이빨고래류", "이빨고래", 20, 8820, 31), ("수염고래류", "수염고래", 7, 1385, 12), ("기각류", "기각류", 9, 489, 6)]
fs_choices = {"이빨고래류": [20000, 40000, 44100, 81920, 96000, 166600, 192000],
              "수염고래류": [5120, 8000, 10000, 14900, 20000], "기각류": [10000, 20000, 40000, 44100]}
types = {"이빨고래류": ["whistle", "click", "pulse_train", "pulsed_call"], "수염고래류": ["moan", "song", "pulsed_call", "growl"],
         "기각류": ["growl", "bark", "song", "moan"]}
rows, site_no = [], 0
for tname, sname, nsp, ncut, nsite in taxa:
    w = rng.dirichlet(np.ones(nsp) * 0.6); cnt = np.maximum(6, (w * ncut).astype(int)); cnt[0] += ncut - cnt.sum()
    sites = [f"출처-{site_no + i + 1:02d}" for i in range(nsite)]; site_no += nsite
    for s in range(nsp):
        own = rng.choice(sites, size=min(len(sites), 1 + rng.poisson(1.6)), replace=False)
        pw = rng.dirichlet(np.ones(len(own)) * 0.5)
        fs_sp = rng.choice(fs_choices[tname], size=2)
        for _ in range(cnt[s]):
            site = rng.choice(own, p=pw)
            rows.append(dict(species=f"{sname}-{s + 1:02d}", taxon=tname, site=site,
                             fs=int(fs_sp[hash(site) % 2]), status="eval", status_reason="",
                             type_single=rng.choice(types[tname]) if rng.random() < 0.354 else ""))
ev = pd.DataFrame(rows)
ev = ev.drop(ev.sample(len(ev) - 10694, random_state=0).index).reset_index(drop=True)

def extra(n, status, reason, taxon, species, site, fs):
    return pd.DataFrame(dict(species=species, taxon=taxon, site=site, fs=fs, status=status, status_reason=reason,
                             type_single=""), index=range(n))
hold_src = ev.sample(2315, random_state=1).copy()
hold_src["status"] = "hold"
hold_src["status_reason"] = rng.choice(["사육", "다종 표기", "공중 녹음", "불량 메모"], size=2315, p=[2018 / 2315, 174 / 2315, 114 / 2315, 9 / 2315])
hold_src["site"] = np.where(hold_src["status_reason"] == "사육", "출처-사육", hold_src["site"])
hold_src["type_single"] = ""
ex = pd.concat([extra(567, "excluded", "참고래", "수염고래류", "참고래(제외)", "출처-참고래", 5120),
                extra(115, "excluded", "저주파 별도", "수염고래류", "저주파 종", "출처-저주파", 2000),
                extra(2, "excluded", "해달", "기타", "해달", "출처-해달", 44100)])
df = pd.concat([ev, hold_src, ex], ignore_index=True)
df.insert(0, "cut_id", [f"M{i:05d}" for i in range(len(df))])
df["dur_s"] = np.round(np.exp(rng.normal(1.3, 1.0, len(df))).clip(0.2, 120), 2)

# 출처 분리 분할과 표본(종 × 출처 상한 40)
site_list = df.loc[df.status == "eval", "site"].unique()
test_sites = set(rng.choice(site_list, size=20, replace=False))
df["split"] = np.where(df.status != "eval", "", np.where(df.site.isin(test_sites), "test", "dev"))
df["in_sample"] = 0
e = df[df.status == "eval"]
for cap in range(20, 2000):
    pick = np.concatenate([g.sample(min(len(g), cap), random_state=3).index.to_numpy() for _, g in e.groupby(["species", "site"])])
    if len(pick) >= 3539:
        break
pick = rng.permutation(pick)[:3539]
df.loc[pick, "in_sample"] = 1
df.to_csv(out / "cuts.csv", index=False)
print(len(df), df.status.value_counts().to_dict(), "sample", int(df.in_sample.sum()))

# 잠재 변수: 분류군 + 종 + 출처(강함) + 샘플링률 + 잡음
L = 14
def emb(keys, scale):
    u = {k: rng.normal(0, scale, L) for k in pd.unique(keys)}
    return np.stack([u[k] for k in keys])
lat = {"taxon": emb(df.taxon, 2.2), "species": emb(df.species, 1.5), "site": emb(df.site, 1.8),
       "type": emb(df.type_single.replace("", "none"), 0.9)}
fsdir = rng.normal(0, 1, L); lfs = (np.log10(df.fs.to_numpy()) - 4.4)[:, None] * fsdir
mix = {"phys": (1.0, 0.9, 0.8, 0.6, 2.6, 1.6, 51), "mfcc": (0.8, 0.7, 1.1, 0.3, 1.2, 1.8, 80),
       "vggish": (0.8, 0.8, 1.2, 0.4, 1.0, 1.7, 128), "ast_zero": (1.3, 1.0, 1.5, 0.5, 1.6, 1.3, 768),
       "ast_tile": (1.4, 1.1, 1.4, 0.5, 1.4, 1.3, 768)}
for name, (a, b, c, d, f, nz, dim) in mix.items():
    z = a * lat["taxon"] + b * lat["species"] + c * lat["site"] + d * lat["type"] + f * lfs + rng.normal(0, nz, (len(df), L))
    X = np.tanh(z @ rng.normal(0, 1 / np.sqrt(L), (L, dim))) + rng.normal(0, 0.35, (len(df), dim))
    np.savez_compressed(out / "features" / f"{name}.npz", cut_id=df.cut_id.to_numpy(), X=X.astype(np.float32))
np.savez_compressed(out / "features" / "meta.npz", cut_id=df.cut_id.to_numpy(),
                    X=np.c_[np.log10(df.fs), np.log10(df.dur_s)].astype(np.float32))
