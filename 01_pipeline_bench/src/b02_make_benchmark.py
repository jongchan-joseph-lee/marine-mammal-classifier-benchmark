# -*- coding: utf-8 -*-
"""2단계. 벤치마크(핵심 표본, 개발/시험 분할, 과제, 골드 검토 목록)를 만든다. 음원은 읽지 않는다.

입력  data/manifest_all.csv
      data/source/design_v1.csv.gz   v1 의 설계(역할·분할·겹·골드 목록). 이 스크립트의 결과가 이것과 같은지 확인한다
출력  data/bench_manifest.csv        13,693행 전체 + 역할(role)·분할(split)·겹(fold)·과제 열
      data/bench_core.csv            평가에 쓰는 핵심 표본만 (bench_manifest 의 부분집합)
      data/gold_review_list.csv      연구자가 듣고 확인할 목록 (연습 30 + 시험 360, 우선순위 순)
      data/tables/*.csv              종·분류군·형태·해역 요약표, 라벨이 바뀐 컷의 목록
      data/bench_summary.json        수량과 점검 결과

설계 (docs/protocol.md 와 같아야 한다)
  - 분할 단위는 '해역 묶음'(g_site). 시험 해역은 개발 자료에 없는 해역이다.
  - 핵심 표본은 (종 × 현장 묶음)마다 최대 CAP 개. 큰 테이프가 결과를 좌우하지 않게 한다.
  - 시험 해역은 고정 씨앗의 무작위 탐색으로 고른다. 목표는 분류군·형태·fs 계열의 시험 비율을 TEST_FRAC 에 가깝게,
    해역이 2개 이상인 종은 개발과 시험 양쪽에 해역이 있게 하는 것이다.
  - 시험 해역에 속한 컷은 핵심 표본이 아니어도 학습에 쓰지 않는다 (role = test_reserve).

v1.1 에서 바뀐 것: 형태 과제의 라벨만 바뀌었다.
  - 표본 설계(핵심 표본 고르기, 시험 해역 고르기, 겹 나누기, 골드 목록 뽑기)는 v1 과 똑같이 낱말 태그(type_tags)로 한다.
    그래서 역할·분할·겹·골드 목록은 v1 과 한 칸도 다르지 않다 (끝에서 design_v1.csv.gz 와 견주어 확인한다).
  - 형태 과제의 라벨(task_type_single, task_type_multi)은 노트를 읽어 만든 라벨(type_weak, type_single)에서 온다.
    v1 의 라벨은 task_type_single_tags, task_type_multi_tags 로 남겨 두어 같은 조건에서 견줄 수 있다.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from wb_common import DATA_DIR, GROUPS3, SEED, TYPE_CLASSES, VERSION, log, sha256_file, stable_key, write_json

CAP = 40                 # (종 × 현장 묶음)당 핵심 표본 상한
TEST_FRAC = 0.30         # 시험에 두려는 비율
N_FOLDS = 5              # 개발 자료의 겹 수 (해역 단위)
N_TEST_FOLDS = 2         # 전 해역 교차검증(참고값)에서 시험 해역을 나누는 겹 수
N_REGION_FOLDS = 6       # 구역 단위 교차검증(참고값)의 겹 수
MIN_DUR, MAX_DUR = 0.25, 60.0   # 핵심 표본에 넣는 길이 범위(초). 범위 밖은 예비로 둔다
N_SEARCH = 40000         # 시험 해역 탐색 횟수
SPECIES_MIN_SITES = 3    # 종 과제에 넣는 최소 해역 수 (개발 2 + 시험 1)
MAX_SITE_SHARE = 0.30    # 한 해역이 한 분류군의 시험 컷에서 차지해도 되는 몫
MAX_TEST_SITE_CORE = 100 # 핵심 표본이 이보다 많은 거점 해역(여러 해·여러 종이 모인 곳)은 개발에 둔다
GOLD_TEST, GOLD_PRACTICE = 360, 30
PILOT_AUDIO_LIST = DATA_DIR / "source" / "audio_in_gpt_packages.txt"   # 이 세션에서 음원을 가진 컷 (있으면 표시만 한다)


def round_robin(rows: pd.DataFrame, bucket_cols: list[str], cap: int) -> list[str]:
    """버킷을 돌아가며 하나씩 뽑는다. 버킷 순서와 버킷 안 순서는 고정 해시로 정한다."""
    buckets = defaultdict(list)
    for r in rows.itertuples(index=False):
        buckets[tuple(getattr(r, c) for c in bucket_cols)].append(r)
    for k in buckets:
        buckets[k].sort(key=lambda r: (r.prio, stable_key("core", r.cut_id)))
    names = sorted(buckets, key=lambda k: stable_key("bucket", *k))
    out, ptr = [], {k: 0 for k in names}
    while len(out) < cap:
        progressed = False
        for k in names:
            if ptr[k] < len(buckets[k]) and len(out) < cap:
                out.append(buckets[k][ptr[k]].cut_id)
                ptr[k] += 1
                progressed = True
        if not progressed:
            break
    return out


def select_core(m: pd.DataFrame) -> pd.Series:
    """role 열을 만든다: core / reserve / held / excluded."""
    role = pd.Series("excluded", index=m.index)
    role[m.pool.isin(["main", "lf"]) & (m.hold_reason != "")] = "held"
    elig = m[m.eligible].copy()
    in_len = (elig.dur_s >= MIN_DUR) & (elig.dur_s <= MAX_DUR)
    # 우선순위: 길이 범위 안 → '좋은 컷' 노트 → 잡음 노트 없음. 숫자가 작을수록 먼저.
    elig["prio"] = (~in_len).astype(int) * 4 + (~elig.f_good_note).astype(int) * 1 + elig.f_noise_note.astype(int) * 2
    role[elig.index] = "reserve"
    chosen = []
    for (_sp, _camp), cell in elig[in_len].groupby(["species", "g_campaign"], sort=True):
        chosen += round_robin(cell, ["g_tape", "type_tags", "fs_hz"], CAP)
    role[m.cut_id.isin(set(chosen))] = "core"
    return role


def choose_test_sites(core: pd.DataFrame) -> tuple[set, dict]:
    """고정 씨앗의 무작위 탐색으로 시험 해역을 고른다."""
    sites = sorted(core.g_site.unique())
    idx = {s: i for i, s in enumerate(sites)}
    n = len(sites)
    site_i = core.g_site.map(idx).to_numpy()

    def site_matrix(labels: pd.Series, classes: list) -> np.ndarray:
        """해역 × 범주의 컷 수."""
        mat = np.zeros((n, len(classes)))
        ci = {c: j for j, c in enumerate(classes)}
        for s, lab in zip(site_i, labels):
            for one in (lab if isinstance(lab, (list, tuple)) else [lab]):
                if one in ci:
                    mat[s, ci[one]] += 1
        return mat

    grp = site_matrix(core.group3, GROUPS3)
    fam = site_matrix(core.fs_family, sorted(core.fs_family.unique()))
    typ = site_matrix(core.type_tags.map(lambda s: [t for t in s.split(";") if t]), TYPE_CLASSES)
    cells = sorted(set(zip(core.group3, core.fs_family)))
    cell = site_matrix(pd.Series(list(zip(core.group3, core.fs_family)), index=core.index).map(lambda t: [t]), cells)
    species = sorted(core.species.unique())
    spc = site_matrix(core.species, species)
    grp_site = (grp > 0).astype(float)            # 해역 수 기준
    sp_site = spc > 0
    multi_site_sp = np.where(sp_site.sum(0) >= 2)[0]

    def score(mask: np.ndarray) -> tuple[float, dict]:
        parts = {}
        f = lambda mat: (mat[mask].sum(0) + 1e-9) / (mat.sum(0) + 1e-9)
        parts["group_cuts"] = float(((f(grp) - TEST_FRAC) ** 2).sum())
        parts["group_sites"] = float(((f(grp_site) - TEST_FRAC) ** 2).sum())
        keep = typ.sum(0) >= 100                    # 드문 형태(bark, song)는 균형 목표에서 뺀다
        parts["type_cuts"] = float(((f(typ)[keep] - TEST_FRAC) ** 2).sum())
        parts["fs_family_cuts"] = float(((f(fam) - TEST_FRAC) ** 2).sum()) * 0.5
        # 한 해역이 그 분류군 시험 컷의 MAX_SITE_SHARE 를 넘으면 벌점 (시험 결과가 한 해역에 좌우되지 않게)
        share = grp[mask].max(0) / (grp[mask].sum(0) + 1e-9)
        parts["site_concentration"] = float((np.clip(share - MAX_SITE_SHARE, 0, None) ** 2).sum()) * 4.0
        keep_cell = cell.sum(0) >= 50
        parts["group_x_fs_cells"] = float(((f(cell)[keep_cell] - TEST_FRAC) ** 2).sum()) * 0.5
        n_test = sp_site[mask][:, multi_site_sp].sum(0)
        n_dev = sp_site[~mask][:, multi_site_sp].sum(0)
        # 해역이 2개 이상인 종이 시험에 없거나, 개발에 없거나, (3개 이상인데) 개발에 2개가 안 되면 벌점
        need_dev = np.where(sp_site.sum(0)[multi_site_sp] >= 3, 2, 1)
        viol = int((n_test < 1).sum() + (n_dev < need_dev).sum())
        parts["species_violations"] = viol
        total = (parts["group_cuts"] + parts["group_sites"] + parts["type_cuts"] + parts["fs_family_cuts"]
                 + parts["site_concentration"] + parts["group_x_fs_cells"] + 0.25 * viol)
        return total, parts

    allowed = grp.sum(1) < MAX_TEST_SITE_CORE       # 거점 해역은 시험 후보에서 뺀다
    p_test = TEST_FRAC * grp.sum() / grp[allowed].sum()   # 남은 해역에서 뽑아 전체의 TEST_FRAC 를 채운다
    rng = np.random.default_rng(SEED)
    best, best_mask, best_parts = np.inf, None, None
    for _ in range(N_SEARCH):
        mask = (rng.random(n) < p_test) & allowed
        if not mask.any() or mask.all():
            continue
        s, parts = score(mask)
        if s < best:
            best, best_mask, best_parts = s, mask.copy(), parts
    # 한 해역씩 뒤집어 보는 국소 개선 (결과가 탐색 횟수에 덜 민감해진다)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if not allowed[i]:
                continue
            trial = best_mask.copy()
            trial[i] = ~trial[i]
            if not trial.any() or trial.all():
                continue
            s, parts = score(trial)
            if s < best - 1e-12:
                best, best_mask, best_parts, improved = s, trial, parts, True
    info = dict(score=round(best, 5), **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in best_parts.items()},
                n_sites=n, n_test_sites=int(best_mask.sum()), n_hub_sites_kept_in_dev=int((~allowed).sum()))
    return {s for s, i in idx.items() if best_mask[i]}, info


def assign_folds(core_dev: pd.DataFrame, n_folds: int = N_FOLDS, tag: str = "fold", key: str = "g_site") -> dict:
    """묶음(key, 기본은 해역)을 n_folds 겹으로. 큰 묶음부터, 범주별 비율의 제곱합이 가장 작아지는 겹에 넣는다."""
    cats = []
    for g in GROUPS3:
        cats.append(("g", g))
    for t in TYPE_CLASSES:
        cats.append(("t", t))
    for f in sorted(core_dev.fs_family.unique()):
        cats.append(("f", f))
    for sp in sorted(core_dev.species.unique()):
        cats.append(("s", sp))
    ci = {c: j for j, c in enumerate(cats)}
    sites = sorted(core_dev[key].unique())
    vec = {s: np.zeros(len(cats)) for s in sites}
    for r in core_dev.itertuples(index=False):
        v = vec[getattr(r, key)]
        v[ci[("g", r.group3)]] += 1
        v[ci[("f", r.fs_family)]] += 1
        v[ci[("s", r.species)]] += 1
        for t in r.type_tags.split(";"):
            if t:
                v[ci[("t", t)]] += 1
    tot = sum(vec.values()) + 1e-9
    load = np.zeros((n_folds, len(cats)))
    order = sorted(sites, key=lambda s: (-vec[s].sum(), stable_key(tag, s)))
    fold = {}
    for s in order:
        costs = [(((load[k] + vec[s]) / tot) ** 2).sum() - ((load[k] / tot) ** 2).sum() for k in range(n_folds)]
        k = int(np.argmin(costs))
        fold[s] = k
        load[k] += vec[s]
    return fold


def gold_sample(core: pd.DataFrame, n: int, tag: str) -> list[str]:
    """분류군 × 약한 형태 × fs 계열의 칸을 돌아가며, 칸 안에서는 해역을 돌아가며 뽑는다."""
    df = core.copy()
    df["sig"] = df.type_tags.map(lambda s: s if s and ";" not in s else ("multi" if s else "none"))
    strata = defaultdict(lambda: defaultdict(list))
    for r in df.itertuples(index=False):
        strata[(r.group3, r.sig, r.fs_family)][r.g_site].append(r.cut_id)
    queues = {}
    for k, by_site in strata.items():
        for s in by_site:
            by_site[s].sort(key=lambda c: stable_key(tag, c))
        site_order = sorted(by_site, key=lambda s: stable_key(tag, "site", s))
        q, j = [], 0
        while any(by_site.values()):
            s = site_order[j % len(site_order)]
            if by_site[s]:
                q.append(by_site[s].pop(0))
            j += 1
            if j > 10 ** 6:
                break
        queues[k] = q
    # 분류군을 먼저 돌고(기각류·수염고래가 밀리지 않게), 분류군 안에서 칸을 돈다
    out = []
    by_group = defaultdict(list)
    for k in sorted(queues, key=lambda k: stable_key(tag, "stratum", *k)):
        by_group[k[0]].append(k)
    ptr = {g: 0 for g in by_group}
    while len(out) < n and any(queues[k] for ks in by_group.values() for k in ks):
        for g in GROUPS3:
            ks = by_group.get(g, [])
            for _ in range(len(ks)):
                k = ks[ptr[g] % len(ks)]
                ptr[g] += 1
                if queues[k]:
                    out.append(queues[k].pop(0))
                    break
            if len(out) >= n:
                break
    return out


def why_changed(r) -> str:
    """v1 의 단일 형태(낱말 태그)와 v1.1 의 단일 형태(노트)가 다른 까닭을 한 줄로."""
    a, b = r.task_type_single_tags, r.task_type_single
    unpl = [u for u in r.type_unplaced.split(";") if u]
    generic = [u for u in unpl if u in ("call", "sound+desc")]
    other = [u for u in unpl if u not in ("call", "sound+desc")]
    if a == b:
        return ""
    if r.note_override:
        return "손으로 고침: " + r.note_override
    if a and not b:
        if r.type_weak_n >= 2:
            return "노트에 다른 형태가 더 적혀 있다 (" + r.type_weak + ")"
        if "tape_log" in r.note_flags:
            return "테이프 전체의 일지라 이 컷의 내용인지 알 수 없다"
        if r.note_other_sound:
            return "다른 종의 소리로 적혀 있다 (" + r.note_other_sound + ")"
        if generic and not other:
            return "형태를 말하지 않는 낱말뿐이거나 함께 적혀 있다 (" + ";".join(generic) + ")"
        if other:
            return "8개 범주 밖의 소리가 함께 적혀 있다 (" + ";".join(other) + ")"
        if r.note_nonanimal:
            return "동물 밖 소리로 적혀 있다 (" + r.note_nonanimal + ")"
        return "노트에 대상 동물의 소리 낱말이 없다"
    if b and not a:
        if r.type_tags_n == 0:
            return "v1 태그에 없던 낱말이다 (" + r.type_terms + ")"
        bits = []
        if r.note_nonanimal:
            bits.append("동물 밖 소리 " + r.note_nonanimal)
        if r.note_other_sound:
            bits.append("다른 종 " + r.note_other_sound)
        if "call" in r.call_tags_raw.split(";") and "call" not in r.type_terms.split(";"):
            bits.append("머리말·수식의 call")
        return "v1 태그는 여러 개였으나 대상 동물의 형태는 하나다" + (" (뺀 것: " + ", ".join(bits) + ")" if bits else " (" + r.type_terms + ")")
    return "같은 낱말을 다른 범주로 읽었다 (" + r.type_terms + ")"


def write_label_tables(m: pd.DataFrame, cm: pd.DataFrame) -> dict:
    """라벨이 바뀐 컷의 목록, 까닭별 수, 낱말별 쓰임을 표로 남긴다."""
    from wb_notes import load_term_map, map_row
    tdir = DATA_DIR / "tables"
    ch = cm[(cm.task_type_single != cm.task_type_single_tags) | (cm.task_type_multi != cm.task_type_multi_tags)].copy()
    ch["why_single"] = [why_changed(r) for r in ch.itertuples(index=False)]
    cols = ["cut_id", "species_ko", "group3", "split", "gold", "gold_order", "call_tags_raw", "task_type_single_tags", "task_type_single",
            "why_single", "task_type_multi_tags", "task_type_multi", "type_weak", "type_terms", "type_unplaced", "note_nonanimal",
            "note_other_species", "note_unknown", "note_flags", "note"]
    ch[cols].sort_values(["species_ko", "cut_id"]).to_csv(tdir / "type_label_changes.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    def kind(r) -> str:
        a, b = r.task_type_single_tags, r.task_type_single
        return "같음" if a == b else ("단일에서 빠짐" if a and not b else "단일에 새로 듦" if b and not a else "범주가 바뀜")

    s1 = cm.assign(kind=[kind(r) for r in cm.itertuples(index=False)])
    s1 = s1[s1.kind != "같음"].assign(why=[why_changed(r).split(" (")[0].split(":")[0] for r in s1[s1.kind != "같음"].itertuples(index=False)])
    summ = s1.groupby(["kind", "why"]).agg(cuts=("cut_id", "size"), dev=("split", lambda x: int((x == "dev").sum())),
                                           test=("split", lambda x: int((x == "test").sum())),
                                           gold=("gold", lambda x: int((x != "").sum())),
                                           species=("species_ko", "nunique")).reset_index().sort_values(["kind", "cuts"], ascending=[True, False])
    summ.to_csv(tdir / "type_label_change_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    cross = pd.crosstab(cm.task_type_single_tags.replace("", "(없음)").rename("v1_tags"), cm.task_type_single.replace("", "(없음)").rename("v1_1_note"))
    cross.to_csv(tdir / "type_single_v1_vs_v1_1.csv", encoding="utf-8-sig", lineterminator="\n")

    table = load_term_map()
    rows = {}
    for r in m.itertuples(index=False):
        for k in [k for k in r.type_terms.split(";") if k]:
            typ, basis = map_row(k, r.group3, table, r.species)
            d = rows.setdefault((k, r.group3, typ, basis), dict(n_all=0, n_core=0, n_gold=0))
            d["n_all"] += 1
            d["n_core"] += r.role == "core"
            d["n_gold"] += r.gold != ""
    use = pd.DataFrame([dict(term_key=k, group3=g, type=t, basis=b, **d) for (k, g, t, b), d in rows.items()])
    use.sort_values(["n_all"], ascending=False).to_csv(tdir / "note_term_usage.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    return dict(core_main=len(cm), single_tags_v1=int((cm.task_type_single_tags != "").sum()), single_note=int((cm.task_type_single != "").sum()),
                multi_tags_v1=int((cm.task_type_multi_tags != "").sum()), multi_note=int((cm.task_type_multi != "").sum()),
                single_changed=int((cm.task_type_single != cm.task_type_single_tags).sum()),
                single_dropped=int(((cm.task_type_single_tags != "") & (cm.task_type_single == "")).sum()),
                single_added=int(((cm.task_type_single_tags == "") & (cm.task_type_single != "")).sum()),
                single_reclassed=int(((cm.task_type_single_tags != "") & (cm.task_type_single != "") & (cm.task_type_single != cm.task_type_single_tags)).sum()),
                label_from_reading_single=int(((cm.task_type_single != "") & (cm.type_reading != "")).sum()))


def check_design(m: pd.DataFrame) -> list[str]:
    """v1 의 설계와 견준다. 다른 열의 이름을 돌려준다 (없으면 빈 목록)."""
    ref = pd.read_csv(DATA_DIR / "source" / "design_v1.csv.gz", dtype=str, keep_default_na=False)
    j = m[["cut_id", "role", "split", "fold", "fold_all", "fold_region", "task_group3", "task_species", "gold", "gold_order", "type_tags",
           "task_type_single_tags", "task_type_multi_tags"]].astype(str).merge(ref, on="cut_id", suffixes=("", "_ref"), validate="one_to_one")
    bad = [c for c in ["role", "split", "fold", "fold_all", "fold_region", "task_group3", "task_species", "gold", "gold_order"]
           if (j[c] != j[c + "_ref"]).any()]
    for now, then in [("type_tags", "type_tags_v1"), ("task_type_single_tags", "task_type_single_v1"), ("task_type_multi_tags", "task_type_multi_v1")]:
        if (j[now] != j[then]).any():
            bad.append(now)
    if len(j) != len(m) or len(j) != len(ref):
        bad.append("cut_id")
    return bad


def main() -> int:
    m = pd.read_csv(DATA_DIR / "manifest_all.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for c in ["fs_hz", "n_frames", "n_species_meta", "type_weak_n", "type_tags_n", "note_dup_n"]:
        m[c] = m[c].astype(int)
    m["dur_s"] = m.dur_s.astype(float)
    for c in [c for c in m.columns if c.startswith("f_")] + ["eligible"]:
        m[c] = m[c] == "True"

    m["role"] = select_core(m)
    core_main = m[(m.role == "core") & (m.pool == "main")]

    test_sites, search_info = choose_test_sites(core_main)
    m["split"] = np.where(m.g_site.isin(test_sites), "test", "dev")
    fold_of = assign_folds(m[(m.role == "core") & (m.split == "dev")])          # main 과 lf 를 함께 (해역 단위)
    m["fold"] = [fold_of.get(s, -1) if sp == "dev" else -1 for s, sp in zip(m.g_site, m.split)]
    # 전 해역 교차검증용 겹(참고값): 개발 겹 0..N_FOLDS-1 은 그대로, 시험 해역은 N_TEST_FOLDS 겹으로 나눠 뒤에 붙인다
    tfold = assign_folds(m[(m.role == "core") & (m.split == "test")], N_TEST_FOLDS, "tfold")
    m["fold_all"] = [fold_of.get(s, -1) if sp == "dev" else (N_FOLDS + tfold[s] if s in tfold else -1)
                     for s, sp in zip(m.g_site, m.split)]
    # 구역 단위 교차검증용 겹(가장 엄격한 참고값): 같은 넓은 구역은 모든 해를 통틀어 한 겹에만 들어간다
    m["region_key"] = [r if r else "site:" + s for r, s in zip(m.region, m.g_site)]
    rfold = assign_folds(m[m.role == "core"], N_REGION_FOLDS, "rfold", key="region_key")
    m["fold_region"] = m.region_key.map(lambda k: rfold.get(k, -1)).astype(int)
    # 시험 해역의 비핵심 컷은 학습에 쓰지 않는다
    m.loc[(m.split == "test") & (m.role.isin(["reserve", "held", "excluded"])), "role"] = \
        m.loc[(m.split == "test") & (m.role.isin(["reserve", "held", "excluded"])), "role"].map(lambda r: "test_" + r)

    core = m[m.role == "core"]
    cm = core[core.pool == "main"]

    # ---- 과제 열 ----------------------------------------------------------------------
    is_cm = (m.role == "core") & (m.pool == "main")
    m["task_group3"] = np.where(is_cm, m.group3, "")
    # 형태(단일): 노트가 대상 동물의 소리로 한 가지 형태만 적은 컷 (wb_notes.summarize 의 type_single)
    m["task_type_single"] = np.where(is_cm, m.type_single, "")
    # 형태(다중): 형태가 하나 이상 적힌 컷. 단, 형태를 모르는 소리가 따로 적혀 있거나(type_incomplete) 다른 종의 소리가 적힌 컷은 뺀다
    #            (8개 범주의 있음·없음을 다 알 수 없기 때문이다)
    multi_ok = (m.type_weak != "") & (m.type_incomplete == "") & (m.note_other_sound == "")
    m["task_type_multi"] = np.where(is_cm & multi_ok, m.type_weak, "")
    # v1 의 라벨(낱말 태그). 같은 분할에서 라벨만 바꿔 견줄 때 쓴다 (b07 --type-labels tags)
    m["task_type_single_tags"] = np.where(is_cm & (m.type_tags_n == 1), m.type_tags, "")
    m["task_type_multi_tags"] = np.where(is_cm & (m.type_tags_n >= 1), m.type_tags, "")
    sp_sites_dev = cm[cm.split == "dev"].groupby("species").g_site.nunique()
    sp_sites_test = cm[cm.split == "test"].groupby("species").g_site.nunique()
    sp_ok = sorted(sp for sp in cm.species.unique()
                   if sp_sites_dev.get(sp, 0) >= SPECIES_MIN_SITES - 1 and sp_sites_test.get(sp, 0) >= 1)
    m["task_species"] = np.where((m.role == "core") & (m.pool == "main") & m.species.isin(sp_ok), m.species, "")

    # ---- 골드 검토 목록 ----------------------------------------------------------------
    gold_test = gold_sample(cm[cm.split == "test"], GOLD_TEST, "gold_test")
    gold_prac = gold_sample(cm[cm.split == "dev"], GOLD_PRACTICE, "gold_practice")
    m["gold"] = ""
    m.loc[m.cut_id.isin(gold_prac), "gold"] = "practice"
    m.loc[m.cut_id.isin(gold_test), "gold"] = "test"
    order = {c: i + 1 for i, c in enumerate(gold_prac)}
    order.update({c: i + 1 for i, c in enumerate(gold_test)})
    m["gold_order"] = m.cut_id.map(lambda c: order.get(c, ""))

    # 이 세션에서 음원을 가진 컷 표시 (GPT 묶음의 1,005 + 31). 연구자 PC 에서는 뜻이 없다.
    if PILOT_AUDIO_LIST.exists():
        have = set(PILOT_AUDIO_LIST.read_text().split())
        m["audio_in_gpt_packages"] = m.cut_id.isin(have)

    # ---- 저장 ------------------------------------------------------------------------
    (DATA_DIR / "tables").mkdir(exist_ok=True)
    m.to_csv(DATA_DIR / "bench_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    core_cols = ["cut_id", "file_name", "species", "species_ko", "group3", "region_scope", "fs_hz", "fs_family", "dur_s",
                 "pool", "split", "fold", "fold_all", "fold_region", "region", "g_tape", "g_session", "g_campaign", "g_site", "type_weak", "type_weak_n",
                 "task_group3", "task_type_single", "task_type_multi", "task_species", "gold", "gold_order",
                 "location", "year", "note",
                 "type_single", "type_terms", "type_unplaced", "type_incomplete", "type_bg_only", "type_reading", "note_nonanimal",
                 "note_other_species", "note_unknown", "note_flags", "note_override", "note_dup_n",
                 "call_tags_raw", "type_tags", "type_tags_n", "task_type_single_tags", "task_type_multi_tags"]
    m[m.role == "core"][core_cols].to_csv(DATA_DIR / "bench_core.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    # 골드 검토 목록. 사람이 붙인 라벨은 하나도 없다. auto_ 로 시작하는 열은 노트에서 자동으로 읽은 것, v1_ 은 v1 의 낱말 태그다.
    gcols = {"gold": "gold", "gold_order": "gold_order", "cut_id": "cut_id", "file_name": "file_name", "species": "species",
             "species_ko": "species_ko", "group3_ko": "group3_ko", "fs_hz": "fs_hz", "dur_s": "dur_s", "location": "location",
             "year": "year", "note": "note", "type_weak": "auto_type_from_note", "type_single": "auto_single_type",
             "type_terms": "auto_terms", "type_unplaced": "auto_unplaced_terms", "note_nonanimal": "auto_nonanimal_sounds",
             "note_other_species": "auto_other_species", "note_unknown": "auto_unknown_origin", "note_flags": "auto_flags",
             "note_override": "auto_hand_fix_reason", "call_tags_raw": "v1_tags_raw", "type_tags": "v1_type_from_tags"}
    g = m[m.gold != ""][list(gcols)].rename(columns=gcols)
    g["gold_order"] = g.gold_order.astype(int)
    g = g.sort_values(["gold", "gold_order"])
    g.insert(2, "label_status", "unreviewed")                  # 이 표는 검토할 컷의 목록이다. 사람이 붙인 라벨이 아니다
    g.to_csv(DATA_DIR / "gold_review_list.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    # ---- 요약표 -----------------------------------------------------------------------
    def effn(s: pd.Series) -> float:
        p = s.value_counts(normalize=True).to_numpy()
        return round(float(1 / (p ** 2).sum()), 2)

    el = m[m.eligible]
    rows = []
    for sp, a in m.groupby("species"):
        c = a[a.role == "core"]
        e = el[el.species == sp]
        rows.append(dict(
            species=sp, species_ko=a.species_ko.iloc[0], group3=a.group3.iloc[0], region_scope=a.region_scope.iloc[0],
            n_all=len(a), n_held=int((a.hold_reason != "").sum()), n_eligible=len(e),
            pool="/".join(f"{k}:{v}" for k, v in a.pool.value_counts().items()),
            tapes=e.g_tape.nunique(), sessions=e.g_session.nunique(), campaigns=e.g_campaign.nunique(), sites=e.g_site.nunique(),
            eff_campaigns=effn(e.g_campaign) if len(e) else 0,
            n_core=len(c), n_core_main=int((c.pool == "main").sum()), n_core_lf=int((c.pool == "lf").sum()),
            core_dev=int((c.split == "dev").sum()), core_test=int((c.split == "test").sum()),
            sites_dev=c[c.split == "dev"].g_site.nunique(), sites_test=c[c.split == "test"].g_site.nunique(),
            in_species_task=sp in sp_ok,
            fs_families=" ".join(f"{k}:{v}" for k, v in c.fs_family.value_counts().items()),
        ))
    sp_tab = pd.DataFrame(rows).sort_values(["group3", "n_core"], ascending=[True, False])
    sp_tab.to_csv(DATA_DIR / "tables" / "species_table.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    grp_tab = cm.groupby(["group3", "split"]).agg(cuts=("cut_id", "size"), species=("species", "nunique"),
                                                 campaigns=("g_campaign", "nunique"), sites=("g_site", "nunique")).reset_index()
    grp_tab.to_csv(DATA_DIR / "tables" / "group_table.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    cm = m[(m.role == "core") & (m.pool == "main")]             # 과제 열이 붙은 뒤의 표로 다시 잡는다

    def type_table(col_multi: str, col_single: str) -> pd.DataFrame:
        trows = []
        for t in TYPE_CLASSES:
            has = cm[cm[col_multi].map(lambda s: t in s.split(";"))]
            single = cm[cm[col_single] == t]
            for part, sub_any, sub_single in [("dev", has[has.split == "dev"], single[single.split == "dev"]),
                                              ("test", has[has.split == "test"], single[single.split == "test"])]:
                trows.append(dict(type=t, split=part, cuts_any=len(sub_any), cuts_single=len(sub_single),
                                  sites_any=sub_any.g_site.nunique(), sites_single=sub_single.g_site.nunique(),
                                  species_single=sub_single.species.nunique(),
                                  groups_single=" ".join(f"{k}:{v}" for k, v in sub_single.group3.value_counts().items())))
        return pd.DataFrame(trows)

    type_table("task_type_multi", "task_type_single").to_csv(
        DATA_DIR / "tables" / "type_table.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    type_table("task_type_multi_tags", "task_type_single_tags").to_csv(
        DATA_DIR / "tables" / "type_table_v1_tags.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    change_summary = write_label_tables(m, cm)

    srows = []
    for s, a in m.groupby("g_site"):
        c = a[a.role == "core"]
        srows.append(dict(
            g_site=s, split=a.split.iloc[0], fold=int(a.fold.iloc[0]), n_all=len(a), n_core=len(c),
            campaigns=a.g_campaign.nunique(), tapes=a.g_tape.nunique(),
            years=",".join(sorted(set(y for y in a.year if y))),
            locations=" | ".join(k for k, _ in Counter(l for l in a.location if l).most_common(4)),
            species_core=", ".join(f"{k}:{v}" for k, v in c.species.value_counts().items()),
            groups_core=" ".join(f"{k}:{v}" for k, v in c.group3.value_counts().items()),
        ))
    pd.DataFrame(srows).sort_values(["split", "n_core"], ascending=[True, False]).to_csv(
        DATA_DIR / "tables" / "site_table.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    fam_tab = pd.crosstab([cm.fs_family, cm.split], cm.group3)
    fam_tab.to_csv(DATA_DIR / "tables" / "fs_family_table.csv", encoding="utf-8-sig", lineterminator="\n")

    # ---- 점검 -------------------------------------------------------------------------
    leak = {lvl: len(set(m[m.split == "dev"][lvl]) & set(m[m.split == "test"][lvl]))
            for lvl in ["g_site", "g_campaign", "g_session", "g_tape"]}
    md5_dev = set(m[m.split == "dev"].md5_b64)
    md5_test = set(m[m.split == "test"].md5_b64)
    fold_leak = 0
    dev = m[(m.split == "dev") & (m.fold >= 0)]
    for lvl in ["g_site"]:
        fold_leak += int((dev.groupby(lvl).fold.nunique() > 1).sum())
    design_diff = check_design(m)
    summ = dict(
        version=VERSION, seed=SEED, cap=CAP, test_frac_target=TEST_FRAC, n_folds=N_FOLDS,
        min_dur=MIN_DUR, max_dur=MAX_DUR,
        role=m.role.value_counts().to_dict(),
        core_main=dict(n=len(cm), dev=int((cm.split == "dev").sum()), test=int((cm.split == "test").sum()),
                       species=cm.species.nunique(), sites=cm.g_site.nunique(), campaigns=cm.g_campaign.nunique(),
                       tapes=cm.g_tape.nunique(), hours=round(cm.dur_s.sum() / 3600, 2)),
        core_lf=dict(n=int(((m.role == "core") & (m.pool == "lf")).sum()),
                     species=m[(m.role == "core") & (m.pool == "lf")].species.value_counts().to_dict()),
        test_search=search_info,
        group3_core={g: dict(dev=int(((cm.group3 == g) & (cm.split == "dev")).sum()),
                             test=int(((cm.group3 == g) & (cm.split == "test")).sum()),
                             sites_dev=cm[(cm.group3 == g) & (cm.split == "dev")].g_site.nunique(),
                             sites_test=cm[(cm.group3 == g) & (cm.split == "test")].g_site.nunique()) for g in GROUPS3},
        species_task=dict(n_species=len(sp_ok), species=sp_ok),
        region_fold_sizes_main={int(k): int(v) for k, v in cm.fold_region.value_counts().sort_index().items()},
        region_folds={int(k): sorted(set(v)) for k, v in cm.groupby("fold_region").region_key},
        dev_fold_sizes_main={int(k): int(v) for k, v in cm[cm.split == "dev"].fold.value_counts().sort_index().items()},
        dev_fold_sites_main={int(k): int(v) for k, v in cm[cm.split == "dev"].groupby("fold").g_site.nunique().items()},
        gold=dict(test=len(gold_test), practice=len(gold_prac),
                  test_by_group=m[m.gold == "test"].group3.value_counts().to_dict()),
        type_labels=change_summary,
        type_single_core={t: dict(dev=int(((cm.task_type_single == t) & (cm.split == "dev")).sum()),
                                  test=int(((cm.task_type_single == t) & (cm.split == "test")).sum()),
                                  sites=int(cm[cm.task_type_single == t].g_site.nunique())) for t in TYPE_CLASSES},
        checks=dict(shared_between_dev_and_test=leak, shared_md5=len(md5_dev & md5_test),
                    dev_sites_in_more_than_one_fold=fold_leak, design_columns_different_from_v1=design_diff),
        sha256=dict(bench_manifest=sha256_file(DATA_DIR / "bench_manifest.csv"),
                    bench_core=sha256_file(DATA_DIR / "bench_core.csv"),
                    gold_review_list=sha256_file(DATA_DIR / "gold_review_list.csv")),
    )
    write_json(DATA_DIR / "bench_summary.json", summ)
    log(json.dumps(summ, ensure_ascii=False, indent=1))
    if any(leak.values()) or summ["checks"]["shared_md5"] or fold_leak:
        log("!! 분할 누출이 있다. 위 checks 를 확인할 것.")
        return 1
    if design_diff:
        log(f"!! 표본 설계가 v1 과 다르다: {design_diff}. 라벨 규칙을 고쳐도 설계는 바뀌지 않아야 한다 (설계는 type_tags 로만 한다).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
