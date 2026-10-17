# -*- coding: utf-8 -*-
"""1단계. 전체 13,693개 컷의 목록표(매니페스트)를 만든다. 음원은 읽지 않는다.

입력  data/source/catalog_rich_13693.csv.gz      노트·날짜·장소·좌표·종 목록 (Watkins 메타데이터 덤프에서 온 것)
      data/source/inventory_native_13693.csv.gz   공개 원본의 MD5, 프레임 수, 채널 수, 분류 체계
      data/note_term_map.csv                      노트의 소리 낱말 → 형태 범주 대응표 (연구자가 고칠 수 있다)
      data/note_overrides.csv                     규칙으로 못 고치는 컷을 손으로 고친 표 (있으면 쓴다)
출력  data/manifest_all.csv                       컷마다 한 행: 분류군, 표시, 출처 묶음, 풀

만드는 것
  1) 분류군(3개)과 지역 범위 표시
  2) 노트에서 뽑은 표시: 사육 시설, 공중 녹음, 다종 표기, 명시적 불량, 잡음, 속도, 재디지털화, 사본
  3) 약한 호출 형태 라벨 두 가지
       type_tags   공개본의 낱말 태그(call_type)를 8개 묶음으로 옮긴 것. v1 의 라벨이다. 표본 설계(핵심 표본·분할·겹)는 계속 이것으로 한다.
       type_weak   노트를 구절 단위로 읽어 만든 것 (wb_notes.py). v1.1 부터 과제의 라벨은 이것이다.
                   "ship creaks", "Sp. wh. clicks", "Calls- 1 upsweep" 같은 문구에서 생긴 태그를 걸러 낸다.
  4) 출처 묶음 네 층: 테이프 → 세션(같은 날짜·장소) → 현장(같은 장소·연도) → 해역(같은 장소, 그리고 같은 구역·같은 해의 원정)
  5) 풀: main / lf / excluded_user / other  +  보류 사유

출처 묶음은 메타데이터로 만든 대리값이다. 검증된 녹음 세션이나 개체 ID 가 아니다.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import pandas as pd

from wb_common import (DATA_DIR, GROUP_KO, LF_POOL_MAX_FS, RE_AIR, RE_BAD, RE_CAPTIVE, RE_COPY, RE_GOOD, RE_NOISE,
                       RE_RESTREAM, RE_SPEED, REGION_SCOPE, SOURCE_DIR, SPECIES, TYPE_CLASSES, USER_EXCLUDED_SPECIES, UnionFind,
                       fs_family, log, map_type_tags, norm_location, region_of, sha256_file, write_json)
from wb_notes import NOTE_COLUMNS, default_map_path, load_term_map, parse_note, summarize

OVERRIDES = DATA_DIR / "note_overrides.csv"


def note_labels(out: pd.DataFrame) -> pd.DataFrame:
    """노트를 읽어 형태 라벨과 표시를 만든다. 같은 (노트, 종)은 한 번만 푼다."""
    table = load_term_map()
    cache, rows = {}, []
    for note, sp, grp in zip(out.note, out.species, out.group3):
        k = (note, sp)
        if k not in cache:
            cache[k] = summarize(parse_note(note, sp, grp), sp, grp, table, TYPE_CLASSES)
        rows.append(cache[k])
    lab = pd.DataFrame(rows, index=out.index)[NOTE_COLUMNS]
    lab["note_override"] = ""
    if OVERRIDES.exists():
        ov = pd.read_csv(OVERRIDES, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        ov = ov[~ov.cut_id.str.startswith("#") & (ov.cut_id != "")]
        unknown = sorted(set(ov.cut_id) - set(out.cut_id))
        if unknown or ov.cut_id.duplicated().any():
            raise SystemExit(f"note_overrides.csv: 모르는 컷 {unknown} 또는 겹친 컷이 있다")
        pos = {c: i for i, c in zip(out.index, out.cut_id)}
        for r in ov.itertuples(index=False):
            types = [t for t in r.type_weak.split(";") if t]
            bad = [t for t in types + ([r.type_single] if r.type_single else []) if t not in TYPE_CLASSES]
            if bad or (r.type_single and types != [r.type_single]):
                raise SystemExit(f"note_overrides.csv: {r.cut_id} 의 값이 맞지 않다 (type_weak={r.type_weak!r}, type_single={r.type_single!r})")
            i = pos[r.cut_id]
            lab.loc[i, ["type_weak", "type_weak_n", "type_single", "note_override"]] = [";".join(types), len(types), r.type_single, r.reason]
            if not types:
                lab.loc[i, ["type_stated", "type_implied"]] = ["", ""]
    return lab


def first_coord(s: str):
    try:
        c = json.loads(s or "[]")
        if c and c[0].get("lat") is not None and c[0].get("lon") is not None:
            return float(c[0]["lat"]), float(c[0]["lon"])
    except (ValueError, TypeError, KeyError):
        pass
    return None


def facility_key(loc: str, note: str) -> str:
    """사육 시설·개체를 가리키는 열쇠. 같은 시설의 같은 종은 해가 달라도 한 묶음으로 본다."""
    t = (loc + " " + note).lower()
    for key, pat in [("oregon_coast_aquarium", "oregon coast"), ("oregon_coast_aquarium", " oca "),
                     ("oregon_coast_aquarium", "keiko"), ("moby_doll_vancouver", "moby doll"),
                     ("ny_aquarium", "new york aquarium"), ("niagara_aquarium", "niagra falls aquarium"),
                     ("marineland_pacific", "marine land of the pacific"), ("marineland_pacific", "marineland of the pacific"),
                     ("marineland_la", "marineland, los angeles"), ("bronx_zoo", "bronx zoo"),
                     ("johns_hopkins_tank", "johns hopkins"), ("silver_springs", "silver springs")]:
        if pat in t:
            return key
    return ""


def build() -> pd.DataFrame:
    cat = pd.read_csv(SOURCE_DIR / "catalog_rich_13693.csv.gz", dtype=str, keep_default_na=False)
    inv = pd.read_csv(SOURCE_DIR / "inventory_native_13693.csv.gz", dtype=str, keep_default_na=False)
    assert cat.cut_id.is_unique and inv.cut_id.is_unique and set(cat.cut_id) == set(inv.cut_id)
    df = cat.merge(inv, on="cut_id", how="left", validate="one_to_one").sort_values("cut_id").reset_index(drop=True)

    unknown = sorted(set(df.species) - set(SPECIES))
    if unknown:
        raise SystemExit(f"종 표에 없는 이름: {unknown}")

    out = pd.DataFrame({"cut_id": df.cut_id, "tape_id": df.tape_id, "file_name": df.local_relative_path})
    out["species"] = df.species
    out["species_ko"] = df.species.map(lambda s: SPECIES[s][1])
    out["species_en"] = df.species.map(lambda s: SPECIES[s][2])
    out["group3"] = df.species.map(lambda s: SPECIES[s][0])
    out["group3_ko"] = out.group3.map(GROUP_KO)
    out["family"] = df.family
    out["region_scope"] = df.species.map(lambda s: REGION_SCOPE.get(s, "other"))

    out["fs_hz"] = df.sample_rate_hz.astype(float).astype(int)
    out["fs_family"] = out.fs_hz.map(fs_family)
    out["dur_s"] = df.duration_s.astype(float).round(4)
    out["n_frames"] = df.native_frames.astype(int)
    out["md5_b64"] = df.source_md5_b64

    out["date"] = df.observation_date.str[:10]
    out["year"] = df.observation_date.str[:4]
    out["location"] = df.location_text
    coords = df.coordinates_json.map(first_coord)
    out["lat"] = coords.map(lambda c: c[0] if c else "")
    out["lon"] = coords.map(lambda c: c[1] if c else "")
    out["note"] = df.metadata_note
    out["cue_s"] = df.source_cue_s
    out["meta_cut_size_s"] = df.source_cut_size_s
    out["meta_fs_hz"] = df.rich_sample_rate_hz
    out["meta_channels"] = df.recorded_channels

    # ---- 노트·장소에서 뽑은 표시 -------------------------------------------------------
    sp_lists = df.metadata_species_json.map(lambda s: json.loads(s or "[]"))
    out["n_species_meta"] = sp_lists.map(len)
    out["other_species_meta"] = [";".join(x for x in lst if x != sp) for lst, sp in zip(sp_lists, df.species)]
    both = df.location_text + " | " + df.metadata_note
    out["f_captive_note"] = both.map(lambda t: bool(RE_CAPTIVE.search(t)))
    out["facility"] = [facility_key(a, b) for a, b in zip(df.location_text, df.metadata_note)]
    out["f_air"] = df.metadata_note.map(lambda t: bool(RE_AIR.search(t)))
    out["f_multi_species"] = out.n_species_meta > 1
    out["f_bad_note"] = df.metadata_note.map(lambda t: bool(RE_BAD.search(t)))
    out["f_noise_note"] = df.metadata_note.map(lambda t: bool(RE_NOISE.search(t)))
    out["f_good_note"] = df.metadata_note.map(lambda t: bool(RE_GOOD.search(t)))
    out["f_speed_note"] = df.metadata_note.map(lambda t: bool(RE_SPEED.search(t)))
    out["f_restream_note"] = df.metadata_note.map(lambda t: bool(RE_RESTREAM.search(t)))
    out["f_copy_note"] = df.metadata_note.map(lambda t: bool(RE_COPY.search(t)))
    # 헤더 길이가 메타데이터의 컷 길이(CS)와 2 % 넘게 다르면 표시 (시간축·잘림 점검 대상)
    cs = pd.to_numeric(df.source_cut_size_s, errors="coerce")
    ratio = out.dur_s / cs
    out["dur_over_meta"] = ratio.round(4)
    out["f_dur_mismatch"] = (ratio.notna()) & ((ratio < 0.98) | (ratio > 1.02))
    mfs = pd.to_numeric(df.rich_sample_rate_hz, errors="coerce")
    out["f_fs_mismatch"] = mfs.notna() & ((mfs - out.fs_hz).abs() / out.fs_hz > 0.02)

    # ---- 약한 호출 형태 라벨 ------------------------------------------------------------
    out["call_tags_raw"] = df.call_type
    tags = df.call_type.map(map_type_tags)
    out["type_tags"] = tags.map(";".join)                    # v1 의 라벨 (낱말 태그). 표본 설계는 이것으로 한다
    out["type_tags_n"] = tags.map(len)
    lab = note_labels(out)                                   # v1.1 의 라벨 (노트를 읽은 것)
    for c in lab.columns:
        out[c] = lab[c]
    out["type_weak_n"] = out.type_weak_n.astype(int)
    out["note_dup_n"] = out.groupby(["tape_id", "note"]).cut_id.transform("size")   # 같은 테이프에서 같은 노트를 쓰는 컷 수

    # ---- 출처 묶음 -------------------------------------------------------------------
    out["loc_key"] = df.location_text.map(norm_location)
    out["cell"] = [f"{round(c[0])},{round(c[1])}" if c else "" for c in coords]

    def components(edge_keys: list[str]) -> pd.Series:
        uf = UnionFind()
        for cid in out.cut_id:
            uf.find(("c", cid))
        for col in edge_keys:
            groups = defaultdict(list)
            for cid, key in zip(out.cut_id, out[col]):
                if key:
                    groups[key].append(cid)
            for members in groups.values():
                for m in members[1:]:
                    uf.union(("c", members[0]), ("c", m))
        return out.cut_id.map(lambda c: uf.find(("c", c))[1])

    out["k_tape"] = "T" + out.tape_id
    md5_n = Counter(out.md5_b64)
    out["k_md5"] = out.md5_b64.map(lambda m: "M" + m if md5_n[m] > 1 else "")
    out["k_date_loc"] = [f"D{d}|{l}" if d and l else "" for d, l in zip(out.date, out.loc_key)]
    out["k_loc_year"] = [f"L{l}|{y}" if l else "" for l, y in zip(out.loc_key, out.year)]
    out["k_cell_year"] = [f"C{c}|{y}" if c else "" for c, y in zip(out.cell, out.year)]
    out["k_facility"] = [f"F{f}|{s}" if f else "" for f, s in zip(out.facility, out.species)]
    out["k_loc"] = out.loc_key.map(lambda l: "L" + l if l else "")
    out["k_cell"] = out.cell.map(lambda c: "C" + c if c else "")

    # 같은 구역·같은 해의 녹음은 같은 원정(항해·현장 조사)으로 본다.
    # 예: 1966년 남극 로스해의 Cape Adare, Cape Hallett, Cape Jones 는 테이프 번호가 이어지는 한 항해다.
    out["region"] = [region_of(l, c) for l, c in zip(df.location_text, out.cell)]
    out["k_exped"] = [f"X{r}|{y}" if r and y else "" for r, y in zip(out.region, out.year)]

    root_session = components(["k_tape", "k_md5", "k_date_loc"])
    root_campaign = components(["k_tape", "k_md5", "k_date_loc", "k_loc_year", "k_cell_year", "k_facility"])
    root_site = components(["k_tape", "k_md5", "k_date_loc", "k_loc_year", "k_cell_year", "k_facility", "k_loc", "k_cell",
                            "k_exped"])

    def relabel(roots: pd.Series, prefix: str) -> pd.Series:
        order = {r: i for i, r in enumerate(sorted(set(roots)))}
        return roots.map(lambda r: f"{prefix}{order[r]:03d}")

    out["g_tape"] = out.tape_id
    out["g_session"] = relabel(root_session, "S")
    out["g_campaign"] = relabel(root_campaign, "K")
    out["g_site"] = relabel(root_site, "A")
    out = out.drop(columns=[c for c in out.columns if c.startswith("k_")])

    # 사육 표시의 전파: 한 현장 묶음(g_campaign)의 절반 이상이 사육 시설 노트면 묶음 전체를 사육으로 본다
    # (예: 1964년 밴쿠버의 'Moby Doll' 테이프 가운데 노트에 이름이 빠진 테이프).
    frac = out.groupby(["g_campaign", "species"]).f_captive_note.transform("mean")
    out["f_captive"] = out.f_captive_note | (frac >= 0.5)
    out["f_captive_inferred"] = out.f_captive & ~out.f_captive_note

    # ---- 풀 --------------------------------------------------------------------------
    def pool(r) -> str:
        if r.species in USER_EXCLUDED_SPECIES:
            return "excluded_user"
        if r.group3 == "Other":
            return "other"
        return "lf" if r.fs_hz < LF_POOL_MAX_FS else "main"

    out["pool"] = out.apply(pool, axis=1)

    def hold(r) -> str:
        reasons = []
        if r.f_captive:
            reasons.append("captive")
        if r.f_air:
            reasons.append("air")
        if r.f_multi_species:
            reasons.append("multi_species")
        if r.f_bad_note:
            reasons.append("bad_note")
        return ";".join(reasons)

    out["hold_reason"] = out.apply(hold, axis=1)
    out["eligible"] = (out.pool.isin(["main", "lf"])) & (out.hold_reason == "")
    return out


def main() -> int:
    out = build()
    path = DATA_DIR / "manifest_all.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    summ = {
        "n_cuts": int(len(out)),
        "n_species": int(out.species.nunique()),
        "n_tapes": int(out.g_tape.nunique()),
        "n_sessions": int(out.g_session.nunique()),
        "n_campaigns": int(out.g_campaign.nunique()),
        "n_sites": int(out.g_site.nunique()),
        "pool": out.pool.value_counts().to_dict(),
        "eligible": int(out.eligible.sum()),
        "hold": {k: int(v) for k, v in Counter(r for s in out.hold_reason for r in s.split(";") if r).items()},
        "captive_inferred": int(out.f_captive_inferred.sum()),
        "group3_all": out.group3.value_counts().to_dict(),
        "group3_eligible_main": out[out.eligible & (out.pool == "main")].group3.value_counts().to_dict(),
        "type_labels": {
            "tags_v1_any": int((out.type_tags != "").sum()), "tags_v1_single": int((out.type_tags_n == 1).sum()),
            "note_any": int((out.type_weak != "").sum()), "note_single": int((out.type_single != "").sum()),
            "note_incomplete": int((out.type_incomplete != "").sum()), "note_other_species_sound": int((out.note_other_sound != "").sum()),
            "note_nonanimal_term": int((out.note_nonanimal != "").sum()), "note_overrides": int((out.note_override != "").sum()),
            "distinct_notes": int(out.note.nunique()),
        },
        "source_sha256": {p.name: sha256_file(p) for p in sorted(SOURCE_DIR.glob("*.gz"))},
        "label_inputs_sha256": {p.name: sha256_file(p) for p in [default_map_path(), OVERRIDES] if p.exists()},
        "manifest_sha256": sha256_file(path),
    }
    write_json(DATA_DIR / "manifest_all_summary.json", summ)
    log(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
