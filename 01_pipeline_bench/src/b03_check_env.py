# -*- coding: utf-8 -*-
"""3단계. 환경과 음원 폴더를 점검한다. 읽기만 한다 (원본 폴더에는 아무것도 쓰지 않는다).

하는 일
  1) 파이썬과 패키지 판을 적는다.
  2) 음원 폴더에서 컷 번호가 붙은 파일을 찾아 목록표와 맞춘다 (전체 13,693개, 핵심 표본, 골드 목록).
  3) --verify N 을 주면 핵심 표본에서 N개를 골라 헤더(fs, 프레임 수)가 목록표와 같은지 본다.

출력  work/env_check.json, work/audio_index.csv
쓰는 법  python src/b03_check_env.py [--verify 200]
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys

import pandas as pd

from wb_common import DATA_DIR, VERSION, index_audio, load_config, log, stable_key, work_dir, write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", type=int, default=200, help="헤더를 대조할 핵심 표본 수 (0 이면 건너뜀)")
    args = ap.parse_args()
    cfg = load_config()
    out = {"version": VERSION, "python": sys.version.split()[0], "platform": platform.platform(),
           "audio_root": cfg["audio_root"], "work_root": cfg["work_root"], "packages": {}, "problems": []}

    for name, need in [("numpy", True), ("scipy", True), ("pandas", True), ("sklearn", True), ("soundfile", True),
                       ("matplotlib", False), ("torch", False), ("torchaudio", False), ("transformers", False),
                       ("tensorflow", False), ("umap", False)]:
        try:
            mod = importlib.import_module(name)
            out["packages"][name] = getattr(mod, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            out["packages"][name] = f"없음 ({type(e).__name__})"
            if need:
                out["problems"].append(f"필수 패키지 {name} 이(가) 없다: pip install {'scikit-learn' if name == 'sklearn' else name}")

    m = pd.read_csv(DATA_DIR / "bench_manifest.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    idx = index_audio(cfg["audio_root"])
    out["audio_files_found"] = len(idx)
    if not idx:
        out["problems"].append(
            "음원을 찾지 못했다. watkins_bench_config.json 의 audio_root 를 고치거나 환경 변수 WATKINS_AUDIO_ROOT 를 지정한다.")
    have = m.cut_id.isin(idx)
    out["manifest_cuts"] = int(len(m))
    out["manifest_cuts_found"] = int(have.sum())
    out["files_not_in_manifest"] = int(len(set(idx) - set(m.cut_id)))
    for label, mask in [("core", m.role == "core"), ("core_main_dev", (m.role == "core") & (m.pool == "main") & (m.split == "dev")),
                        ("core_main_test", (m.role == "core") & (m.pool == "main") & (m.split == "test")),
                        ("core_lf", (m.role == "core") & (m.pool == "lf")),
                        ("gold_test", m.gold == "test"), ("gold_practice", m.gold == "practice")]:
        out[f"{label}_found"] = f"{int((mask & have).sum())} / {int(mask.sum())}"
    pd.DataFrame({"cut_id": list(idx), "path": list(idx.values())}).sort_values("cut_id").to_csv(
        work_dir() / "audio_index.csv", index=False, encoding="utf-8-sig")

    if args.verify and idx and "없음" not in out["packages"].get("soundfile", "없음"):
        import soundfile as sf
        cand = m[(m.role == "core") & have].copy()
        cand["k"] = cand.cut_id.map(lambda c: stable_key("verify", c))
        cand = cand.sort_values("k").head(args.verify)
        bad = []
        for r in cand.itertuples(index=False):
            info = sf.info(idx[r.cut_id])
            if int(info.samplerate) != int(r.fs_hz) or int(info.frames) != int(r.n_frames):
                bad.append(dict(cut_id=r.cut_id, file_fs=int(info.samplerate), manifest_fs=int(r.fs_hz),
                                file_frames=int(info.frames), manifest_frames=int(r.n_frames)))
        out["header_checked"] = int(len(cand))
        out["header_mismatch"] = len(bad)
        out["header_mismatch_examples"] = bad[:10]
        if bad:
            out["problems"].append(f"헤더가 목록표와 다른 파일 {len(bad)}개. 음원이 공개 원본(FLAC)과 다른 판일 수 있다.")

    write_json(work_dir() / "env_check.json", out)
    for k, v in out.items():
        log(f"{k}: {v}")
    log("\n결과:", "문제 없음" if not out["problems"] else "확인 필요 → " + " / ".join(out["problems"]))
    return 0 if not out["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
