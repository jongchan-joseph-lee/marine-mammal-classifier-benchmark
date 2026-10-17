# -*- coding: utf-8 -*-
"""6단계. 사전학습 모델의 동결 임베딩을 뽑는다 (레퍼런스 방법의 표현 부분).

2026-10-06: 실제 AST·VGGish 사전학습 가중치로 자체 시험과 실제 Watkins 추론을 확인했다.
전체 main 핵심 표본의 배치 계산은 b14_compute_actual_references.py, 완료한 수치와 기록은
work/actual_reference_run 및 work/eval/actual_ast_vggish_primary 에 있다.
새 환경에서는 --selftest 로 모양과 유한값을 확인한다. 임의 가중치나 가짜 모델로 대체하지 않는다.

지원하는 모델
  ast      Audio Spectrogram Transformer (AudioSet 사전학습, 'MIT/ast-finetuned-audioset-10-10-0.4593').
           입력 16 kHz, 멜 128, 10.24초(1024 프레임). 임베딩 768차원 (pooler_output).
           --pad zero    : 길면 앞 10.24초만 쓴다. 짧으면 모델의 특징 추출기가 **멜 스펙트로그램**의 뒤를 0 으로 채운다
                           (AST 의 기본 동작). 파일 길이가 임베딩에 새겨진다 (380 표본에서 길이 복원 R² 0.89).
           --pad wavzero : 짧으면 **파형**의 뒤를 0(무음)으로 채워 10.24초로 만든 뒤 넣는다.
                           Lipari 외(2026)는 "10초로 채우거나 자른다"고만 적었다(요약 도구로 읽음). 멜에서 채웠는지
                           파형에서 채웠는지 확인하지 못해 두 가지를 다 둔다. 두 조건은 채운 부분의 값이 다르다.
           --pad tile    : 짧으면 되풀이해 채우고, 길면 10.24초 창 여러 개로 고르게 덮어 평균한다. 우리 쪽에서 바꾼 조건.
  vggish   VGGish (AudioSet 사전학습). 입력 16 kHz, 0.96초 단위, 임베딩 128차원을 단위 평균.
           1초보다 짧은 컷은 되풀이해 채운다 (그러지 않으면 임베딩이 나오지 않는다).
           Cominelli 외(2024)의 표현 부분에 해당한다. 원문의 세부 조건(후처리 여부 등)은 원문과 대조하지 못했다.
           TorchAudio 2.8의 공개 구현을 로컬 vggish_reference.py에 보존하고 공개 사전학습 상태 사전을 엄격히 로드한다.
           TorchAudio prototype의 누락된 import와 2.9 이후 삭제에 의존하지 않는다.
           PCA·8비트 양자화 후처리를 적용하지 않은 부동소수점 임베딩이다.

공통 대역 조건
  --band4k  를 주면 모든 컷을 먼저 8 kHz 로 내려(0-4 kHz 만 남김) 그 뒤 모델의 입력 fs 로 올린다.
            fs 가 달라서 생기는 '빈 윗대역'을 없앤 조건이다. fs 8 kHz 미만 컷은 뺀다. 이름 뒤에 _band4k 가 붙는다.

다른 모델(Perch 2.0, BirdNET, AVES 등)이나 이미 가진 임베딩은 같은 형식의 npz 로 저장하면 b07 이 그대로 읽는다:
  numpy.savez_compressed("work/features/<이름>.npz", cut_id=<문자열 배열>, X=<float32 (n, d)>)

쓰는 법  python src/b06_embed_pretrained.py --model ast --pad zero [--set core] [--band4k] [--selftest] [--limit 50]
필요    pip install torch torchaudio transformers   (모델 가중치를 처음 한 번 인터넷에서 받는다)
계산    다른 단계와 달리 스레드를 여러 개 쓴다(기본 최대 8개. 환경 변수 WATKINS_TORCH_THREADS 로 바꾼다).
        스레드 수에 따라 임베딩의 끝자리가 달라질 수 있다.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from b05_features_baseline import resample_to
from wb_common import DATA_DIR, MAX_ANALYSIS_S, index_audio, load_config, log, read_audio, work_dir

AST_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SECONDS = 10.24
VGGISH_MIN_SECONDS = 1.0     # 0.96초 단위 하나가 나오는 최소 길이(0.975초)에 여유를 둔 값


def set_torch_threads(torch) -> int:
    """wb_common 이 다른 단계의 재현을 위해 스레드를 1개로 묶어 두었다. 모델 계산은 느리므로 여기서만 푼다."""
    n = int(os.environ.get("WATKINS_TORCH_THREADS", "0") or 0) or min(8, os.cpu_count() or 1)
    try:
        torch.set_num_threads(n)
    except Exception as e:  # noqa: BLE001
        log(f"스레드 수를 바꾸지 못했다({type(e).__name__}). 그대로 진행한다")
    return n


def tile_to(x: np.ndarray, n: int) -> np.ndarray:
    """되풀이해 길이 n 으로 만든다. 이미 n 이상이면 그대로 둔다."""
    if len(x) >= n:
        return x
    reps = int(np.ceil(n / max(len(x), 1)))
    return np.tile(x, reps)[:n]


def window_starts(n: int, n_win: int) -> list[int]:
    """길이 n 을 길이 n_win 의 창으로 고르게 덮는 시작점. 창은 모두 꽉 찬 길이이고, 마지막 창은 끝에 맞춘다(조금 겹칠 수 있다)."""
    if n <= n_win:
        return [0]
    k = int(np.ceil(n / n_win))
    return [int(round(v)) for v in np.linspace(0, n - n_win, k)]


class AstEmbedder:
    dim = 768

    def __init__(self, pad: str, revision: str | None = None):
        import torch
        from transformers import ASTFeatureExtractor, ASTModel
        self.torch = torch
        self.threads = set_torch_threads(torch)
        self.fe = ASTFeatureExtractor.from_pretrained(AST_NAME, revision=revision)
        self.model, self.loading_info = ASTModel.from_pretrained(
            AST_NAME, revision=revision, output_loading_info=True, attn_implementation="sdpa")
        if self.loading_info.get("missing_keys") or self.loading_info.get("mismatched_keys"):
            raise RuntimeError(f"사전학습 AST 가중치가 완전히 로드되지 않았습니다: {self.loading_info}")
        self.model.eval()
        self.pad = pad

    def chunks(self, x16: np.ndarray) -> list[np.ndarray]:
        n_win = int(AST_SECONDS * 16000)
        if self.pad == "zero":
            return [x16[:n_win]]                           # 짧으면 추출기가 멜 스펙트로그램의 뒤를 0 으로 채운다
        if self.pad == "wavzero":
            c = x16[:n_win]
            return [np.concatenate([c, np.zeros(n_win - len(c), dtype=c.dtype)])]   # 파형의 뒤를 무음으로 채운다
        if len(x16) <= n_win:
            return [tile_to(x16, n_win)]
        return [x16[s:s + n_win] for s in window_starts(len(x16), n_win)]

    def __call__(self, x16: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            inputs = self.fe([c.astype(np.float32) for c in self.chunks(x16)], sampling_rate=16000, return_tensors="pt")
            out = self.model(**inputs)
        return out.pooler_output.mean(0).cpu().numpy()


class VggishEmbedder:
    dim = 128

    def __init__(self):
        import torch
        self.torch = torch
        self.threads = set_torch_threads(torch)
        # TorchAudio 2.8's prototype loader has a missing import and is removed in
        # 2.9. Keep its Apache-2.0 implementation locally and load the published
        # pretrained state dictionary explicitly, never random/fallback weights.
        from vggish_reference import VGGish, VGGishInputProcessor
        self.proc = VGGishInputProcessor()
        self.model = VGGish()
        weights = torch.hub.load_state_dict_from_url(
            "https://download.pytorch.org/torchaudio/models/vggish.pt",
            map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()
        self.kind = "torchaudio_reference"

    def __call__(self, x16: np.ndarray) -> np.ndarray:
        torch = self.torch
        x16 = tile_to(x16, int(VGGISH_MIN_SECONDS * 16000))
        with torch.no_grad():
            if self.kind in ("torchaudio", "torchaudio_reference"):
                emb = self.model(self.proc(torch.from_numpy(x16.astype(np.float32))))
            else:
                emb = self.model.forward(x16.astype(np.float32), 16000)
        emb = emb.reshape(-1, self.dim)
        return emb.mean(0).cpu().numpy()


def prepare(x: np.ndarray, fs: int, band4k: bool):
    """모델 입력용 16 kHz 파형. band4k 면 먼저 8 kHz 로 내려 4 kHz 위를 없앤다. 쓸 수 없으면 None."""
    x = x - x.mean()
    peak = np.abs(x).max()
    if peak > 0:
        x = x / peak * 0.5                                  # 녹음 이득은 임의이므로 크기를 맞춘다
    if band4k:
        if fs < 8000:
            return None
        x = resample_to(x, fs, 8000)
        return resample_to(x, 8000, 16000)
    return resample_to(x, fs, 16000)


def selftest(emb) -> int:
    rng = np.random.default_rng(0)
    ok = True
    for fs, dur in [(81920, 0.3), (10000, 2.0), (16000, 15.0)]:
        t = np.arange(int(fs * dur)) / fs
        x = 0.1 * np.sin(2 * np.pi * 1200 * t) + 0.01 * rng.standard_normal(len(t))
        for band in (False, True):
            x16 = prepare(x, fs, band)
            v = emb(x16)
            good = v.shape == (emb.dim,) and np.isfinite(v).all()
            ok &= bool(good)
            log(f"  fs {fs:>6}, {dur:>4.1f}초, band4k={band}: 모양 {v.shape}, 유한 {np.isfinite(v).all()}, 평균 {v.mean():+.4f}, 표준편차 {v.std():.4f}")
    log("자체 시험:", "통과" if ok else "실패")
    return 0 if ok else 1


def cache_token(audio_path: Path, settings: dict) -> str:
    stat = audio_path.stat()
    value = dict(settings=settings, path=str(audio_path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def load_checkpoint(path: Path, token: str, dim: int):
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["token"].item()) != token:
                return None
            value = data["X"].astype(np.float32)
            if value.shape != (dim,) or not np.isfinite(value).all():
                return None
            return value
    except (OSError, ValueError, KeyError, EOFError):
        return None


def save_checkpoint(path: Path, token: str, value: np.ndarray):
    temp = path.with_suffix(".npz.tmp")
    with temp.open("wb") as output:
        np.savez_compressed(output, token=token, X=value.astype(np.float32))
    temp.replace(path)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["ast", "vggish"])
    ap.add_argument("--pad", default="zero", choices=["zero", "wavzero", "tile"], help="ast 에만 쓴다")
    ap.add_argument("--band4k", action="store_true")
    ap.add_argument("--set", default="core", choices=["core", "gold", "all"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"], help="개발용은 dev. 기존 분할을 변경하지 않음")
    ap.add_argument("--no-resume", action="store_true", help="저장된 컷별 임베딩을 사용하지 않고 다시 계산")
    args = ap.parse_args()

    emb = AstEmbedder(args.pad) if args.model == "ast" else VggishEmbedder()
    log(f"모델 {args.model} 준비됨. 계산 스레드 {emb.threads}개")
    if args.selftest:
        return selftest(emb)
    name = (f"ast_{args.pad}" if args.model == "ast" else "vggish") + ("_band4k" if args.band4k else "")
    if args.split != "all":
        name += "_" + args.split

    m = pd.read_csv(DATA_DIR / "bench_manifest.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    sel = m[m.role == "core"] if args.set == "core" else (m[m.gold != ""] if args.set == "gold" else m)
    if args.split != "all":
        sel = sel[sel.split == args.split]
    idx = index_audio(load_config()["audio_root"])
    sel = sel[sel.cut_id.isin(idx)]
    if args.limit:
        sel = sel.head(args.limit)
    log(f"{name}: 대상 {len(sel)}개")
    cache = work_dir("embedding_cache", name)
    model_config = getattr(getattr(emb, "model", None), "config", None)
    settings = dict(model=args.model, pad=args.pad, band4k=args.band4k, backend=getattr(emb, "kind", "transformers"),
                    model_id=AST_NAME if args.model == "ast" else "vggish", threads=emb.threads,
                    model_revision=getattr(model_config, "_commit_hash", None),
                    torch_version=getattr(emb.torch, "__version__", ""), cache_version=1,
                    transformers_version=package_version("transformers"), torchaudio_version=package_version("torchaudio"),
                    numpy_version=np.__version__)
    ids, rows, skipped, resumed, t0 = [], [], 0, 0, time.time()
    for i, r in enumerate(sel.itertuples(index=False)):
        try:
            token = cache_token(Path(idx[r.cut_id]), settings)
            checkpoint = cache / (r.cut_id + ".npz")
            v = load_checkpoint(checkpoint, token, emb.dim) if not args.no_resume else None
            if v is not None:
                resumed += 1
            else:
                x, fs = read_audio(idx[r.cut_id], MAX_ANALYSIS_S)    # 핵심 표본은 60초 이하라 잘리지 않는다
                x16 = prepare(x, fs, args.band4k)
                if x16 is None or len(x16) < 400:
                    skipped += 1
                    continue
                v = emb(x16)
                if v.shape != (emb.dim,) or not np.isfinite(v).all():
                    raise ValueError("임베딩의 차원 또는 유한값 검사가 실패했습니다")
                save_checkpoint(checkpoint, token, v)
        except Exception as e:  # noqa: BLE001
            log(f"  실패 {r.cut_id}: {type(e).__name__}: {e}")
            skipped += 1
            continue
        ids.append(r.cut_id)
        rows.append(v.astype(np.float32))
        if (i + 1) % 10 == 0 or i + 1 == len(sel):
            log(f"  {i + 1}/{len(sel)}  {time.time() - t0:.0f}초  저장 {len(ids)}, 이어 사용 {resumed}")
    if not rows:
        log("임베딩이 하나도 나오지 않았다")
        return 1
    X = np.vstack(rows)
    info = dict(model=args.model, pad=args.pad if args.model == "ast" else "", band4k=args.band4k, skipped=skipped,
                backend=getattr(emb, "kind", "transformers"), threads=emb.threads,
                split=args.split, resumed=resumed, cache_settings=settings,
                note="원문 조건의 재현이 아니라 같은 표현을 이 벤치마크에 적용한 것")
    np.savez_compressed(work_dir("features") / f"{name}.npz", cut_id=np.array(ids), X=X,
                        names=np.array([f"e{i}" for i in range(X.shape[1])]), info=json.dumps(info, ensure_ascii=False))
    log(f"저장 {name}: {X.shape}, 건너뜀 {skipped}, {time.time() - t0:.0f}초")
    return 0


if __name__ == "__main__":
    sys.exit(main())
