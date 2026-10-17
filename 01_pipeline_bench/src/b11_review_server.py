"""Local review and re-analysis app. python src/b11_review_server.py [--no-browser]"""
from __future__ import annotations

import argparse
import csv
import io
import importlib.util
import json
import math
import mimetypes
import os
import random
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
import uuid
import wave
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from wb_common import DATA_DIR, GROUP_KO, PKG_ROOT, TYPE_CLASSES, TYPE_KO, index_audio, load_config
from wb_review_store import FIELDS, ReviewStore, csv_bytes, normalize, now, write_snapshot
from b12_feature_study import cached_references

WEB = PKG_ROOT / "web"
BASELINE_FEATURES = {"meta", "mfcc16k", "mfcc_band4k", "physv0"}
ID_PATTERN = re.compile(r"^[A-Za-z0-9]{8}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def atomic_json(path, value):
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def acquire_workspace_lock(root):
    """Only one server may update a workspace, including jobs recovered at startup."""
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / "server.lock").open("a+b")
    if handle.seek(0, 2) == 0:
        handle.write(b"0"); handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


class Application:
    def __init__(self):
        self.cfg = load_config()
        self.work = Path(self.cfg["work_root"])
        self.root = self.work / "review_loop"
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = ReviewStore(self.root)
        self.manifest = pd.read_csv(DATA_DIR / "bench_manifest.csv", dtype=str, keep_default_na=False)
        self.core = pd.read_csv(DATA_DIR / "bench_core.csv", dtype=str, keep_default_na=False)
        self.records = self.manifest.set_index("cut_id", drop=False).to_dict("index")
        self.audio = index_audio(self.cfg["audio_root"])
        self.lock = threading.RLock()
        self.render_lock = threading.Lock()
        self.active = None
        self.process = None
        self.cancel = threading.Event()
        self.jobs = {}
        for p in sorted((self.root / "runs").glob("*/job.json")):
            try:
                job = json.loads(p.read_text(encoding="utf-8"))
                if job["status"] in ("queued", "running"):
                    job.update(status="failed", error="이전 실행이 종료되었습니다. 다시 분석을 눌러 주세요.", finished_at=now())
                    atomic_json(p, job)
                self.jobs[job["id"]] = job
            except (ValueError, KeyError, OSError):
                continue
        self.predictions_cache = None
        self.physical_cache = None

    def feature_names(self):
        return sorted(p.stem for p in (self.work / "features").glob("*.npz") if SAFE_NAME.fullmatch(p.stem))

    def reference_status(self):
        cached = cached_references(self.work / "features")
        missing = [name for name in ("torch", "torchaudio", "transformers") if importlib.util.find_spec(name) is None]
        executable = '& "' + sys.executable + '"' if os.name == "nt" else shlex.quote(sys.executable)
        return dict(cached=cached, ast_cached=next((n for n in cached if n.startswith("ast_")), None),
                    missing=missing, install_command=executable + " -m pip install torch torchaudio transformers")

    def status(self):
        annotations = self.store.all()
        counts = {"development": 0, "evaluation": 0, "excluded": 0, "hold": 0}
        type_counts = dict.fromkeys(TYPE_CLASSES, 0)
        for a in annotations:
            counts[a["review_mode"]] += 1
            counts["excluded"] += a["usable"] == "n"
            counts["hold"] += a["usable"] == "hold"
            for t in a["types"]:
                if t in type_counts:
                    type_counts[t] += 1
        scope = self.core[(self.core.pool == "main") & (self.core.split == "dev")]
        return dict(audio_root=self.cfg["audio_root"], audio_found=len(self.audio),
                    core_available=int(self.core.cut_id.isin(self.audio).sum()),
                    dev_available=int(scope.cut_id.isin(self.audio).sum()),
                    saved=len(annotations), counts=counts, type_counts=type_counts,
                    features=self.feature_names(), types=[dict(value=t, label=TYPE_KO[t]) for t in TYPE_CLASSES],
                    groups=GROUP_KO, active=self.active,
                    reference=self.reference_status(),
                    runs=sorted(self.jobs.values(), key=lambda j: j["created_at"], reverse=True)[:30])

    def record(self, cut_id):
        if not ID_PATTERN.fullmatch(cut_id) or cut_id not in self.records:
            raise ValueError("샘플을 찾을 수 없습니다.")
        return self.records[cut_id]

    def cut(self, cut_id):
        r = self.record(cut_id)
        return dict(cut_id=cut_id, fs=int(float(r["fs_hz"])), duration=float(r["dur_s"]),
                    mode="evaluation" if r["split"] == "test" else "development",
                    available=cut_id in self.audio, annotation=self.store.get(cut_id))

    def sample(self, body):
        if body.get("source") == "failures":
            return self.failure_sample(body)
        count = int(body.get("count", 8))
        if not 1 <= count <= 50:
            raise ValueError("한 번에 1~50개를 불러올 수 있습니다.")
        mode = body.get("mode", "development")
        if mode not in ("development", "evaluation"):
            raise ValueError("검사 모드를 확인해 주세요.")
        split = "dev" if mode == "development" else "test"
        d = self.core[(self.core.pool == "main") & (self.core.split == split)]
        group = body.get("group", "all")
        if group != "all":
            if group not in GROUP_KO:
                raise ValueError("분류군을 확인해 주세요.")
            d = d[d.group3 == group]
        d = d[d.cut_id.isin(self.audio)]
        if not body.get("include_reviewed", False):
            d = d[~d.cut_id.isin({a["cut_id"] for a in self.store.all()})]
        if d.empty:
            raise ValueError("조건에 맞는 미검사 음원이 없습니다. 음원 폴더나 검사 조건을 확인해 주세요.")
        seed = random.SystemRandom().getrandbits(63)
        ids = random.Random(seed).sample(d.cut_id.tolist(), min(count, len(d)))
        draw_id = uuid.uuid4().hex
        self.store.draw(draw_id, dict(seed=seed, created_at=now(), options=body, cut_ids=ids))
        return dict(draw_id=draw_id, remaining=len(d), cuts=[self.cut(c) for c in ids])

    def failure_sample(self, body):
        count = int(body.get("count", 8))
        if not 1 <= count <= 50:
            raise ValueError("한 번에 1~50개를 불러올 수 있습니다.")
        completed = sorted((j for j in self.jobs.values() if j["status"] == "completed"),
                           key=lambda j: j["created_at"], reverse=True)
        for job in completed:
            path = self.work / "eval" / job["id"] / "review_candidates.csv"
            if path.exists():
                candidates = pd.read_csv(path, dtype=str, keep_default_na=False)
                break
        else:
            raise ValueError("분석 결과 페이지에서 먼저 '특징군 비교'를 실행해 주세요.")
        seen = set() if body.get("include_reviewed", False) else {a["cut_id"] for a in self.store.all()}
        ids = [c for c in candidates.cut_id if c in self.records and c in self.audio and c not in seen
               and self.records[c]["split"] == "dev" and self.records[c]["pool"] == "main"]
        group = body.get("group", "all")
        if group != "all":
            if group not in GROUP_KO:
                raise ValueError("분류군을 확인해 주세요.")
            ids = [c for c in ids if self.records[c]["group3"] == group]
        ids = list(dict.fromkeys(ids))
        if not ids:
            raise ValueError("조건에 맞는 미검사 불일치 후보가 없습니다. 무작위 검사 또는 검사한 샘플 포함을 사용하세요.")
        chosen = ids[:count]
        draw_id = uuid.uuid4().hex
        self.store.draw(draw_id, dict(created_at=now(), options=body, source_run=job["id"], cut_ids=chosen))
        return dict(draw_id=draw_id, remaining=len(ids), source="failures", source_run=job["id"], cuts=[self.cut(c) for c in chosen])

    def save(self, body, draft=False):
        r = self.record(str(body.get("cut_id", "")))
        value = normalize(body, r, complete=not draft)
        return self.store.save_many([value], draft=draft)[0]

    def import_annotations(self, body):
        text = body.get("text", "")
        if not isinstance(text, str):
            raise ValueError("표기 파일을 확인해 주세요.")
        if text.lstrip("\ufeff \n\r").startswith(("{", "[")):
            data = json.loads(text.lstrip("\ufeff"))
            rows = data.get("annotations", []) if isinstance(data, dict) else data
        else:
            reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
            required = {"cut_id", "target_present", "types", "usable", "confidence"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("검사 화면에서 내보낸 표기 CSV 또는 JSON을 선택해 주세요.")
            rows = list(reader)
        if not isinstance(rows, list):
            raise ValueError("표기 목록이 올바르지 않습니다.")
        values, ignored, unfinished = [], 0, 0
        for raw in rows:
            if not isinstance(raw, dict):
                raise ValueError("표기 목록의 행이 올바르지 않습니다.")
            r = self.records.get(str(raw.get("cut_id", "")))
            if not r:
                ignored += 1
                continue
            if not all(str(raw.get(k, "") or "") for k in ("target_present", "usable", "confidence")):
                unfinished += 1
                continue
            values.append(normalize(raw, r))
        # Validate the entire file before changing any completed annotation.
        saved = self.store.save_many(values)
        return dict(imported=len(saved), ignored=ignored, unfinished=unfinished)

    def _physical(self, cut_id):
        p = self.work / "features" / "physv0.npz"
        if not p.exists():
            return {}
        stamp = (p.stat().st_mtime_ns, p.stat().st_size)
        with self.lock:
            if self.physical_cache is None or self.physical_cache[0] != stamp:
                with np.load(p, allow_pickle=False) as n:
                    names = n["names"].astype(str).tolist()
                    values = {str(i): dict(zip(names, row.tolist())) for i, row in zip(n["cut_id"], n["X"])}
                self.physical_cache = (stamp, values)
            return self.physical_cache[1].get(cut_id, {})

    def automatic(self, cut_id):
        r = self.record(cut_id)
        self.store.expose(cut_id)
        f = self._physical(cut_id)
        def value(name, transform=lambda v: v):
            v = f.get(name)
            return transform(v) if v is not None and math.isfinite(v) else None
        metrics = [
            dict(label="신호 대역 하한", value=value("log_sig_f05", lambda v: 10 ** v), unit="Hz"),
            dict(label="신호 대역 중앙", value=value("log_sig_f50", lambda v: 10 ** v), unit="Hz"),
            dict(label="신호 대역 상한", value=value("log_sig_f95", lambda v: 10 ** v), unit="Hz"),
            dict(label="음조 집중도", value=value("tonal_conc"), unit=""),
            dict(label="주파수 변화 속도", value=value("log1p_fm_oct_per_s", math.expm1), unit="octave/s"),
            dict(label="활동 시간 비율", value=value("duty"), unit=""),
            dict(label="사건 길이 중앙값", value=value("log_event_dur_med", lambda v: 10 ** v), unit="s"),
            dict(label="충격음 탐지 빈도", value=value("log1p_tr_clicks_per_s", math.expm1), unit="/s"),
            dict(label="충격음 간격 중앙값", value=value("tr_log_ici_med", lambda v: 1000 * 10 ** v), unit="ms"),
        ]
        prediction = None
        complete = sorted((j for j in self.jobs.values() if j["status"] == "completed"),
                          key=lambda j: j["created_at"], reverse=True)
        for job in complete[:1]:
            p = self.work / "eval" / job["id"] / "predictions.csv"
            if p.exists():
                with self.lock:
                    if self.predictions_cache is None or self.predictions_cache[0] != str(p):
                        d = pd.read_csv(p, dtype=str, keep_default_na=False)
                        self.predictions_cache = (str(p), d)
                    d = self.predictions_cache[1]
                    d = d[(d.cut_id == cut_id) & (d.task == "type_single")]
                    if not d.empty:
                        prefer = d[d.feature == "physv0"]
                        v = (prefer if not prefer.empty else d).iloc[0]
                        prediction = dict(type=v["pred"], feature=v["feature"],
                                          probability=float(v["p_max"]), protocol=v["protocol"])
        return clean_json(dict(types=r.get("type_weak", "").split(";") if r.get("type_weak") else [],
                               note=r.get("note", ""), species=r.get("species_ko", ""),
                               metrics=metrics, prediction=prediction))

    def media(self, cut_id):
        self.record(cut_id)
        if cut_id not in self.audio:
            raise ValueError("음원을 찾을 수 없습니다. 음원 폴더를 확인해 주세요.")
        path = Path(self.audio[cut_id])
        stamp = f"{path.stat().st_size}-{path.stat().st_mtime_ns}"
        folder = self.root / "media" / cut_id
        folder.mkdir(parents=True, exist_ok=True)
        info_path = folder / "info.json"
        with self.render_lock:
            if info_path.exists():
                info = json.loads(info_path.read_text(encoding="utf-8"))
                if info.get("stamp") == stamp and (folder / "plot.png").exists() and (folder / "audio.wav").exists():
                    return info
            from b09_make_review_page import listening_rate, signal_band, spectrogram, _font_setup
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            if path.suffix.lower() == ".wav":
                from scipy.io import wavfile
                fs, x = wavfile.read(path)
                if x.ndim > 1:
                    x = x[:, 0]
                if np.issubdtype(x.dtype, np.integer):
                    if x.dtype == np.uint8:
                        x = (x.astype(float) - 128) / 128
                    else:
                        x = x.astype(float) / max(abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max)
                else:
                    x = x.astype(float)
            else:
                from wb_common import read_audio
                x, fs = read_audio(str(path), 60)
            full_duration = len(x) / fs
            x = x[:int(fs * 60)]
            if not len(x):
                raise ValueError("음원이 비어 있습니다.")
            x = np.nan_to_num(x - x.mean())
            _font_setup()
            spec, freq, times, nfft, hop = spectrogram(x, fs)
            lo, hi = signal_band(spec, freq)
            fig, axes = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True,
                                     gridspec_kw={"height_ratios": [1, 2, 2]})
            step = max(1, len(x) // 20000)
            axes[0].plot(np.arange(len(x))[::step] / fs, x[::step], lw=.6, color="#1c7071")
            axes[0].set_ylabel("Amplitude")
            axes[0].set_title(f"{cut_id}  |  {fs:,} Hz  |  {len(x)/fs:.3f} s")
            top = np.percentile(spec, 99.9)
            for ax in axes[1:]:
                ax.imshow(spec, origin="lower", aspect="auto", extent=[0, len(x)/fs, freq[0], freq[-1]],
                          cmap="magma", vmax=top, vmin=max(np.percentile(spec, 5), top-70))
                ax.set_ylabel("Frequency (Hz)")
            axes[2].set_ylim(max(0, lo / 2), min(fs / 2, max(hi * 1.5, lo + fs/nfft)))
            axes[2].set_xlabel("Time (s) · signal band")
            try:
                fig.savefig(folder / "plot.png", dpi=100)
            finally:
                plt.close(fig)
            rate, speed = listening_rate(fs)
            gain = min(8.0, .9 / max(float(np.abs(x).max()), 1e-8))
            y = np.clip(x * gain * 32767, -32768, 32767).astype("<i2")
            with wave.open(str(folder / "audio.wav"), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(y.tobytes())
            info = dict(stamp=stamp, fs=int(fs), duration=len(x)/fs, full_duration=full_duration,
                        play_rate=rate, speed=speed, gain=gain, band_lo=lo, band_hi=hi)
            atomic_json(info_path, info)
        return info

    def _persist_job(self, job):
        atomic_json(self.root / "runs" / job["id"] / "job.json", job)

    def start_job(self, body):
        kind = body.get("kind", "evaluation")
        if kind not in ("evaluation", "study", "reference"):
            raise ValueError("분석 종류를 확인해 주세요.")
        features = body.get("features", ["meta", "mfcc16k", "physv0"]) if kind == "evaluation" else ["meta", "mfcc16k", "physv0"]
        if not isinstance(features, list) or not features or len(features) > 12:
            raise ValueError("평가할 특징을 하나 이상 선택해 주세요.")
        allowed = set(self.feature_names()) | BASELINE_FEATURES
        if any(not isinstance(f, str) or f not in allowed or not SAFE_NAME.fullmatch(f) for f in features):
            raise ValueError("특징 이름을 확인해 주세요.")
        boot = int(body.get("boot", 500))
        if boot not in (100, 500, 2000):
            raise ValueError("분석 정밀도를 확인해 주세요.")
        exclude = body.get("exclude_classes", [])
        if not isinstance(exclude, list) or any(t not in TYPE_CLASSES for t in exclude):
            raise ValueError("제외할 형태를 확인해 주세요.")
        if kind == "reference" and not self.reference_status()["ast_cached"]:
            missing = self.reference_status()["missing"]
            if missing:
                raise ValueError("AST 준비에 필요한 패키지가 없습니다: " + ", ".join(missing) + ". 화면의 설치 명령을 먼저 실행하세요.")
            if not self.audio:
                raise ValueError("AST 임베딩을 만들 음원 폴더를 먼저 설정해 주세요.")
        with self.lock:
            if self.active and self.jobs[self.active]["status"] in ("queued", "running"):
                raise ValueError("분석이 진행 중입니다. 완료 후 다시 실행할 수 있습니다.")
            revision, rows = self.store.snapshot()
            job_id = "review_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            folder = self.root / "runs" / job_id
            stats = write_snapshot(self.core, rows, revision, folder)
            job = dict(id=job_id, kind=kind, status="queued", created_at=now(), stage="준비 중", error="",
                       revision=revision, features=list(dict.fromkeys(features)), boot=boot,
                       include_test=bool(body.get("include_test", False)) if kind == "evaluation" else False,
                       include_references=bool(body.get("include_references", True)) or kind == "reference",
                       exclude_classes=exclude, summary=stats)
            self.jobs[job_id] = job
            self.active = job_id
            self.cancel.clear()
            self._persist_job(job)
            threading.Thread(target=self._run_job, args=(job,), daemon=True).start()
            return dict(job_id=job_id, revision=revision)

    def _command(self, args, job, stage):
        if self.cancel.is_set():
            raise InterruptedError("분석을 중지했습니다.")
        job["stage"] = stage
        self._persist_job(job)
        log_path = self.root / "runs" / job["id"] / "analysis.log"
        env = dict(os.environ, PYTHONIOENCODING="utf-8", WATKINS_WORK_ROOT=str(self.work.resolve()),
                   WATKINS_AUDIO_ROOT=self.cfg["audio_root"])
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{now()}] {stage}\n"); log.flush()
            p = subprocess.Popen([sys.executable, "-u", *args], cwd=PKG_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, shell=False)
            with self.lock:
                self.process = p
                if self.cancel.is_set():
                    p.terminate()
            code = p.wait()
            with self.lock:
                self.process = None
            if self.cancel.is_set():
                raise InterruptedError("분석을 중지했습니다.")
            if code:
                raise RuntimeError(f"{stage} 중 오류가 발생했습니다. 실행 로그를 확인해 주세요.")

    def _run_job(self, job):
        try:
            job.update(status="running", started_at=now())
            self._persist_job(job)
            missing = [f for f in job["features"] if not (self.work / "features" / (f + ".npz")).exists()]
            if missing:
                if any(f not in BASELINE_FEATURES for f in missing):
                    raise ValueError("사전학습 특징 파일을 먼저 생성해 주세요.")
                if any(f != "meta" for f in missing) and not self.audio:
                    raise ValueError("특징 계산에 사용할 음원 폴더를 설정해 주세요.")
                self._command(["src/b05_features_baseline.py", "--set", "core", "--only", ",".join(missing)],
                              job, "빠진 특징 계산")
            folder = self.root / "runs" / job["id"]
            study = job.get("kind", "evaluation") != "evaluation"
            if job.get("kind") == "reference" and not self.reference_status()["ast_cached"]:
                self._command(["src/b06_embed_pretrained.py", "--model", "ast", "--pad", "zero", "--selftest"],
                              job, "AST 자체 시험·최초 가중치 준비")
                self._command(["src/b06_embed_pretrained.py", "--model", "ast", "--pad", "zero", "--set", "core", "--split", "dev"],
                              job, "AST 개발 임베딩 계산·컷별 이어 저장")
            manifest = folder / "reviewed_manifest.csv"
            if study:
                prepare_command = ["src/b12_feature_study.py", "prepare", "--manifest", str(manifest), "--out", str(folder)]
                if not job["include_references"]:
                    prepare_command += ["--no-references"]
                self._command(prepare_command, job, "개발 자료·특징군 사본 준비")
                plan = json.loads((folder / "study_plan.json").read_text(encoding="utf-8"))
                job["features"] = plan["features"]
                self._persist_job(job)
                manifest = folder / "study_manifest.csv"
            protocols = "devcv,test" if job["include_test"] else "devcv"
            command = ["src/b07_evaluate.py", "--features", ",".join(job["features"]), "--run", job["id"],
                       "--manifest", str(manifest), "--protocols", protocols,
                       "--boot", str(job["boot"]), "--no-random"]
            if study:
                command += ["--tasks", "group3,type_single,type_multi", "--feature-dir", str(folder / "features")]
            if job["exclude_classes"]:
                command += ["--exclude-classes", ",".join(job["exclude_classes"])]
            if job["include_test"] and job["summary"]["independent_usable"]:
                command += ["--gold", str(folder / "independent_gold.csv")]
            self._command(command, job, "수정 표기를 반영해 평가")
            if study:
                self._command(["src/b12_feature_study.py", "summarize", "--plan", str(folder / "study_plan.json"),
                               "--out", str(self.work / "eval" / job["id"])], job, "특징군 기여·녹음 조건·재검사 후보 요약")
            self._report_feedback(rows_path=folder / "annotations_snapshot.csv", folder=folder)
            result_dir = self.work / "eval" / job["id"]
            for p in folder.iterdir():
                if p.suffix in (".csv", ".json") and p.name != "job.json":
                    (result_dir / p.name).write_bytes(p.read_bytes())
            job.update(status="completed", stage="분석 완료", finished_at=now())
        except InterruptedError as e:
            job.update(status="cancelled", stage="중지됨", error=str(e), finished_at=now())
        except Exception as e:
            job.update(status="failed", stage="오류", error=str(e), finished_at=now())
            with (self.root / "runs" / job["id"] / "analysis.log").open("a", encoding="utf-8") as log:
                log.write(traceback.format_exc())
        finally:
            self._persist_job(job)

    def _report_feedback(self, rows_path, folder):
        rows = pd.read_csv(rows_path, dtype=str, keep_default_na=False).to_dict("records")
        detail = []
        for a in rows:
            r = self.records[a["cut_id"]]
            human = set(a["types"].split(";")) & set(TYPE_CLASSES)
            weak = set(r.get("type_weak", "").split(";")) & set(TYPE_CLASSES)
            detail.append(dict(cut_id=a["cut_id"], mode=a["review_mode"], group3=r["group3"],
                               weak_types=";".join(t for t in TYPE_CLASSES if t in weak),
                               human_types=";".join(t for t in TYPE_CLASSES if t in human),
                               set_equal=human == weak, usable=a["usable"], confidence=a["confidence"],
                               meta_seen=a["meta_seen"], memo=a["memo"]))
        columns = ["cut_id", "mode", "group3", "weak_types", "human_types", "set_equal", "usable", "confidence", "meta_seen", "memo"]
        pd.DataFrame(detail, columns=columns).to_csv(folder / "label_audit.csv", index=False, encoding="utf-8-sig")

    def stop_job(self, job_id):
        with self.lock:
            if job_id != self.active or self.jobs[job_id]["status"] not in ("queued", "running"):
                raise ValueError("진행 중인 분석이 아닙니다.")
            self.cancel.set()
            if self.process and self.process.poll() is None:
                self.process.terminate()
        return dict(stopping=True)

    def job_result(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError("분석 결과를 찾을 수 없습니다.")
            job = dict(self.jobs[job_id])
        folder = self.root / "runs" / job_id
        log_path = folder / "analysis.log"
        if log_path.exists():
            with log_path.open("rb") as f:
                f.seek(max(0, log_path.stat().st_size - 18000))
                job["log"] = f.read().decode("utf-8", errors="replace")
        result_dir = self.work / "eval" / job_id
        job["results"] = []
        job["study"] = None
        if job["status"] == "completed" and (result_dir / "results.csv").exists():
            try:
                d = pd.read_csv(result_dir / "results.csv", keep_default_na=False)
                d = d[(d.subset == "all") & d.metric.isin(["balanced_acc", "macro_auroc"])]
                job["results"] = clean_json(d.to_dict("records"))
            except pd.errors.EmptyDataError:
                pass
        if job["status"] == "completed" and (result_dir / "study_summary.json").exists():
            job["study"] = json.loads((result_dir / "study_summary.json").read_text(encoding="utf-8"))
        job["downloads"] = [p.name for p in result_dir.glob("*") if p.is_file() and p.suffix in (".csv", ".md", ".json")]
        return job

    def export_run(self, job_id):
        if job_id not in self.jobs or self.jobs[job_id]["status"] != "completed":
            raise ValueError("완료된 분석을 선택해 주세요.")
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted((self.work / "eval" / job_id).glob("*")):
                if p.is_file() and p.suffix in (".csv", ".md", ".json"):
                    z.write(p, p.name)
            for name in ("analysis.log", "job.json"):
                p = self.root / "runs" / job_id / name
                if p.exists():
                    z.write(p, name)
        return output.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version = "WatkinsReview/1.0"

    @property
    def app(self):
        return self.server.app

    def log_message(self, format, *args):
        if " 500 " in format % args:
            super().log_message(format, *args)

    def send(self, data, content_type="application/json; charset=utf-8", code=200, filename=None):
        if isinstance(data, (dict, list)):
            data = json.dumps(clean_json(data), ensure_ascii=False, allow_nan=False).encode("utf-8")
        elif isinstance(data, str):
            data = data.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def error(self, e, code=400):
        self.send(dict(error=str(e)), code=code)

    def do_GET(self):
        try:
            if not self.local_host():
                return self.error("로컬 주소로 접속해 주세요.", 403)
            url = urlparse(self.path)
            path, query = url.path, parse_qs(url.query)
            if path == "/":
                self.send_response(302); self.send_header("Location", "/review"); self.end_headers(); return
            if path in ("/review", "/analysis"):
                return self.send((WEB / (path[1:] + ".html")).read_bytes(), "text/html; charset=utf-8")
            if path in ("/app.css", "/app.js", "/review.js", "/analysis.js"):
                kind = "text/css; charset=utf-8" if path.endswith(".css") else "text/javascript; charset=utf-8"
                return self.send((WEB / path[1:]).read_bytes(), kind)
            if path == "/favicon.ico":
                return self.send(b"", "image/x-icon", 204)
            if path == "/api/status":
                return self.send(self.app.status())
            if path == "/api/annotations":
                return self.send(sorted(self.app.store.all(), key=lambda a: a["updated_at"], reverse=True))
            if path == "/api/cut":
                return self.send(self.app.cut(query.get("id", [""])[0]))
            if path == "/api/automatic":
                return self.send(self.app.automatic(query.get("id", [""])[0]))
            if path == "/api/media":
                cut_id = query.get("id", [""])[0]
                info = self.app.media(cut_id)
                return self.send(dict(**info, image=f"/media/{cut_id}/plot.png", audio=f"/media/{cut_id}/audio.wav"))
            if path.startswith("/media/"):
                parts = path.split("/")
                if len(parts) != 4 or parts[3] not in ("plot.png", "audio.wav"):
                    raise ValueError("파일을 찾을 수 없습니다.")
                self.app.record(parts[2])
                p = self.app.root / "media" / parts[2] / parts[3]
                if not p.exists():
                    raise ValueError("음원 화면을 먼저 불러와 주세요.")
                return self.send(p.read_bytes(), "image/png" if p.suffix == ".png" else "audio/wav")
            if path == "/api/job":
                return self.send(self.app.job_result(query.get("id", [""])[0]))
            if path == "/api/export/annotations":
                rows = self.app.store.all()
                if query.get("format", ["csv"])[0] == "json":
                    return self.send(dict(version=1, annotations=rows), filename="watkins_reviews.json")
                return self.send(csv_bytes(rows), "text/csv; charset=utf-8", filename="watkins_reviews.csv")
            if path == "/api/export/run":
                job_id = query.get("id", [""])[0]
                return self.send(self.app.export_run(job_id), "application/zip", filename=job_id + ".zip")
            if path == "/api/download":
                job_id, name = query.get("id", [""])[0], query.get("file", [""])[0]
                if job_id not in self.app.jobs or name != Path(name).name:
                    raise ValueError("결과 파일을 찾을 수 없습니다.")
                p = self.app.work / "eval" / job_id / name
                if not p.is_file() or p.suffix not in (".csv", ".json", ".md"):
                    raise ValueError("결과 파일을 찾을 수 없습니다.")
                return self.send(p.read_bytes(), mimetypes.guess_type(name)[0] or "application/octet-stream", filename=name)
            self.send(dict(error="페이지를 찾을 수 없습니다."), code=404)
        except (ValueError, KeyError, FileNotFoundError) as e:
            self.error(e)
        except Exception as e:
            traceback.print_exc(); self.error(e, 500)

    def do_POST(self):
        try:
            if not self.local_host():
                return self.error("로컬 주소로 접속해 주세요.", 403)
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.error("다른 페이지에서 보낸 요청입니다.", 403)
            if "application/json" not in self.headers.get("Content-Type", ""):
                return self.error("요청 형식을 확인해 주세요.", 415)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8 * 1024 * 1024:
                return self.error("표기 파일은 8 MB 이내로 불러와 주세요.", 413)
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("요청 내용을 확인해 주세요.")
            path = urlparse(self.path).path
            if path == "/api/sample":
                return self.send(self.app.sample(body))
            if path in ("/api/save", "/api/draft"):
                return self.send(self.app.save(body, draft=path == "/api/draft"))
            if path == "/api/import":
                return self.send(self.app.import_annotations(body))
            if path == "/api/run":
                return self.send(self.app.start_job(body))
            if path == "/api/stop":
                return self.send(self.app.stop_job(body.get("id", "")))
            if path == "/api/settings":
                audio_root = str(body.get("audio_root", "")).strip().strip('"')
                if not Path(audio_root).is_dir():
                    raise ValueError("실제로 존재하는 음원 폴더 경로를 입력해 주세요.")
                settings = PKG_ROOT / "watkins_bench_config.local.json"
                cfg = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
                cfg["audio_root"] = audio_root
                atomic_json(settings, cfg)
                with self.app.lock:
                    self.app.cfg["audio_root"] = audio_root
                    self.app.audio = index_audio(audio_root)
                return self.send(self.app.status())
            return self.send(dict(error="요청 경로를 찾을 수 없습니다."), code=404)
        except (ValueError, KeyError, TypeError, FileNotFoundError) as e:
            self.error(e)
        except Exception as e:
            traceback.print_exc(); self.error(e, 500)

    def local_host(self):
        host = self.headers.get("Host", "")
        return host in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")


def main():
    parser = argparse.ArgumentParser(description="Watkins 검사·재분석 화면")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    root = Path(load_config()["work_root"]) / "review_loop"
    workspace_lock = acquire_workspace_lock(root)
    if workspace_lock is None:
        print("검사·재분석 화면이 이미 실행 중입니다. 기존 실행 창을 사용합니다.", flush=True)
        try:
            url = json.loads((root / "server.json").read_text(encoding="utf-8"))["url"]
            if re.fullmatch(r"http://127\.0\.0\.1:[0-9]+/review", url):
                print(url, flush=True)
                if not args.no_browser:
                    webbrowser.open(url)
        except (OSError, ValueError, KeyError):
            pass
        return
    app = None
    server = None
    try:
        app = Application()
        for port in range(args.port, args.port + 20):
            try:
                server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
                break
            except OSError as e:
                if port == args.port + 19:
                    raise e
        server.app = app
        url = f"http://127.0.0.1:{server.server_port}/review"
        atomic_json(root / "server.json", dict(url=url, started_at=now()))
        print(f"\nWatkins 검사·재분석 화면: {url}", flush=True)
        print("창을 열어 둔 채로 사용합니다. 종료는 Ctrl+C.\n", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if app:
            with app.lock:
                if app.process and app.process.poll() is None:
                    app.cancel.set(); app.process.terminate()
        if server:
            server.server_close()
        (root / "server.json").unlink(missing_ok=True)
        workspace_lock.close()


if __name__ == "__main__":
    main()
