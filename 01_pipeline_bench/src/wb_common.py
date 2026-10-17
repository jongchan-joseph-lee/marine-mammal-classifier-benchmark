# -*- coding: utf-8 -*-
"""Watkins_Bench_v1 공통 모듈.

경로 설정, 종·분류군 표, 호출 태그 대응표, 음원 읽기처럼 여러 단계가 같이 쓰는 것을 모았다.
표준 라이브러리 + numpy 만 요구한다 (pandas, soundfile 은 쓰는 함수 안에서 불러온다).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

VERSION = "Watkins_Bench_v1.1"
SEED = 20261004

PKG_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PKG_ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
DOCS_DIR = PKG_ROOT / "docs"

# 윈도우 콘솔(cp949)에서 찍을 수 없는 글자가 있어도 멈추지 않게 한다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass

# 계산 스레드를 1개로 고정한다. 자료가 작아서 여러 스레드가 오히려 느리고,
# 스레드 수에 따라 끝자리가 달라지는 것을 앞선 묶음(Watkins_Review_v1)에서 확인했다.
_THREADS = os.environ.get("WATKINS_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)


# --------------------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "audio_root": "D:/Watkins_Exp0/audio_original/watkins_wmmsdb",
    "work_root": "",  # 비우면 묶음 폴더 아래 work/ 를 쓴다
}


def csv_arg(value) -> list[str]:
    """명령줄의 목록 값을 낱개로 나눈다. "a,b", ["a,b"], ["a", "b"], ["a,b", "c"] 를 모두 받는다.

    PowerShell 은 쉼표로 이은 값(a,b,c)을 띄어 쓴 값 여러 개로 바꿔 넘긴다. 그래서 목록을 받는 옵션은 두 꼴을 다 받게 했다.
    """
    if isinstance(value, str):
        value = [value]
    return [x.strip() for part in (value or []) for x in str(part).split(",") if x.strip()]


def load_config() -> dict:
    """watkins_bench_config.json → .local.json → 환경 변수 순으로 덮어쓴다."""
    cfg = dict(DEFAULT_CONFIG)
    for name in ("watkins_bench_config.json", "watkins_bench_config.local.json"):
        p = PKG_ROOT / name
        if p.exists():
            cfg.update(json.loads(p.read_text(encoding="utf-8")))
    if os.environ.get("WATKINS_AUDIO_ROOT"):
        cfg["audio_root"] = os.environ["WATKINS_AUDIO_ROOT"]
    if os.environ.get("WATKINS_WORK_ROOT"):
        cfg["work_root"] = os.environ["WATKINS_WORK_ROOT"]
    if not cfg.get("work_root"):
        cfg["work_root"] = str(PKG_ROOT / "work")
    return cfg


def work_dir(*parts: str) -> Path:
    p = Path(load_config()["work_root"]).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------------------
# 종 표: 원자료의 종 이름 → (분류군, 국명, 영문명)
# 분류군은 연구자가 정한 3개(수염고래류·이빨고래류·기각류). 해달은 기각류가 아니므로 '기타'.
# --------------------------------------------------------------------------------------
SPECIES = {
    # 수염고래류
    "Balaena mysticetus": ("Mysticeti", "북극고래", "Bowhead whale"),
    "Balaenoptera acutorostrata": ("Mysticeti", "밍크고래", "Common minke whale"),
    "Balaenoptera musculus": ("Mysticeti", "대왕고래", "Blue whale"),
    "Balaenoptera physalus": ("Mysticeti", "참고래", "Fin whale"),
    "Eschrichtius robustus": ("Mysticeti", "귀신고래", "Gray whale"),
    "Eubalaena australis": ("Mysticeti", "남방긴수염고래", "Southern right whale"),
    "Eubalaena glacialis": ("Mysticeti", "북대서양긴수염고래", "North Atlantic right whale"),
    "Megaptera novaeangliae": ("Mysticeti", "혹등고래", "Humpback whale"),
    # 이빨고래류
    "Cephalorhynchus commersonii": ("Odontoceti", "커머슨돌고래", "Commerson's dolphin"),
    "Cephalorhynchus heavisidii": ("Odontoceti", "헤비사이드돌고래", "Heaviside's dolphin"),
    "Delphinapterus leucas": ("Odontoceti", "흰고래(벨루가)", "Beluga"),
    "Delphinus bairdii": ("Odontoceti", "긴부리참돌고래", "Long-beaked common dolphin"),
    "Delphinus delphis": ("Odontoceti", "참돌고래", "Common dolphin"),
    "Globicephala macrorhynchus": ("Odontoceti", "들쇠고래", "Short-finned pilot whale"),
    "Globicephala melas": ("Odontoceti", "참거두고래", "Long-finned pilot whale"),
    "Grampus griseus": ("Odontoceti", "큰머리돌고래", "Risso's dolphin"),
    "Inia geoffrensis": ("Odontoceti", "아마존강돌고래", "Amazon river dolphin"),
    "Lagenodelphis hosei": ("Odontoceti", "사라와크돌고래", "Fraser's dolphin"),
    "Lagenorhynchus acutus": ("Odontoceti", "대서양낫돌고래", "Atlantic white-sided dolphin"),
    "Lagenorhynchus albirostris": ("Odontoceti", "흰부리돌고래", "White-beaked dolphin"),
    "Lagenorhynchus obscurus": ("Odontoceti", "더스키돌고래", "Dusky dolphin"),
    "Monodon monoceros": ("Odontoceti", "일각고래", "Narwhal"),
    "Orcinus orca": ("Odontoceti", "범고래", "Killer whale"),
    "Peponocephala electra": ("Odontoceti", "고양이고래", "Melon-headed whale"),
    "Pseudorca crassidens": ("Odontoceti", "흑범고래", "False killer whale"),
    "Sotalia fluviatilis": ("Odontoceti", "투쿠시", "Tucuxi"),
    "Sousa sahulensis": ("Odontoceti", "오스트레일리아혹등돌고래", "Australian humpback dolphin"),
    "Stenella attenuata": ("Odontoceti", "알락돌고래", "Pantropical spotted dolphin"),
    "Stenella clymene": ("Odontoceti", "클리멘돌고래", "Clymene dolphin"),
    "Stenella coeruleoalba": ("Odontoceti", "줄박이돌고래", "Striped dolphin"),
    "Stenella frontalis": ("Odontoceti", "대서양알락돌고래", "Atlantic spotted dolphin"),
    "Stenella longirostris": ("Odontoceti", "긴부리돌고래", "Spinner dolphin"),
    "Steno bredanensis": ("Odontoceti", "뱀머리돌고래", "Rough-toothed dolphin"),
    "Tursiops truncatus": ("Odontoceti", "큰돌고래", "Common bottlenose dolphin"),
    # 기각류
    "Arctocephalus forsteri": ("Pinnipedia", "뉴질랜드물개", "New Zealand fur seal"),
    "Arctocephalus philippii": ("Pinnipedia", "후안페르난데스물개", "Juan Fernandez fur seal"),
    "Cystophora cristata": ("Pinnipedia", "두건물범", "Hooded seal"),
    "Erignathus barbatus": ("Pinnipedia", "턱수염물범", "Bearded seal"),
    "Eumetopias jubatus": ("Pinnipedia", "큰바다사자", "Steller sea lion"),
    "Halichoerus grypus": ("Pinnipedia", "회색물범", "Gray seal"),
    "Histriophoca fasciata": ("Pinnipedia", "띠무늬물범", "Ribbon seal"),
    "Hydrurga leptonyx": ("Pinnipedia", "표범물범", "Leopard seal"),
    "Leptonychotes weddellii": ("Pinnipedia", "웨델물범", "Weddell seal"),
    "Odobenus rosmarus": ("Pinnipedia", "바다코끼리", "Walrus"),
    "Ommatophoca rossi": ("Pinnipedia", "로스물범", "Ross seal"),
    "Pagophilus groenlandicus": ("Pinnipedia", "하프물범", "Harp seal"),
    "Phoca largha": ("Pinnipedia", "점박이물범", "Spotted seal"),
    "Pusa hispida": ("Pinnipedia", "고리무늬물범", "Ringed seal"),
    # 기타 (기각류가 아님)
    "Enhydra lutris": ("Other", "해달", "Sea otter"),
}
GROUP_KO = {"Mysticeti": "수염고래류", "Odontoceti": "이빨고래류", "Pinnipedia": "기각류", "Other": "기타"}
GROUPS3 = ["Mysticeti", "Odontoceti", "Pinnipedia"]

# 지역 범위. 종의 서식 범위를 기준으로 한 표시이며, 컷의 녹음 장소가 아니다.
# 출처: 연구자가 GPT 와 정한 두 목록 (Watkins_Regional_Sample_v2 의 9종, Watkins_Full_3groups 의 후보 14종).
# 'later' 는 GPT 가 '추후 확장 후보'로 남긴 북극·아북극 기각류로, 연구자가 아직 정하지 않았다.
REGION_SCOPE = {
    "Balaena mysticetus": "arctic",
    "Delphinapterus leucas": "arctic",
    "Monodon monoceros": "arctic",
    "Pusa hispida": "arctic",
    "Odobenus rosmarus": "arctic",
    "Lagenorhynchus albirostris": "arctic",       # 북극 인접(북대서양 냉수역)
    "Orcinus orca": "arctic+korea",
    "Megaptera novaeangliae": "arctic+korea",
    "Phoca largha": "arctic+korea",
    "Eschrichtius robustus": "arctic+korea",
    "Balaenoptera acutorostrata": "arctic+korea",
    "Delphinus delphis": "korea",
    "Tursiops truncatus": "korea",
    "Grampus griseus": "korea",
    "Pseudorca crassidens": "korea",
    "Eumetopias jubatus": "korea",
    "Erignathus barbatus": "later",
    "Histriophoca fasciata": "later",
    "Pagophilus groenlandicus": "later",
    "Cystophora cristata": "later",
}
# 연구자가 주 표본에서 빼기로 한 종 (2026-10-04, 신호 품질 사유). 기록은 남기고 풀에서만 뺀다.
USER_EXCLUDED_SPECIES = {"Balaenoptera physalus"}

# --------------------------------------------------------------------------------------
# 샘플링 주파수 계열: 옥타브 폭. 경계는 5 k, 10 k, 20 k ... 의 기하평균.
# 'fs 계열'은 디지털화 조건이고 실제 기록 대역과는 다르다.
# --------------------------------------------------------------------------------------
LF_POOL_MAX_FS = 5000.0  # 이 값 미만이면 저주파(LF) 별도 풀
FS_FAMILIES = [  # (이름, 하한 포함, 상한 미포함)
    ("LF(<5k)", 0.0, 5000.0),
    ("5k", 5000.0, 7071.0),
    ("10k", 7071.0, 14142.0),
    ("20k", 14142.0, 28284.0),
    ("40k", 28284.0, 56569.0),
    ("80k", 56569.0, 113137.0),
    ("160k", 113137.0, 1e12),
]


def fs_family(fs: float) -> str:
    for name, lo, hi in FS_FAMILIES:
        if lo <= fs < hi:
            return name
    return "?"


# --------------------------------------------------------------------------------------
# 호출 태그 대응표 (v1 의 라벨. v1.1 부터 과제의 라벨은 wb_notes.py 가 노트를 읽어 만든다)
# 원자료(Earth Species Project 공개본)의 call_type 은 Watkins 노트 문구에서 뽑은 낱말이다.
# 전문가가 사건 단위로 확인한 정답이 아니라 컷 단위의 약한 라벨이다.
# 이 표는 표본 설계(핵심 표본·분할·겹·골드 목록)를 v1 과 똑같이 되풀이하는 데, 그리고 v1 라벨과 견주는 데 쓴다. 고치지 않는다.
# 연구자가 정한 6종(click, pulse train, moan, whistle, growl, bark)에 맞추고,
# 6종에 넣을 곳이 정해지지 않은 두 묶음(pulsed_call, song)은 따로 둔다 (결정 D-B).
# --------------------------------------------------------------------------------------
TYPE_CLASSES = ["click", "pulse_train", "whistle", "moan", "growl", "bark", "pulsed_call", "song"]
# 범주를 나누는 기준은 소리의 모양이다. 주파수 대역과 종은 기준이 아니다 (대역은 따로 잰다).
TYPE_KO = {
    "click": "클릭(낱개로 구분되는 충격음)",
    "pulse_train": "펄스열·버스트(빠르게 이어진 펄스)",
    "whistle": "휘슬·스윕(음조, 주파수가 뚜렷이 변함)",
    "moan": "모운(음조, 길고 변화가 작음)",
    "growl": "그롤·그런트(거칠고 낮은 소리)",
    "bark": "바크(짧고 거친 소리의 되풀이)",
    "pulsed_call": "펄스성 호출·스퀼(촘촘한 배음 줄무늬)",
    "song": "노래(단위가 차례로 이어짐)",
}
TAG_TO_TYPE = {
    "click": "click",
    "pulsed_click": "pulse_train", "burst_pulse": "pulse_train", "buzz": "pulse_train", "creak": "pulse_train",
    "whistle": "whistle", "chirp": "whistle", "upsweep": "whistle", "downsweep": "whistle",
    "trill": "whistle", "whine": "whistle", "warble": "whistle",
    "moan": "moan", "groan": "moan",
    "growl": "growl", "grunt": "growl",
    "bark": "bark",
    "squeal": "pulsed_call", "call": "pulsed_call", "scream": "pulsed_call", "trumpet": "pulsed_call",
    "song": "song",
    # knock, slap, thump 은 동물 발성이 아닐 수 있어(수중청음기 부딪힘 등) 어느 형태에도 넣지 않는다.
}


def map_type_tags(call_type: str) -> list[str]:
    out = []
    for t in (call_type or "").split(";"):
        t = t.strip()
        c = TAG_TO_TYPE.get(t)
        if c and c not in out:
            out.append(c)
    return [c for c in TYPE_CLASSES if c in out]


# --------------------------------------------------------------------------------------
# 노트에서 뽑는 표시 (정규식). 사람의 청취 판정을 대신하지 않는다.
# --------------------------------------------------------------------------------------
RE_CAPTIVE = re.compile(
    r"aquari|\bzoo\b|marine[\s-]*land|sea[\s-]*world|seaquar|oceanar|\bcaptiv|\btank\b|enclosure|"
    r"moby\s*doll|keiko|\boca\b|johns hopkins|silver springs|net corral|rock pool|laboratory of", re.I)
RE_AIR = re.compile(r"\bin\s*-?\s*air\b|recording on ice|on the ice\b|above water", re.I)
RE_BAD = re.compile(
    r"\b(?:poor|bad|noisy) (?:cut|recording|signal|sounds?)\b|\b(?:very|extremely) noisy\b|"
    r"\b(?:poor|bad) (?:speed|speed control)\b|\b(?:wrong|incorrect) (?:tape )?speed\b", re.I)
RE_NOISE = re.compile(r"noise|reverberat|echo|clunk|engine|radio|faint|weak|overlap|overload|distort", re.I)
RE_GOOD = re.compile(r"\b(?:good|excellent|great) (?:cut|sounds?|recording|series|tape)\b", re.I)
RE_SPEED = re.compile(r"speed|slowed|slower|faster|time.?compress|time.?expand", re.I)
RE_RESTREAM = re.compile(r"restream|streamed|re-?digiti", re.I)
RE_COPY = re.compile(r"\bcopy\b|\bcopied\b|made from (?:rn|orig)", re.I)


def norm_location(text: str) -> str:
    """장소 문구를 묶음 열쇠로 쓰기 위해 다듬는다. 끝에 붙은 낱자(O, X, BD)와 문장부호를 떼어 낸다."""
    t = (text or "").strip()
    t = re.sub(r"\s+(?:X|BD)\s*$", "", t)
    t = re.sub(r"(?<=[a-z])(?:O|X|x)\s*$", "", t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\b(?:mi|miles?|nautical|approx|near|off|of|the|n|s|e|w|ne|nw|se|sw|nne|nnw|ene|ese|sse|ssw|wsw|wnw)\b", " ", t)
    t = re.sub(r"\b\d+(?:\s*\d+)*\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------------------
# 해역(넓은 구역) 대응표. 장소 문구의 낱말로 정한다. 좌표는 부호가 뒤집히거나 위도·경도가 바뀐 것이 있어
# (예: 노르웨이 Andenes 가 서경으로, McMurdo 의 Castle Rock 이 서경으로 적혀 있다) 거리 계산의 근거로 쓰지 않는다.
# 같은 구역·같은 해의 녹음은 같은 원정으로 묶는다. 위에서부터 먼저 맞는 규칙을 쓴다.
# 이 표는 구현에서 정한 것이다. 연구자가 고칠 수 있고, 고치면 b01 → b02 를 다시 돌린다 (잠금 시험셋이 바뀐다).
# --------------------------------------------------------------------------------------
REGION_RULES = [
    ("STLAW", r"saguenay|baie ste|ile aux coudres|gulf of st\. lawrence|st\. lawrence river"),
    ("BERING", r"gambell|st\. lawrence i|bering strait"),
    ("BEAUFORT", r"barrow|beaufort|bailie"),
    ("CAN_ARCTIC", r"baffin|pond inlet|ellesmere|grise fjord|stratton|cumins|koluktoo|t3 ice island"),
    ("ANTARCTIC_ROSS", r"antarctica|mcmurdo|ross sea|cape adare|cape adair|cape hallett|cape jones|coulman|cape crozier|razor back|castle rock"),
    ("CARIBBEAN", r"dominica|st\. lucia|bequia|st\. vincent|tortola|virgin islands|puerto rico|san juan"),
    ("BAHAMAS_FLORIDA", r"bimini|bahamas|tongue of the ocean|biscayne|sarasota|silver springs|fla\.|florida"),
    ("BERMUDA", r"bermuda|argus island|plantagenet|gibbs hill"),
    ("MEDITERRANEAN", r"sicily|malta|alicudi|ustica|selina|panarea|capraia|san remo|sardinia|capo caccia|losinij|yugoslavia|gibraltar|italy"),
    ("CANARY_MOROCCO", r"canary|lanzarote|isla de lobo|morocco"),
    ("NORWAY", r"norway|andenes|kikvika"),
    ("HAWAII", r"hawaii|maui|kealakekua|olowalu"),
    ("BAJA", r"magdalena bay|baja"),
    ("OREGON", r"oregon"),
    ("CALIFORNIA", r"california|calif|catalina|san pedro|santa cruz island|point loma|san diego|los angeles"),
    ("NE_PACIFIC", r"vancouver|british columbia|saanich|dabob|seattle|washington"),
    ("GULF_OF_ALASKA", r"prince william sound|knight island"),
    ("NEW_ZEALAND", r"new zealand|kaikoura"),
    ("AUSTRALIA", r"australia|warrnambool|warmwarnbool"),
    ("ARGENTINA", r"argentina|golfo san jose"),
    ("CHILE", r"magellan|juan fernandez|chile"),
    ("COSTA_RICA", r"costa rica"),
    ("EAST_AFRICA", r"kenya"),
    ("SOUTH_AFRICA", r"south africa|patanostra"),
    ("JAPAN", r"japan"),
    ("NY_FACILITY", r"new york aquarium|bronx zoo|niagra falls|johns hopkins"),
    ("NW_ATLANTIC", r"cape cod|stellwagen|race point|provincetown|georges bank|browns bank|baccaro|sable i|gully|nova scotia|"
                    r"halifax|vineyard|menemsha|nauset|martha|maine|cape porpoise|fundy|hudson canyon|block canyon|delaware|"
                    r"north atlantic|newfoundland|trinity bay|st\. john|sasuit"),
]
_REGION_RE = [(name, re.compile(pat, re.I)) for name, pat in REGION_RULES]
# 장소 문구가 빈 컷은 좌표 칸으로 구역을 정한다 (빈 문구의 컷이 놓인 칸만 적었다).
REGION_BY_CELL = {
    "15,-61": "CARIBBEAN", "49,-67": "STLAW",
    "38,-64": "NW_ATLANTIC", "38,-71": "NW_ATLANTIC", "38,-72": "NW_ATLANTIC", "38,-73": "NW_ATLANTIC",
    "38,-74": "NW_ATLANTIC", "39,-71": "NW_ATLANTIC", "39,-72": "NW_ATLANTIC", "40,-71": "NW_ATLANTIC",
    "42,-61": "NW_ATLANTIC", "42,-67": "NW_ATLANTIC", "42,-70": "NW_ATLANTIC",
}


def region_of(location: str, cell: str) -> str:
    for name, rx in _REGION_RE:
        if rx.search(location or ""):
            return name
    if not (location or "").strip():
        return REGION_BY_CELL.get(cell, "")
    return ""


def stable_key(*parts) -> str:
    """고정 씨앗과 값으로 만든 해시. 표의 행 순서가 바뀌어도 같은 순서를 준다."""
    return hashlib.sha256("|".join(str(p) for p in (SEED,) + parts).encode("utf-8")).hexdigest()


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while self.p[x] != root:
            self.p[x], x = root, self.p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra


# --------------------------------------------------------------------------------------
# 음원
# --------------------------------------------------------------------------------------
AUDIO_EXT = (".flac", ".wav", ".wave", ".aif", ".aiff")
# 한 컷에서 분석에 읽는 길이의 상한(초). 핵심 표본은 60초 이하라 해당이 없다. 전체 13,693개 가운데 56개가 이보다 길고
# (가장 긴 것은 1,260초, 샘플 9천만 개), 통째로 읽으면 한 파일에 메모리가 몇 GB 든다. 구현에서 고른 값이다.
MAX_ANALYSIS_S = 120.0
_ID_RE = re.compile(r"^[0-9]{5}[0-9A-Za-z]{3}$")


def index_audio(audio_root: str | os.PathLike) -> dict[str, str]:
    """audio_root 아래(하위 폴더 포함)의 음원을 컷 번호 → 경로로 모은다. 읽기만 한다."""
    root = Path(audio_root)
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for parent, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            stem, ext = os.path.splitext(name)
            if ext.lower() in AUDIO_EXT and _ID_RE.match(stem):
                out.setdefault(stem, os.path.join(parent, name))
    return out


def read_audio(path: str, max_s: float | None = None):
    """(샘플 float64 1차원, fs) 를 돌려준다. 다채널이면 첫 채널. 리샘플하지 않는다.

    max_s 를 주면 앞에서부터 그 길이(초)까지만 읽는다.
    soundfile 이 있으면 그것으로, 없으면 ffmpeg 로 읽는다.
    """
    import numpy as np
    try:
        import soundfile as sf
    except ImportError:
        sf = None
    if sf is not None:
        frames = -1
        if max_s:
            info = sf.info(path)
            frames = min(int(info.frames), int(round(max_s * info.samplerate)))
        x, fs = sf.read(path, frames=frames, dtype="float64", always_2d=True)
        return x[:, 0].copy(), int(fs)
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                         "stream=sample_rate", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    fs = int(pr.stdout.strip().split("\n")[0])
    cut = ["-t", f"{max_s:.6f}"] if max_s else []
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, *cut, "-map", "0:a:0", "-ac", "1", "-f", "f64le",
                          "-acodec", "pcm_f64le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f8").copy(), fs


def read_audio_int16(path: str):
    """정수 샘플(int16, 첫 채널)과 fs. 양자화 간격·중복 검사용."""
    import soundfile as sf
    x, fs = sf.read(path, dtype="int16", always_2d=True)
    return x[:, 0].copy(), int(fs)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def log(*a) -> None:
    print(*a, flush=True)


if __name__ == "__main__":
    cfg = load_config()
    log(VERSION, "| python", sys.version.split()[0])
    log("audio_root:", cfg["audio_root"], "| exists:", Path(cfg["audio_root"]).is_dir())
    log("work_root :", cfg["work_root"])
    log("종", len(SPECIES), "| 분류군별:", {g: sum(1 for v in SPECIES.values() if v[0] == g) for g in GROUP_KO})
