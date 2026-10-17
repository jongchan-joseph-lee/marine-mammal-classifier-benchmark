# -*- coding: utf-8 -*-
"""Watkins 노트(NT)에서 '어떤 소리가 들어 있다고 적혀 있는가'를 읽어 낸다. 음원은 읽지 않는다.

왜 다시 만드는가
  공개본(Earth Species Project v0.1.0)의 call_type 태그는 노트에서 낱말을 뽑은 것이다(18,740개 전부가 같은 컷의 노트에 있는 낱말).
  낱말만 뽑으면 "ship creaks", "Buzzing ice sounds", "Sp. wh. clicks"(향고래), "Cut made for odontocete click database",
  "Calls- 1 upsweep"(머리말의 Calls) 같은 문구에서도 태그가 생긴다. 이 모듈은 노트를 구절과 토막으로 나눠,
  소리 낱말마다 '누가 낸 소리로 적혀 있는가'를 함께 읽는다.

두 층으로 나눈다
  1) 낱말 층 (노트 그대로): 토막마다 소리 낱말(term), 꾸밈말(pulsed, tonal ...), 귀속을 적는다.
       귀속: target(대상 동물) / implied(문구가 암시) / other:<종> / nonanimal:<소리원> / unknown_origin(출처를 알 수 없는 충격음)
             / neg(없다고 적음) / tape_log(테이프 전체의 일지라 이 컷의 내용인지 알 수 없음)
  2) 형태 층 (해석): 낱말 → 형태 범주. data/note_term_map.csv 의 대응표로 정한다. 연구자가 고칠 수 있다.

노트의 약속 (자료에서 읽어 낸 것)
  - 맨 앞의 부호(BE7A, AC2A ...)는 종의 분류 부호다. 부호가 둘 이상이면 그 컷에 다른 종도 있다는 뜻이다. BA2A 는 향고래다.
  - 부호 뒤의 X 는 '다른 소리가 섞여 있다'는 표시로 보인다(X 가 있는 노트의 97 % 에 잡음 낱말이 있다). 문장 중간의 X 는 그 뒤가 그 소리의 설명이다.
  - PU, NL, NS, BS, BL 은 신호 유형의 줄임말이다: PU 펄스성, NL 좁고 긴(narrow long), NS 좁고 짧은, BS 넓고 짧은(broad short), BL 넓고 긴.
    ("Clang (Broad short)", "Unknown broad short X", "PU calls had a component with a pulse repetition rate ..." 같은 문구에서 확인)

한계
  노트는 컷 전체에 대한 자유 서술이다. 소리가 '언제' 나는지, 어느 것이 주된 소리인지는 대개 적혀 있지 않다.
  같은 노트가 한 테이프의 여러 컷에 되풀이된 것도 있다. 그래서 이 결과는 여전히 약한 라벨이다. 청취로 확인한 정답이 아니다.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------------------
# 0. 종 부호. 부호 → 종은 자료에서 센 것이다(그 종의 컷에서 가장 흔한 부호).
# --------------------------------------------------------------------------------------------------
CODE_RE = re.compile(r"\b([A-D][A-G]\d{1,2}[A-Z])\b")
OWN_CODES = {
    "Balaena mysticetus": {"AA1A"}, "Eubalaena glacialis": {"AA3A"}, "Eubalaena australis": {"AA3B"},
    "Eschrichtius robustus": {"AB1A"}, "Balaenoptera acutorostrata": {"AC1A"}, "Balaenoptera musculus": {"AC1E"},
    "Balaenoptera physalus": {"AC1F"}, "Megaptera novaeangliae": {"AC2A"},
    "Delphinapterus leucas": {"BB1A"}, "Monodon monoceros": {"BB2A"},
    "Cephalorhynchus commersonii": {"BD1A"}, "Cephalorhynchus heavisidii": {"BD1C"},
    "Delphinus bairdii": {"BD3A"}, "Delphinus delphis": {"BD3B"}, "Grampus griseus": {"BD4A"},
    "Lagenodelphis hosei": {"BD5A"}, "Lagenorhynchus acutus": {"BD6A"}, "Lagenorhynchus albirostris": {"BD6B"},
    "Peponocephala electra": {"BD10A"}, "Sousa sahulensis": {"BD12A"}, "Sotalia fluviatilis": {"BD12B"},
    "Stenella attenuata": {"BD15A"}, "Stenella clymene": {"BD15B"}, "Stenella coeruleoalba": {"BD15C"},
    "Stenella frontalis": {"BD15F"}, "Stenella longirostris": {"BD15L"}, "Steno bredanensis": {"BD17A"},
    "Tursiops truncatus": {"BD19D"}, "Globicephala macrorhynchus": {"BE3B", "BE3D"}, "Globicephala melas": {"BE3C"},
    "Orcinus orca": {"BE7A"}, "Pseudorca crassidens": {"BE9A"}, "Inia geoffrensis": {"BG2A"},
    "Arctocephalus forsteri": {"CA1F"}, "Arctocephalus philippii": {"CA1P"}, "Eumetopias jubatus": {"CA3B"},
    "Odobenus rosmarus": {"CB1A"}, "Cystophora cristata": {"CC1A"}, "Erignathus barbatus": {"CC2A"},
    "Halichoerus grypus": {"CC3A"}, "Hydrurga leptonyx": {"CC4A"}, "Leptonychotes weddellii": {"CC5A"},
    "Histriophoca fasciata": {"CC12F"}, "Pagophilus groenlandicus": {"CC12G"}, "Pusa hispida": {"CC12H"},
    "Phoca largha": {"CC12L", "CC12V"}, "Ommatophoca rossi": {"CC14A"}, "Enhydra lutris": {"CD1A"},
}
CODE_NAME = {c: sp for sp, cs in OWN_CODES.items() for c in cs}
CODE_NAME["BA2A"] = "sperm whale"

# --------------------------------------------------------------------------------------------------
# 1. 지우는 문구. 소리의 내용이 아니라 테이프·디지털화·분석 기록이다. 소리 낱말이 들어 있는 것이 있어 먼저 지운다.
#    이름이 implied_* 인 것은 지우되 '그런 소리가 있다고 암시한다'는 표시를 남긴다.
# --------------------------------------------------------------------------------------------------
BOILER = [
    ("implied_click", r"Cuts? made for (?:the )?odontocete click database\.?"),
    ("tech", r"Historc?ally throughout all these Keiko cuts.*$"),
    ("tech", r"Tape (?:digitized|restreamed|streamed)[^.]*?(?:filters?|DVD)\b[^.]*\.?"),
    ("tech", r"\band stored on Buffalo terastation and DVD\.?"),
    ("tech", r"Tape (?:digitized|restreamed|streamed) at[^.]*\."),
    ("tech", r"Tape restreamed (?:for|from)[^.]*\."),
    ("tech", r"Cut made from RN\s*#?\s*\w+,? and cropped for (?:use|analysis) in Acc?o?ustat\.?"),
    ("tech", r"Cut made from (?:RN|CD)\s*#?\s*\w+(?:, channel \d)?\.?"),
    ("tech", r"\bcropped for (?:use|analysis) in Acc?o?ustat\.?"),
    ("tech", r"Cuts? made for [A-Z][a-z]+ [a-z]+ analysis\.?"),
    ("tech", r"Cut made for Vincent Janik looking for chorusing behavior\.?"),
    ("tech", r"(?:Long )?[Cc]uts? made (?:for|of|from digitized tape for) ['\"]Best of[^'\"]*['\"](?: CD)?(?: series)?\.?"),
    ("tech", r"From streamed file \S+(?:\s*\([^)]*\)?)?\.?"),
    ("tech", r"\((?:Cue|cue) [\d\s-]+\)\.?"),
    ("tech", r"\bcue \d+-\d+\)\.?"),
    ("tech", r"Started stream file at cue \d+\.?"),
    ("tech", r"Saved as streamed tape \w+ on Exabyte\.?"),
    ("tech", r"Cue times are approximate--original file on CD restreamed with HP filter of [\d.]+ ?kHz and LP filter of [\d.]+ ?kHz\.?"),
    ("tech", r"Restreamed channel \d for this cut\.?"),
    ("tech", r"Channels? \d(?: and \d)? streamed @[^.]*\.(?:[^.]*demultiplexed[^.]*\.)?"),
    ("tech", r"All BU were streamed at twenty minute intervals\.?"),
    ("tech", r"Track \d+ from 'Sounds of Norwegian Killer Whales'(?:[^.]*?Norway\.?)?"),
    ("tech", r"OCA\s+Tape\s*#?\s*\d+\.?"),
    ("tech", r"Hydrophones recorded on HiFi L & R\.?"),
    ("tech", r"SMPTE code on Norm\.?"),
    ("tech", r"No SMPTE\s*\.?"),
    ("tech", r"Time Code (?:not )?available on tape #?\d+(?:, but probably not reliable)?\.?"),
    ("tech", r"Time Code reset often during Tape \d+\.?"),
    ("tech", r"Radio station occurs [^.]*?HiFi Chs\.?"),
    ("tech", r"During the recordings on tape \d+ there was no natural progression from day to night, and darkness occurred at [\d:]+\.?"),
    ("tech", r"There was natural progression from day to night on tape #?\d+, with darkness at [\d:]+\.?"),
    ("tech", r"\d+ minute cassette given by Hanne Strager for Acoustat run\.?"),
    ("tech", r"The killer whales were quite spread out and were involved in social activity only\.?"),
    ("tech", r"No feeding or unidirectional travel was observed\.?"),
    ("tech", r"The recording is a mixture of pods P and K, photos also were taken\.?"),
    ("tech", r"Underwater biological sound recordings from the Navy Electronics Laboratory \(summer 1967\)\.?"),
    ("tech", r"(?:Watkins'? )?[Cc]opy of (?:C\. )?(?:Ray|Johnson)'s[^.]*\.(?: last part not copied\.)?"),
    ("tech", r"C\. Ray's tape NYZS\s*#\s*\d+\.?"),
    ("tech", r"Ray's \d{4}-\d+ tape\.?"),
    ("tech", r"Ray's Flight \d+,? Tape #?\d+\.?"),
    ("tech", r"Flight \d+,? Tape #?\s*\d+\.?"),
    ("tech", r"Copy of original (?:SOFAR recordings|tape)\.?"),
    ("tech", r"\b(?:\d of \d )?[Rr]ecuts? of \w+\.?"),
    ("tech", r"\b(?:Part of|Taken from|Smaller cut from) (?:cut |RN )?\d{5}\w{3}\.?"),
    ("tech", r"Similar to \w{2,3}\.?"),
    ("tech", r"\b[NS]\s?\d{1,2}[\s'.°]+\d{0,2}(?:\.\d+)?['\"]?\s*[,;]?\s*[WE]\s?\d{1,3}[\s'.°]*\d{0,2}(?:\.\d+)?['\"]?\.?"),
    ("tech", r"\b(?:BS|BT|ES|ET)\s*=\s*[\d.]+,?"),
    ("tech", r"\bS/N\s*=\s*\d+"),
    ("tech", r"\bSo-so signal-\s?noise\.?"),
    ("tech", r"\bCh\.? ?1\s*=\s*voice, [Cc]h\.? ?2\s*=\s*(?:data|[Mm]usic)\.?"),
    ("tech", r"\bSIDE [AB]\b|\bSide \d:"),
    ("tech", r"\b(?:Recording )?Session (?:\d+|IV|I{1,3})\b\s*[:.]?"),
    ("tech", r"\bTape\s*#\s*\d+\s*(?::|--)?"),
    ("tech", r"\bPart I{1,2} of tape\.?"),
    ("tech", r"Upper limit of recording equipment was ~?\d+ ?[Kk][Hh]z\.?"),
    ("tech", r"Hamming Window;?|FFT \d+;?|Time Axis [\d.]+ sec\.?;?"),
    ("tech", r"(?:\d(?:st|nd|rd|th) harmonic(?: signal length)? [\d.]+ ?(?:Hz|sec\.?);?)"),
    ("tech", r"Strong emphasis of \d(?:st|nd|rd|th) harmonic\.?"),
    ("tech", r"Fund\.? ?(?:freq\.?)? ?[\d.]+ ?Hz;?"),
    ("tech", r"\b(?:FF|FE|FR)\s*(?:\([^)]*\)\s*)?=?\s*[\d.]+\s*k?(?:Hz)?\.?"),
    ("tech", r"\b(?:Frequency|Freq\.?) emphasis\s*=?\s*[\d.]+\s*Hz"),
    ("tech", r"Wh\. freq\. range [\d\s-]+Hz"),
    ("tech", r"\b(?:a )?(?:pulse repetition rate|pulse rate|PRR)\b[^.;)]*"),
    ("tech", r"\bboomer ball\b"),
    ("tech", r"Photos on bow at \d+\.?"),
]
BOILER_RE = [(name, re.compile(pat, re.I)) for name, pat in BOILER]

# 마침표가 붙지만 문장의 끝이 아닌 줄임말과 흔한 오타. 구절을 나누기 전에 바로잡는다.
ABBREV = [
    (r"\bfreq\.", "freq"), (r"\bapprox\.", "approx"), (r"\bb(?:ac|c)?k?g?r?nd\.(?=\s+[A-Z])", "background."), (r"\bb(?:ac|c)?k?g?r?nd\.?(?=\W|$)", "background"),
    (r"\bbackgnd\b", "background"), (r"\bforegrnd\.(?=\s+[A-Z])", "foreground."), (r"\bforegrnd\.?", "foreground"), (r"\bsp\. ?wh(?:ale)?\.?(?=\W)", "sperm whale"), (r"\bsperm ww+hales?\b", "sperm whale"),
    (r"\bwh\.", "whistle"), (r"\bincl\.", "including"), (r"\betc\.", "etc"), (r"\bmin\.", "min"), (r"\bsec\.", "sec"), (r"\bmi\.", "mi"),
    (r"\bft\.", "ft"), (r"\bca\.", "ca"), (r"\bvs\.", "vs"), (r"\bno\. ?(?=\d)", "number "), (r"\bst\. ?(?=[A-Z])", "St "),
    (r"\bI\.D\.'d", "identified"), (r"\b([A-Z])\. (?=[a-z]{4,})", r"\1 "), (r"\bC\. (?=Ray)", "C "), (r"\bH2O\b", "water"),
    (r"\bw/", "with "), (r"\b[Ii]n\s*-\s*air\b", "in-air"), (r"\bpusle\b", "pulse"), (r"\bscr(?:each|eetch)", "screech"),
    (r"\bsqeak", "squeak"), (r"\bwhis(?:tel|lt|it)l?e?y\b", "whistley"), (r"\bcliks\b", "clicks"), (r"\bpuled\b", "pulsed"),
]
ABBREV_RE = [(re.compile(p, re.I), r) for p, r in ABBREV]
# 컷이 아니라 테이프·녹음 전체를 요약한 글투. 표시만 한다(라벨은 그대로 둔다).
TAPE_SUMMARY_RE = re.compile(r"(?<!Hz )\bthroughout (?:much of )?(?:the )?tape\b|\b(?:for )?most of (?:the )?tape\b|\bon (?:the )?tape as well\b"
                             r"|\b\d+ minute (?:cassette|recording)\b|\bhalf way through\b|\b(?:beginning|end) of (?:the )?tape\b", re.I)
DATE_RE = re.compile(r"\b\d{1,2}[- ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:[- ]\d{4})?\b", re.I)


# 신호 유형 줄임말을 늘어놓고 끝에 낱말을 한 번만 쓴 꼴: "combo of BL, FM, PU calls", "NL and PU call"
SIGLIST_RE = re.compile(r"\b((?:(?:BL|BS|NL|NS|PU|FM)\s*(?:,|\band\b|&)\s*)+)(BL|BS|NL|NS|PU|FM)\s+(calls?|clicks?)\b")


def _expand_siglist(mo) -> str:
    kinds = re.findall(r"BL|BS|NL|NS|PU|FM", mo.group(1)) + [mo.group(2)]
    return ", ".join(f"{k} {mo.group(3)}" for k in kinds)


@dataclass
class Item:
    """노트의 가장 작은 단위(쉼표·and 로 나뉜 한 토막)."""
    text: str
    clause: int
    codes: tuple = ()            # 이 토막에 걸려 있는 종 부호
    heading: str = ""            # 'Pilot whales--', 'Ship noise--', 'Calls-' 같은 머리말
    extraneous: bool = False     # 문장 중간의 X 표시 바로 뒤 구절 (다른 소리의 설명)


@dataclass
class Cleaned:
    text: str
    codes: list = field(default_factory=list)       # 노트에 나온 종 부호(순서대로, 겹침 없이)
    x_mark: bool = False                            # X 표시가 있는가
    u_mark: bool = False                            # U 표시(알 수 없는 소리)가 있는가
    implied: list = field(default_factory=list)     # 지운 문구가 암시하는 소리 (예: click)
    tape_log: bool = False                          # 날짜가 둘 이상 적힌 테이프 일지
    tape_summary: bool = False                      # 테이프·녹음 전체를 요약한 글투가 있다


def clean_note(note: str) -> Cleaned:
    """부호·표시·기록 문구를 떼고 소리 서술만 남긴다."""
    t = re.sub(r"\s+", " ", note or "").strip()
    out = Cleaned(text="")
    for c in CODE_RE.findall(t):
        if c not in out.codes:
            out.codes.append(c)
    lead = re.match(r"^\s*(?:Ice\s+)?(?:[A-D][A-G]\d{1,2}[A-Z]\b[\s,;/]*)*((?:[OUV]\s*)?V?X\b|U\b)?", t)
    mark = (lead.group(1) or "") if lead else ""
    out.x_mark = "X" in mark or bool(re.search(r"\sX\s+(?=[A-Z])|[a-z)]\s+X$", t))
    out.u_mark = mark.startswith("U")
    out.tape_log = len(DATE_RE.findall(t)) >= 2
    out.tape_summary = bool(TAPE_SUMMARY_RE.search(t))
    for name, rx in BOILER_RE:
        def _sub(mo, name=name):
            if name.startswith("implied_"):
                out.implied.append(name.split("_", 1)[1])
            return " "
        t = rx.sub(_sub, t)
    for rx, rep in ABBREV_RE:
        t = rx.sub(rep, t)
    t = SIGLIST_RE.sub(_expand_siglist, t)
    out.text = re.sub(r"\s+", " ", t).strip()
    return out


_GLUE = re.compile(r"[\s,;/]*(?:[OUV]\s*)?(?:V?X\b)?[\s,;/]*")


def _segments(text: str) -> list[tuple[tuple, str]]:
    """종 부호의 위치로 구간을 나눈다. 연달아 붙은 부호는 한 구간에 함께 걸린다. (부호 묶음, 글) 의 목록."""
    toks = list(CODE_RE.finditer(text))
    segs, cur_codes, cur_start, i = [], [], 0, 0
    while i < len(toks):
        j, group = i, [toks[i].group(1)]
        while j + 1 < len(toks) and _GLUE.fullmatch(text[toks[j].end():toks[j + 1].start()] or ""):
            j += 1
            group.append(toks[j].group(1))
        before = text[cur_start:toks[i].start()]
        if before.strip(" ();,.-"):
            segs.append((tuple(cur_codes), before))
        cur_codes, cur_start, i = group, toks[j].end(), j + 1
    tail = text[cur_start:]
    if tail.strip() or not segs:
        segs.append((tuple(cur_codes), tail))
    return segs


CLAUSE_SPLIT = re.compile(r"(?<=[^\d])[.;!?]+(?=\s|$)|(?<=\d)[.;!?]+(?=\s+[A-Za-z'(§]|$)")
HEADING = re.compile(r"^([A-Za-z][A-Za-z '?.]{2,45}?)\s*(?:--|-\s+(?=\d|[a-z])|:\s)\s*(.+)$")
ITEM_SPLIT = re.compile(r"\s*,\s*|\s+and\s+|\s*&\s*|\s+\+\s+|\s*\+\s*(?=[a-z])|\s+with\s+|\s+then\s+|\s+followed by\s+|\s+plus\s+|\s+but\s+|\s+or\s+"
                        r"|\s+into\s+|\s+changing to\s+|\s+leading into\s+|\s+fading to\s+")


def split_items(cl: Cleaned) -> list[Item]:
    """부호로 구간을 나누고, 구절(. ; ! ? 와 X 표시)로, 다시 토막(쉼표, and, +, with ...)으로 나눈다."""
    items: list[Item] = []
    k = 0
    for codes, seg in _segments(cl.text):
        seg = re.sub(r"^\s*(?:[OUV]\s+)?V?X\b", " ", seg)                # 부호 바로 뒤의 X 표시
        seg = re.sub(r"^\s*[UO]\s+(?=[A-Z])", " ", seg)                   # 부호 바로 뒤의 U, O 표시
        seg = re.sub(r"\s+X\s+(?=[A-Z])", " . §X ", seg)                   # 문장 중간의 X: 그 뒤는 다른 소리의 설명
        seg = re.sub(r"(?<=[a-z)])\s+X\s*$", " ", seg)
        for clause in CLAUSE_SPLIT.split(seg):
            clause = (clause or "").strip(" .;:-")
            if not clause:
                continue
            extraneous = clause.startswith("§X")
            clause = clause.replace("§X", "").strip(" .;:-")
            if not clause:
                continue
            heading = ""
            mh = HEADING.match(clause)
            if mh:
                heading, clause = mh.group(1).strip(), mh.group(2).strip()
            for part in ITEM_SPLIT.split(clause):
                part = (part or "").strip(" .;:-")
                if part:
                    items.append(Item(text=part, clause=k, codes=codes, heading=heading, extraneous=extraneous))
            k += 1
    return items


# --------------------------------------------------------------------------------------------------
# 2. 소리 낱말 (낱말 층). 노트에 적힌 낱말을 표준 이름으로 모을 뿐, 형태를 해석하지 않는다.
#    먼저 묶음말(두 낱말 이상)을 찾고, 남은 글에서 낱말 하나짜리를 찾는다.
# --------------------------------------------------------------------------------------------------
COMPOUND_TERMS = [
    ("click_burst", r"(?:pulsed?\s+)?clicks?\s+(?:pulsed\s+)?bursts?(?:\s+series)?|bursts?\s+(?:of\s+)?(?:pulsed\s+)?clicks?"),
    ("click_series", r"(?:pulsed\s+)?clicks?\s+(?:series|trains?|sequences?)(?!\s+of)|(?:series|trains?|sequences?)\s+of\s+(?:\w+\s+){0,2}?(?:pulsed\s+)?clicks?"),
    ("pulse_burst", r"pulsed?\s+bursts?|bursts?[- ]pulsed?|bursts?\s+pulses?"),
    ("pulse_series", r"pulsed?\s+(?:series|trains?)|(?:series|trains?)\s+of\s+(?:\w+\s+){0,2}?pulses"),
    ("pulsed_click", r"pulsed?\s+clicks?|clicks?\s*-\s*pulsed|pulsed\s+clicking"),
    ("echolocation", r"echolocation(?:[- ](?:type|like))?(?:\s+clicks?)?"),
    ("call_squeal", r"calls?\s*/\s*sque(?:al|ak)s?"),
    ("social_sound", r"social\s+sounds?|social(?=\s*$)"),
    ("song", r"song\s+(?:calls?|elements?|units?|notes?|phrases?)|(?:parts?|pieces?|portions?|sections?|end|beginning|phrases? \w+)\s+of\s+(?:[\w']+\s+){0,2}songs?"),
    ("knock_series", r"knocks?\s+series|(?:series|sets?)\s+of\s+knocks"),
]
SINGLE_TERMS = [
    ("click", r"clicks?|clicking|clickers?"),
    ("buzz", r"buzz(?:es|ing)?"),
    ("creak", r"creaks?|creaking"),
    ("pulse", r"pulses?|pulsing"),
    ("burst", r"bursts?"),
    ("knock", r"knocks?|knocking"),
    ("whistle", r"whistles?|whistling|whistlers?"),
    ("chirp", r"chirps?|chirping"),
    ("upsweep", r"up-?\s?sweeps?|upcalls?"),
    ("downsweep", r"down-?\s?sweeps?|downweeps?"),
    ("sweep", r"sweeps?"),
    ("trill", r"trills?"),
    ("warble", r"warbles?|warbling"),
    ("tone", r"tonals|tones?"),
    ("whine", r"whines?|whining"),
    ("moan", r"moans?|moaning"),
    ("groan", r"groans?|groaning|grown"),
    ("moo", r"moos?"),
    ("bell", r"bells?(?!\s+buoy)"),
    ("squeal", r"squeals?|squealing"),
    ("squeak", r"squeaks?|squeaking"),
    ("squawk", r"squawks?"),
    ("scream", r"screams?|screaming"),
    ("shriek", r"shrieks?"),
    ("screech", r"screech(?:es|ing|in)?"),
    ("trumpet", r"trumpets?|trumpeting"),
    ("cry", r"cries|cry"),
    ("honk", r"honks?"),
    ("hoot", r"hoots?|hooting"),
    ("call", r"calls?"),
    ("growl", r"growls?|growling"),
    ("grunt", r"grunts?|grunting"),
    ("grumble", r"grumbles?|grumbling"),
    ("roar", r"roars?|roaring"),
    ("bark", r"barks?|barking"),
    ("snort", r"snorts?"),
    ("wheeze", r"wheezes?|wheezing"),
    ("hiss", r"hiss(?:es|ing)?"),
    ("blow", r"blows?"),
    ("chug", r"chugs?|chugging"),
    ("drumming", r"drumm?ings?|drums?"),
    ("pop", r"pops?"),
    ("crack", r"cracks?"),
    ("bang", r"bangs?|banging"),
    ("thump", r"thumps?|thuds?"),
    ("slap", r"slaps?|slapping|lobtail"),
    ("clap", r"claps?"),
    ("snap", r"snaps?"),
    ("rap", r"raps?"),
    ("flap", r"flaps?"),
    ("rattle", r"rattles?"),
    ("song", r"songs?|singing|sings?"),
    ("phrase", r"phrases?|phrasing"),
    ("coda", r"codas?"),
    # 동물 소리가 아닌 것을 가리키는 낱말. 귀속을 따지지 않고 '잡음'으로 둔다.
    ("noise", r"noises?|noice|interference|static|hum|crackle|clangs?|clanks?|clunks?|crunch(?:es)?|crash(?:ing)?|bumps?|bumping|pings?|splash(?:es)?"
              r"|whip|rushing|droplets?"),
    # 형태를 말하지 않는 일반 낱말
    ("sound", r"sounds?|signals?|vocali[sz]ations?|phonations?"),
]
_COMPOUND_RE = [(n, re.compile(r"\b(?:" + p + r")\b", re.I)) for n, p in COMPOUND_TERMS]
_SINGLE_RE = [(n, re.compile(r"\b(?:" + p + r")\b", re.I)) for n, p in SINGLE_TERMS]
GENERIC = {"sound", "call", "social_sound"}                 # 형태를 말하지 않는 낱말
SWEEPS = ("upsweep", "downsweep", "sweep")
WEAK_HEADS = {"sound", "burst", "phrase"}                    # 같은 토막에 더 구체적인 낱말이 있으면 버리는 것
# 충격음: 동물이 낸 것인지 노트만으로는 알 수 없는 낱말. 따로 근거가 없으면 unknown_origin 으로 둔다.
AMBIG = {"knock", "knock_series", "pop", "crack", "bang", "thump", "slap", "clap", "snap", "rap", "flap", "rattle"}
BODY_RE = re.compile(r"\bjaws?\b|\bbaleen\b|\bflipper\b|\btail\b|\blobtail\b|\bfluke\b", re.I)
# 그 종의 발성으로 알려진 충격음 (노트의 쓰임에서 확인: 귀신고래 'Knocks', '3 sets of knocks')
AMBIG_OWN = {("knock", "Eschrichtius robustus"), ("knock_series", "Eschrichtius robustus"), ("knock", "Odobenus rosmarus"),
             ("knock_series", "Odobenus rosmarus")}

MODIFIERS = [
    ("pulsed", r"\bpulsed?\b(?!\s+rate)|-pulsed\b"),
    ("burst", r"\bbursts?\b"),
    ("series", r"\bseries\b|\btrains?\b|\bsequences?\b"),
    ("fast", r"\brapid\b|\bfast\b|\bquick\b"),
    ("buzzy", r"\bbuzzy\b"),
    ("tonal", r"\btonal\b|\bnarrow(?:band)?\b|\bpure\b"),
    ("fm", r"(?<!non-)\bfm\b|(?<!non-)\bfrequency[- ]modulat\w+|(?<!non-)(?<!non-frequency )\bmodulat\w+|\bwavery?\b|\bwavered\b"
           r"|(?<!non-)\boscillating\b"),
    ("up", r"\bupsweep\w*|\bupswept\b|\bascending\b|\brising\b|\bupward\b|\bupstepped\b|\buplooped\b"),
    ("down", r"\bdownsweep\w*|\bdownswept\b|\bdescending\b|\bfalling\b|\bdownward\b|\bdownlooped\b"),
    ("whistley", r"\bwhistle?y\b|\bwhistle-like\b"),
    ("chirpy", r"\bchirpy\b|\bchirp[- ]like\b"),
    ("looped", r"\b(?:\w+-)?loop(?:ed|s|y)?\b|\bu-shaped\b"),
    ("stepped", r"\b(?:\w+-)?step(?:ped|s)?\b"),
    ("long", r"\blong\b"),
    ("short", r"\bshort\b"),
    ("low", r"\blow(?:er)?(?:[- ]freq\w*)?\b(?!\s+level)"),
    ("high", r"\bhigh(?:er)?(?:[- ](?:freq\w*|pitch(?:ed)?))?\b(?!\s+level)"),
]
_MOD_RE = [(n, re.compile(p, re.I)) for n, p in MODIFIERS]
# Watkins 의 신호 유형 줄임말 (대문자일 때만)
SIGTYPE_RE = [("pulsed", re.compile(r"\bPU\b")), ("tonal", re.compile(r"\bN[LS]\b")), ("broad", re.compile(r"\bB[SL]\b|\b[Bb]road\b"))]
BG_PRE = re.compile(r"\b(?:background|faint(?:er|ly)?|weak|distant|far|soft|quiet|low[- ]level|barely)\b", re.I)
BG_POST = re.compile(r"^(?:\s+\S+){0,4}?\s+(?:in|at)\s+(?:the\s+|a\s+)?(?:background|back|distance)\b|^\s*\((?:distant|faint)", re.I)
NEG_RE = re.compile(r"\b(?:no|not|cannot|can't|isn't|aren't|without|none|never|absence of|lack of)\b(?!\s+(?:many|much|only|all|too|animals?|visual|visible|trainers?))", re.I)
AIR_RE = re.compile(r"\bin-air\b|\bin air\b|\bair\b(?!plane)", re.I)
COUNT_RE = re.compile(r"^(?:~|about |approx(?:imately)? )?(\d+(?:\s*-\s*\d+)?(?![\d.]*\s*(?:k?hz|sec|min|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))|one|two|three|four|five|six|seven|eight|nine|ten|dozen|several|many|numerous|"
                      r"multiple|few|a few|some|a lot of|lots of|a series of|series of|pair of|a|an)\b", re.I)
SIMILE_PRE = re.compile(r"(?:\b(?:sound(?:s|ing|ed)?|resembl\w+)\s+like\s+(?:an?\s+|the\s+)?|\blike\s+(?:an?|the)\s+)(?:[\w']+\s+){0,2}$", re.I)
XLIKE_POST = re.compile(r"^[- ]like\b(?!\s+(?:an?|the)\b)", re.I)      # "chirp like call", "whistle-like": 앞 낱말은 꾸밈말
XLIKE_MOD = {"chirp": "chirpy", "whistle": "whistley", "buzz": "buzzy", "pulse": "pulsed"}
UNKNOWN_PRE = re.compile(r"\bunknown\b(?:\s+[\w-]+){0,3}\s*$", re.I)
STOP = r"during|with|and|or|over|under|at|in|on|of|from|to|by|while|before|after|throughout|overlapping|obscur\w+|near|then|plus|above|below|also|is|are"

# 동물이 아닌 소리원
SOURCES = [
    ("ship", r"ships?(?:'s)?|boats?|vessels?|sailboat|ferry|outboard|propell[eo]rs?|engines?|motors?|generators?|machinery|machine|mechanical|pumps?"
             r"|winch|hull|rigging|maruffa|abel-j|ice ?breaker"),
    ("electrical", r"electrical|elecrtical|electrial|electronic|power (?:supply|source)|radio|recorder|recording noise|feedback|overload\w*"
                   r"|dropouts?|60 ?hz|equipment"),
    ("cable", r"cables?|hydrophones?|phones?"),
    ("sonar", r"sonar|echo ?sounders?|fathometer|pingers?|pings?|interrogation|explosions?"),
    ("ice", r"ice"),
    ("water", r"water|waves?|surf|rain|bubbles?|wind"),
    ("human", r"voices?|music|trainers?|divers?|scuba|airplanes?|planes?|helicopter|footsteps|camera"),
    ("other_animal", r"snapping shrimp|shrimp|fish(?:es)?|birds?"),
]
_SRC_RE = [(n, re.compile(r"\b(?:" + p + r")\b", re.I)) for n, p in SOURCES]
# 적혀 있기만 해도 그 소리가 난다고 볼 수 있는 낱말. (ship, ice, phones, tape, water 는 장소·장비로도 쓰여서 짧은 토막일 때만 센다)
STRONG_SRC_RE = re.compile(r"\b(?:engines?|motors?|generators?|machinery|propell[eo]rs?|outboard|pumps?|winch|ice ?breaker|sonar|echo ?sounders?"
                           r"|fathometer|pingers?|explosions?|rain|surf|airplanes?|helicopter|music|snapping shrimp|60 ?hz|electrical|electronic"
                           r"|feedback|dropouts?|overload\w*)\b", re.I)

# 분류군 이름. (이름, 정규식, 그 이름으로 불릴 수 있는 종의 집합 또는 넓은 묶음)
_STEN = {"Stenella attenuata", "Stenella clymene", "Stenella coeruleoalba", "Stenella frontalis", "Stenella longirostris"}
WHALE_SP = {"Balaena mysticetus", "Eubalaena glacialis", "Eubalaena australis", "Eschrichtius robustus", "Balaenoptera acutorostrata",
            "Balaenoptera musculus", "Balaenoptera physalus", "Megaptera novaeangliae", "Delphinapterus leucas", "Monodon monoceros",
            "Globicephala macrorhynchus", "Globicephala melas", "Orcinus orca", "Pseudorca crassidens", "Peponocephala electra"}
DOLPHIN_SP = {"Cephalorhynchus commersonii", "Cephalorhynchus heavisidii", "Delphinus bairdii", "Delphinus delphis", "Grampus griseus",
              "Lagenodelphis hosei", "Lagenorhynchus acutus", "Lagenorhynchus albirostris", "Sousa sahulensis", "Sotalia fluviatilis",
              "Steno bredanensis", "Tursiops truncatus", "Inia geoffrensis", "Lagenorhynchus obscurus"} | _STEN
PINNIPED_GROUP = "Pinnipedia"
TAXA = [
    ("sperm whale", r"sperm whales?|physeter|catodon|cachalots?", set()),
    ("bowhead", r"bowheads?|balaena", {"Balaena mysticetus"}),
    ("right whale", r"right whales?|eubalaena", {"Eubalaena glacialis", "Eubalaena australis"}),
    ("gray whale", r"gr[ae]y whales?|eschrichtius", {"Eschrichtius robustus"}),
    ("minke", r"minkes?(?:'s)?", {"Balaenoptera acutorostrata"}),
    ("blue whale", r"blue whales?|blues", {"Balaenoptera musculus"}),
    ("finback", r"finbacks?|fin whales?", {"Balaenoptera physalus"}),
    ("humpback", r"humpbacks?|megaptera", {"Megaptera novaeangliae"}),
    ("beluga", r"belugas?|white whales?|delphinapterus", {"Delphinapterus leucas"}),
    ("narwhal", r"narwhals?|monodon", {"Monodon monoceros"}),
    ("pilot whale", r"pilot whales?|globicephala|potheads?|melaena|macrorhynchus", {"Globicephala melas", "Globicephala macrorhynchus"}),
    ("killer whale", r"killer whales?|orcas?|orcinus|keiko|moby doll", {"Orcinus orca"}),
    ("false killer whale", r"pseudorca|false killer whales?", {"Pseudorca crassidens"}),
    ("common dolphin", r"delphinus|saddlebacks?|bairdii", {"Delphinus delphis", "Delphinus bairdii"}),
    ("grampus", r"grampus|risso'?s?", {"Grampus griseus"}),
    ("lagenodelphis", r"lagenodelphis|hosei", {"Lagenodelphis hosei"}),
    ("white-beaked dolphin", r"albirostris", {"Lagenorhynchus albirostris"}),
    ("white-sided dolphin", r"acutus", {"Lagenorhynchus acutus"}),
    ("lagenorhynchus", r"lagenorhynchus", {"Lagenorhynchus acutus", "Lagenorhynchus albirostris", "Lagenorhynchus obscurus"}),
    ("dusky dolphin", r"obscurus", {"Lagenorhynchus obscurus"}),
    ("melon-headed whale", r"peponocephala|electra", {"Peponocephala electra"}),
    ("stenella", r"stenella", _STEN),
    ("spotted dolphin", r"attenuata", {"Stenella attenuata"}),
    # 노트의 plagiodon 은 S. frontalis 의 옛 이름이다. 종 라벨이 S. attenuata 인 컷에도 "I.D.'d as plagiodon" 이 있다(종 라벨 확인 대상).
    ("atlantic spotted dolphin", r"plagiodon|frontalis", {"Stenella frontalis", "Stenella attenuata"}),
    ("spinner", r"longirostris|spinners?", {"Stenella longirostris"}),
    ("steno", r"steno|bredanensis", {"Steno bredanensis"}),
    ("tursiops", r"tursiops|bottlenosed?", {"Tursiops truncatus"}),
    ("sotalia", r"sotalia|borneensis", {"Sotalia fluviatilis", "Sousa sahulensis"}),
    ("bearded seal", r"bearded seals?|erignathus", {"Erignathus barbatus"}),
    ("walrus", r"walrus(?:es)?|odobenus", {"Odobenus rosmarus"}),
    ("ringed seal", r"ringed seals?|hispida", {"Pusa hispida"}),
    ("ribbon seal", r"ribbon seals?|fasciata", {"Histriophoca fasciata"}),
    ("harp seal", r"harp seals?|groenlandica", {"Pagophilus groenlandicus"}),
    ("weddell seal", r"weddell(?: seals?)?|leptonychotes", {"Leptonychotes weddellii"}),
    ("leopard seal", r"leopard seals?|hydrurga", {"Hydrurga leptonyx"}),
    ("ross seal", r"ross seals?|ommatophoca", {"Ommatophoca rossi"}),
    ("crabeater seal", r"crabeater(?: seals?)?", set()),
    ("phoca", r"phoca", {"Phoca largha", "Pusa hispida", "Pagophilus groenlandicus", "Histriophoca fasciata"}),
    ("delphinid", r"delphinids?", "dolphin"),
    # 넓은 이름: 대상 종을 그렇게 부를 수 있을 때만 대상으로 본다
    ("whale", r"whales?", "whale"),
    ("dolphin", r"dolphins?|porpoises?", "dolphin"),
    ("seal", r"seals?", "seal"),
]
_TAXA_RE = [(n, re.compile(r"\b(?:" + p + r")\b", re.I), who) for n, p, who in TAXA]


def _taxa_in(text: str, species: str, group: str) -> tuple[list, list]:
    """글에 나온 분류군 이름을 (대상 종을 가리키는 것, 다른 종인 것)으로 나눈다.

    구체적인 이름(sperm whale, Erignathus ...)이 다른 종을 가리키면, 같은 토막의 넓은 이름(whale, seal)은 대상으로 세지 않는다.
    """
    spec_self, spec_other, gen_self, gen_other, used = [], [], [], [], []
    for name, rx, who in _TAXA_RE:
        for mo in rx.finditer(text):
            if any(a <= mo.start() < b for a, b in used):
                continue                       # 'pilot whale' 안의 'whale' 은 다시 세지 않는다
            used.append(mo.span())
            if isinstance(who, set):
                (spec_self if species in who else spec_other).append(name)
            elif who == "whale":
                (gen_self if species in WHALE_SP else gen_other).append(name)
            elif who == "dolphin":
                (gen_self if species in DOLPHIN_SP else gen_other).append(name)
            else:
                (gen_self if group == PINNIPED_GROUP else gen_other).append(name)
    if spec_other and not spec_self:
        return [], spec_other + gen_other
    return spec_self + gen_self, spec_other + gen_other


def _sources_in(text: str) -> list:
    return [name for name, rx in _SRC_RE if rx.search(text)]


@dataclass
class Term:
    term: str                 # 표준 낱말 (click, pulsed_click, whistle, call ...)
    mods: tuple               # 그 낱말에 붙은 꾸밈말 (pulsed, tonal, fm, up, down ...)
    attr: str                 # target / implied / other:<종> / nonanimal:<무엇> / unknown_origin / neg / tape_log
    bg: bool                  # 배경·희미함으로 적혀 있는가
    count: str                # 노트에 적힌 수 (one, 2, many ...). 없으면 빈칸
    text: str                 # 그 토막의 글
    rule: str = ""            # 귀속을 정한 규칙


def _find_terms(text: str) -> list[tuple[str, tuple]]:
    """글에서 소리 낱말을 찾는다. (표준 이름, (시작, 끝)) 을 자리 순서로."""
    found, work = [], text
    for name, rx in _COMPOUND_RE:
        for mo in rx.finditer(work):
            found.append((name, mo.span()))
        work = rx.sub(lambda mo: " " * (mo.end() - mo.start()), work)
    for name, rx in _SINGLE_RE:
        for mo in rx.finditer(work):
            found.append((name, mo.span()))
        work = rx.sub(lambda mo: " " * (mo.end() - mo.start()), work)
    found.sort(key=lambda f: f[1][0])
    return found


def _local_mods(text: str, span: tuple, prev_end: int, next_start: int) -> tuple:
    """낱말 바로 앞(앞 낱말 뒤부터, 많아야 네 낱말)과 바로 뒤의 덧붙임('-modulated', '(stepped)')에서 꾸밈말을 찾는다."""
    pre = " ".join(text[prev_end:span[0]].split()[-4:])
    inner = text[span[0]:span[1]]
    post_full = text[span[1]:next_start]
    mo = re.match(r"^\s*(?:-\s*[\w-]+|\([^)]{0,40}\)?)", post_full)
    post = mo.group(0) if mo else ""
    ctx = f"{pre} {inner} {post}"
    out = [n for n, rx in _MOD_RE if rx.search(ctx)]
    out += [n for n, rx in SIGTYPE_RE if rx.search(f"{pre} {post}") and n not in out]
    return tuple(out)


def _src_adjacent(text: str, span: tuple) -> str:
    """소리원 낱말 바로 뒤에 소리 낱말이 오는 꼴: "ship creaks", "ship's click", "ice crunches", "Maruffa roll-cracks"."""
    before = text[:span[0]]
    for name, rx in _SRC_RE:
        for mo in rx.finditer(before):
            gap = before[mo.end():]
            if re.fullmatch(r"(?:'s)?(?:[\s-]+(?:noise|sounds?|background|engine|generator|roll))*(?:\s*--\s*|[\s(-]+)(?:(?!(?:" + STOP + r")\b)[\w'-]+[\s-]+){0,1}",
                            gap, re.I):
                return name
    return ""


def _src_after(text: str, span: tuple, percussive: bool = False) -> str:
    """소리 낱말 뒤에 소리원이 오는 꼴: "clicks are sounds of ice melting", "noise from ship", "click produced from the hydrophone", "crack on ship"."""
    after = text[span[1]:]
    link = r"is|are|=|of|from|by|made by|caused by|due to|produced (?:by|from)" + (r"|on" if percussive else "")
    mo = re.match(r"\s*(?:present\s+|heard\s+|audible\s+)?(?:" + link + r")\s+(?:the\s+|a\s+)?(?:sounds?\s+of\s+|noise\s+of\s+)?(.{0,40})",
                  after, re.I)
    if not mo:
        return ""
    for name, rx in _SRC_RE:
        if rx.search(mo.group(1)):
            return name
    return ""


def _src_following(text: str, span: tuple) -> str:
    """소리 낱말 바로 뒤에 소리원이 오는 꼴: "Buzzing ice sounds", "squeaking cable"."""
    after = text[span[1]:]
    ing = text[span[0]:span[1]].lower().endswith("ing")
    for name, rx in _SRC_RE:
        mo = rx.search(after)
        if mo and not after[:mo.start()].strip():
            rest = after[mo.end():]
            if re.match(r"\s*sounds?\b", rest, re.I) or (ing and re.match(r"\s*(?:noises?)?\s*(?:$|[),.])", rest, re.I)):
                return name
    return ""


def parse_note(note: str, species: str, group: str) -> dict:
    """노트 하나를 읽어 낱말 층의 결과를 돌려준다.

    돌려주는 것: terms(Term 의 목록), codes, other_codes, x_mark, u_mark, implied, tape_log, air, cleaned,
                 noise_sources(노트에 적힌 동물 밖 소리원), target_bg(대상 동물이 배경에 있다고만 적혀 있는가), other_present(소리 낱말 없이 언급된 다른 종)
    """
    cl = clean_note(note)
    own = OWN_CODES.get(species, set())
    other_codes = [c for c in cl.codes if c not in own]
    items = split_items(cl)
    multi_code = len(cl.codes) >= 2
    terms: list[Term] = []
    noise_sources, other_present = [], []
    target_bg = False
    clause_has_noise = {}
    parsed = []
    for it in items:
        found = _find_terms(it.text)
        srcs = _sources_in(it.text)
        self_tx, other_tx = _taxa_in(it.text, species, group)
        parsed.append((it, found, srcs, self_tx, other_tx))
        heads = {n for n, _ in found}
        if (srcs and not (heads - {"noise", "sound"})) or (heads and heads <= {"noise"}):
            clause_has_noise[it.clause] = True
    seen_heading, seen_clause = set(), set()
    for it, found, srcs, self_tx, other_tx in parsed:
        text = it.text
        seg_other = [c for c in it.codes if c not in own]
        seg_self = [c for c in it.codes if c in own]
        head_self, head_other = _taxa_in(it.heading, species, group) if it.heading else ([], [])
        head_src = _sources_in(it.heading) if it.heading else []
        head_terms = [n for n, _ in _find_terms(it.heading)] if it.heading else []
        head_spec = [n for n in head_terms if n not in GENERIC and n not in ("noise", "burst", "phrase") and n not in AMBIG]
        head_gen = [n for n in head_terms if n in GENERIC]
        mc = COUNT_RE.match(text.strip())
        count = mc.group(1).lower() if mc else ""
        heads = [n for n, _ in found]
        spec_here = [h for h in heads if h not in GENERIC and h != "noise"]
        if not cl.tape_log:
            if "noise" in heads:
                for src in (srcs or ["unspecified"]):
                    if src not in noise_sources:
                        noise_sources.append(src)
            elif not found:                                     # 소리 낱말 없이 소리원만 적힌 토막: "Sonar", "Engine running"
                short = len(text.split()) <= 3
                for src in srcs:
                    if src not in noise_sources and (short or STRONG_SRC_RE.search(text)):
                        noise_sources.append(src)
        if not spec_here and (other_tx and not self_tx):
            for tx in other_tx:
                if tx not in other_present:
                    other_present.append(tx)
        if not heads and self_tx and BG_PRE.search(text):
            target_bg = True

        def attribute(name: str, span: tuple) -> tuple[str, str]:
            pre = text[:span[0]]
            if name == "noise":
                return "nonanimal:" + (srcs[0] if srcs else "unspecified"), "noise_word"
            if cl.tape_log:
                return "tape_log", "tape_log"
            if NEG_RE.search(pre):
                return "neg", "negation"
            if it.extraneous:
                return "nonanimal:extraneous", "after_x_mark"
            if head_src and not head_self:
                return "nonanimal:" + head_src[0], "heading_source"
            if other_tx and not self_tx:
                return "other:" + other_tx[0], "other_taxon_in_item"
            s = _src_adjacent(text, span)
            if s:
                return "nonanimal:" + s, "source_before_term"
            s = _src_after(text, span, name in AMBIG)
            if s and not self_tx:
                return "nonanimal:" + s, "term_of_source"
            s = _src_following(text, span)
            if s:
                return "nonanimal:" + s, "term_then_source"
            if head_other and not head_self and not self_tx:
                return "other:" + head_other[0], "heading_other_taxon"
            if multi_code and seg_other and not seg_self and not self_tx and not head_self:
                return "other:" + CODE_NAME.get(seg_other[0], seg_other[0]), "segment_code"
            if name in AMBIG:
                if UNKNOWN_PRE.search(pre):
                    return "unknown_origin", "unknown_word"
                if self_tx or BODY_RE.search(text) or (name, species) in AMBIG_OWN:
                    return "target", "body_or_species"
                return "unknown_origin", ("percussive_with_noise" if clause_has_noise.get(it.clause) else "percussive")
            if UNKNOWN_PRE.search(pre):
                return "unknown_origin", "unknown_word"
            if srcs and not self_tx:
                return "target", "source_word_in_item"         # 소리원 낱말이 같은 토막에 있다 (검토 대상)
            return "target", ""

        def not_a_sound(name: str, span: tuple) -> bool:
            if name == "crack" and re.match(r"\s+(?:in|at|throughout|through|during|before|within)\b[^,;]*\bwhistles?\b", text[span[1]:], re.I):
                return True                                     # "cracks in whistle" 는 휘슬이 끊긴다는 뜻이다
            return bool(SIMILE_PRE.search(text[:span[0]]) or XLIKE_POST.match(text[span[1]:]))

        dead = [not_a_sound(n, sp) for n, sp in found]         # 비유·꾸밈말로 쓰인 낱말
        head_extra = ()
        if (len(found) == 1 and found[0][0] in SWEEPS and head_spec and len(text.split()) <= 2
                and not (head_src and not head_self) and not (head_other and not head_self)):
            dead[0] = True                                      # "One Grunt--upsweep": 올라가는 grunt 하나
            head_extra = ("up",) if found[0][0] == "upsweep" else ("down",) if found[0][0] == "downsweep" else ("fm",)
        adj_mods = {}
        for i in range(len(found) - 1):
            (a, sa), (b, sb) = found[i], found[i + 1]
            if dead[i] or dead[i + 1] or not re.fullmatch(r"\s*(?:\([^)]*\))?\s*(?:pulsed\s+|PU\s+)?", text[sa[1]:sb[0]], re.I):
                continue
            if a in SWEEPS and b in ("moan", "groan", "call"):
                dead[i] = True                                  # 앞의 sweep 은 뒤 낱말의 꾸밈말이다
                adj_mods[i + 1] = ("up",) if a == "upsweep" else ("down",) if a == "downsweep" else ("fm",)
            elif a == "call" and b in SWEEPS:
                dead[i] = True                                  # "Long call sweep"
            elif b == "call" and a not in GENERIC and a != "noise":
                dead[i + 1] = True                              # "PU buzz call", "chirp calls": 앞 낱말이 구체적이다
        live_heads = [n for (n, _), d in zip(found, dead) if not d]
        emitted = []
        for idx, (name, span) in enumerate(found):
            prev_end = found[idx - 1][1][1] if idx else 0
            next_start = found[idx + 1][1][0] if idx + 1 < len(found) else len(text)
            if dead[idx]:
                continue
            if name in WEAK_HEADS and any(h not in WEAK_HEADS and h != "noise" for h in live_heads):
                continue
            if name == "sound" and "noise" in live_heads:
                continue                                        # "ship noise present throughout sound cut"
            if name in GENERIC and head_spec and not (head_src or (head_other and not head_self)):
                continue                                        # "Whistles - a series of 'Hey Suzy' calls": 머리말의 낱말이 구체적이다
            attr, rule = attribute(name, span)
            mods = _local_mods(text, span, prev_end, next_start)
            mods = mods + tuple(x for x in adj_mods.get(idx, ()) if x not in mods)
            if idx and dead[idx - 1] and XLIKE_POST.match(text[found[idx - 1][1][1]:]):
                extra = XLIKE_MOD.get(found[idx - 1][0])       # "chirp like call" → call 에 chirpy
                if extra and extra not in mods:
                    mods = mods + (extra,)
            bg = bool(BG_PRE.search(" ".join(text[prev_end:span[0]].split()[-3:])) or BG_POST.search(text[span[1]:]))
            emitted.append(Term(term=name, mods=mods, attr=attr, bg=bg, count=count, text=text, rule=rule))
        # 머리말의 낱말: 구체적이면 한 번 적는다. 일반 낱말(Calls-)이면 토막에 낱말이 없고 꾸밈말·수만 있을 때 적는다.
        if it.heading and not cl.tape_log and not (head_src and not head_self) and not (head_other and not head_self):
            if head_spec and (it.clause, "spec") not in seen_heading:
                seen_heading.add((it.clause, "spec"))
                for n in head_spec:
                    hm = tuple(m for m, rx in _MOD_RE if rx.search(it.heading))
                    hm = hm + tuple(x for x in head_extra if x not in hm)
                    emitted.append(Term(term=n, mods=hm, attr="target", bg=False, count="", text=it.heading + " --", rule="heading_term"))
            elif head_gen and not head_spec and not [e for e in emitted if e.term != "noise"]:
                mods = tuple(n for n, rx in _MOD_RE if rx.search(text)) + tuple(n for n, rx in SIGTYPE_RE if rx.search(text))
                if mods or count:
                    emitted.append(Term(term=head_gen[0], mods=mods, attr="target", bg=bool(BG_PRE.search(text)), count=count,
                                        text=f"{it.heading} -- {text}", rule="heading_generic"))
        # 낱말 없이 series 만 있는 토막: "Good clicks; fast series" → 바로 앞의 클릭 낱말을 이어받는다
        if not found and terms and re.fullmatch(r"(?:including\s+)?(?:an?\s+)?(?:fast|rapid|slow|long|short|good)?\s*(?:series|trains?|sequences?)",
                                                 text.strip(), re.I):
            prev = terms[-1]
            if prev.term in ("click", "pulsed_click", "pulse") and prev.attr in ("target", "tape_log"):
                emitted.append(Term(term="pulse_series" if prev.term == "pulse" else "click_series",
                                    mods=tuple(n for n, rx in _MOD_RE if rx.search(text)), attr=prev.attr,
                                    bg=False, count="", text=text, rule="series_of_previous"))
        # 명사 없이 쓰인 tonal: "Long tonal also in cut 97504028" (구절의 첫 토막이고 머리말에 구체적인 낱말이 없을 때만)
        first_in_clause = it.clause not in seen_clause
        seen_clause.add(it.clause)
        if (not found and not emitted and first_in_clause and not head_spec and not cl.tape_log and not it.extraneous
                and re.match(r"^(?:long |short |faint )?tonals?\b(?!\s+(?:characteristics?|quality))", text, re.I)):
            emitted.append(Term(term="sound", mods=("tonal",), attr="target", bg=bool(BG_PRE.search(text)), count="", text=text, rule="tonal_noun"))
        # 낱말 없이 신호 유형 줄임말만 있는 토막: "NL in back", "2 PU"
        if not found and not emitted and re.match(r"^\s*(?:\d+\s+)?(?:PU|N[LS]|B[SL]|FM)\b(?!\s+component)", text):
            mods = tuple(n for n, rx in SIGTYPE_RE if rx.search(text)) + (("fm",) if re.search(r"\bFM\b", text) else ())
            attr = "tape_log" if cl.tape_log else ("nonanimal:extraneous" if it.extraneous else "target")
            emitted.append(Term(term="sound", mods=mods, attr=attr, bg=bool(BG_POST.search(" " + text)), count=count, text=text, rule="sigtype_only"))
        terms += emitted
    for t in terms:                                             # 동물 밖 소리로 읽은 낱말의 소리원도 잡음원에 넣는다
        if t.attr.startswith("nonanimal:"):
            src = t.attr.split(":", 1)[1]
            src = "unspecified" if src == "extraneous" else src
            if src not in noise_sources:
                noise_sources.append(src)
    for name in cl.implied:
        terms.append(Term(term=name, mods=(), attr="implied", bg=False, count="", text="(cut made for odontocete click database)", rule="implied"))
    return dict(terms=terms, codes=cl.codes, other_codes=other_codes, x_mark=cl.x_mark, u_mark=cl.u_mark, implied=sorted(set(cl.implied)),
                tape_log=cl.tape_log, tape_summary=cl.tape_summary, items=[it.text for it in items], air=bool(AIR_RE.search(cl.text)), cleaned=cl.text,
                noise_sources=noise_sources, target_bg=target_bg, other_present=other_present)


# --------------------------------------------------------------------------------------------------
# 3. 형태 층: 낱말 → 형태 범주. data/note_term_map.csv 의 대응표를 쓴다.
# --------------------------------------------------------------------------------------------------
CLICK_FAMILY = {"click", "click_series", "pulse", "burst", "knock", "echolocation"}
PULSED_BUILTIN = {"pulsed_click", "pulse_burst", "pulse_series", "pulse", "click_burst"}     # 이름에 이미 pulsed 가 들어 있는 낱말
CONTOUR = {"fm", "up", "down", "whistley", "chirpy", "looped", "stepped"}                    # 주파수 윤곽을 말하는 꾸밈말


def term_key(t: Term) -> str:
    """대응표의 열쇠: 낱말 + 형태를 바꾸는 꾸밈말 한 가지.

    +pulsed  펄스성이라고 적힘 (PU, pulsed)
    +train   클릭·펄스 낱말에 빠른 반복을 뜻하는 말이 붙음 (fast, rapid, burst, buzzy, tonal)
    +fm      일반 낱말(call, sound)에 주파수 윤곽을 뜻하는 말이 붙음 (FM, modulated, upsweeping, looped ...)
    +tonal   일반 낱말에 좁은 대역이라는 말만 붙음 (tonal, NL, narrow)
    """
    m = set(t.mods)
    if "pulsed" in m and t.term not in PULSED_BUILTIN:
        return t.term + "+pulsed"
    if t.term in CLICK_FAMILY and m & {"fast", "burst", "buzzy", "tonal"}:
        return t.term + "+train"
    if t.term in GENERIC:
        if m & CONTOUR:
            return t.term + "+fm"
        if "tonal" in m:
            return t.term + "+tonal"
    return t.term


def load_term_map(path=None) -> dict:
    """대응표를 읽는다. (낱말 열쇠, 범위) → (형태 범주, 근거).

    범위(scope)는 '*'(모든 분류군), 분류군 이름(Odontoceti ...), 또는 종의 학명.
    근거(basis)는 word(노트의 낱말이 곧 그 형태) / reading(해석이 들어감, 청취로 확인할 대상).
    """
    out = {}
    with open(path or default_map_path(), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["term_key"] and not r["term_key"].startswith("#"):
                out[(r["term_key"].strip(), (r["scope"] or "*").strip())] = (r["type"].strip(), (r.get("basis") or "").strip())
    return out


def map_row(key: str, group: str, table: dict, species: str = "") -> tuple[str, str]:
    """대응표에서 (범주, 근거)를 찾는다: 종 → 분류군 → '*' 의 순서. 꾸밈말이 붙은 열쇠가 표에 없으면 꾸밈말을 떼고 다시 찾는다."""
    base = key.split("+")[0]
    for k in ((key, species), (key, group), (key, "*"), (base, species), (base, group), (base, "*")):
        if k in table:
            return table[k]
    return "other", "unmapped"


def map_class(key: str, group: str, table: dict, species: str = "") -> str:
    return map_row(key, group, table, species)[0]


PULSED_FLAG_TERMS = {"whistle", "moan", "groan", "chirp", "upsweep", "downsweep", "sweep", "trill"}   # pulsed 가 붙으면 형태가 애매해지는 낱말
DESC_MODS = {"low", "high", "long", "short", "broad"}
DESC_RE = re.compile(r"\b(?:percussive|rubbing|hitting|unusual|weird|strange|odd)\b", re.I)


def summarize(res: dict, species: str, group: str, table: dict, type_classes: list) -> dict:
    """컷 하나의 낱말 층 결과를 표의 열로 간추린다.

    type_weak        노트가 대상 동물의 소리로 적은 형태 (여러 개일 수 있다). 지운 문구가 암시하는 것(click database)도 넣는다.
    type_single      형태가 정확히 하나이고, 다른 것이 섞였다는 말이 없는 컷만. 아래 가운데 하나라도 있으면 비운다.
                       - 8개 범주에 들지 않는 소리 낱말 (bell, chug, squeak ...)                → type_unplaced
                       - 형태를 알 수 없는 소리가 따로 적혀 있음 (call, 'lower frequency sounds')  → type_unplaced 의 call, sound+desc
                       - 다른 종의 소리가 적혀 있음                                              → note_other_sound
    type_incomplete  형태를 알 수 없는 대상 동물의 소리가 따로 적혀 있다(call, sound+desc). 8개 범주의 유무를 다 알 수 없으므로 다중 형태 과제에서도 뺀다.

    형태를 말하지 않는 낱말의 처리
      sound, signals, social sounds   적힌 소리를 통틀어 가리키는 말로 보고 세지도 막지도 않는다. 단, 'low sound', 'percussive sounds' 처럼
                                      꾸밈말이 붙어 따로 있는 소리를 가리키면 sound+desc 로 적고 단일 형태를 막는다.
      call                            목록의 한 항목으로 적힌 call("Calls; pulsed calls", "2 calls")은 형태를 모르는 소리가 따로 있다는 뜻으로 본다.
                                      노래가 함께 적혀 있거나("2 calls. Song elements"), 긴 설명문 속에서 앞의 'tonal call' 등을 다시 가리킬 때는 세지 않는다.
    """
    tgt = [t for t in res["terms"] if t.attr == "target"]
    imp = [t for t in res["terms"] if t.attr == "implied"]
    rows = [(t, term_key(t)) + map_row(term_key(t), group, table, species) for t in tgt]      # (낱말, 열쇠, 범주, 근거)
    has_modified_call = any(t.term in ("call", "call_squeal") and k != "call" for t, k, _c, _b in rows)
    has_song = any(c == "song" for _t, _k, c, _b in rows)
    keys_t, stated, fg, unplaced, readings = [], set(), set(), [], []
    for t, k, c, b in rows:
        if k not in keys_t:
            keys_t.append(k)
        if c in type_classes:
            stated.add(c)
            if not t.bg:
                fg.add(c)
            if b != "word" and k not in readings:
                readings.append(k)
        elif c == "unspec":
            mark = ""
            if k == "call":
                listed = len(t.text.split()) <= 4               # "Calls", "2 calls", "One long call"
                if not (has_song or (has_modified_call and not listed)):
                    mark = "call"
            elif t.term == "sound" and (set(t.mods) & DESC_MODS or DESC_RE.search(t.text)):
                mark = "sound+desc"
            if mark and mark not in unplaced:
                unplaced.append(mark)
        elif k not in unplaced:
            unplaced.append(k)                                  # other: 8개 범주 밖의 소리
    implied = {map_class(term_key(t), group, table, species) for t in imp} & set(type_classes)
    named = [c for c in type_classes if c in (stated | implied)]
    stated_l = [c for c in type_classes if c in stated]
    incomplete = any(u in ("call", "sound+desc") for u in unplaced)

    def pack(prefix: str) -> str:
        out = []
        for t in res["terms"]:
            if t.attr.startswith(prefix) and t.term != "noise":
                who = t.attr.split(":", 1)[1] if ":" in t.attr else ""
                s1 = f"{who}:{term_key(t)}" if who else term_key(t)
                if s1 not in out:
                    out.append(s1)
        return ";".join(out)

    other_sound = pack("other:")
    single = named[0] if (len(named) == 1 and not unplaced and not other_sound) else ""
    flags = []
    if res["x_mark"]:
        flags.append("x_mark")
    if res["u_mark"]:
        flags.append("u_mark")
    if res["tape_log"]:
        flags.append("tape_log")
    if res.get("tape_summary"):
        flags.append("tape_summary")
    if imp:
        flags.append("implied_click_only" if not stated else "implied_click")
    if res["target_bg"] and not tgt:
        flags.append("target_in_background")
    if res["other_codes"]:
        flags.append("other_code:" + ",".join(res["other_codes"]))
    if res["air"]:
        flags.append("air")
    if any(t.rule == "source_word_in_item" for t in tgt):
        flags.append("source_word_near_term")
    if any("pulsed" in t.mods and t.term in PULSED_FLAG_TERMS for t in tgt):
        flags.append("pulsed_modifier")
    other_sp = [other_sound] + [f"{t}:(present)" for t in res["other_present"]]
    return dict(
        type_weak=";".join(named), type_weak_n=len(named), type_single=single,
        type_stated=";".join(stated_l), type_implied=";".join(c for c in type_classes if c in implied),
        type_terms=";".join(keys_t), type_unplaced=";".join(unplaced), type_incomplete="1" if incomplete else "",
        type_bg_only=";".join(c for c in stated_l if c not in fg), type_reading=";".join(readings),
        note_nonanimal=pack("nonanimal:"), note_other_sound=other_sound, note_other_species=";".join(x for x in other_sp if x),
        note_unknown=pack("unknown_origin"), note_negated=pack("neg"), note_noise_src=";".join(res["noise_sources"]),
        note_flags=";".join(flags),
        note_evidence=" | ".join(dict.fromkeys(f"{t.text} → {term_key(t)}" for t in tgt))[:400],
    )


NOTE_COLUMNS = ["type_weak", "type_weak_n", "type_single", "type_stated", "type_implied", "type_terms", "type_unplaced", "type_incomplete",
                "type_bg_only", "type_reading", "note_nonanimal", "note_other_sound", "note_other_species", "note_unknown", "note_negated",
                "note_noise_src", "note_flags", "note_evidence"]


def default_map_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "note_term_map.csv"
