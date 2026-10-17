#!/usr/bin/env python3
"""fix_ppr.py — pptxgenjs 가 한 문단 안 두 번째 런부터 끼워 넣는 <a:pPr><a:buNone/></a:pPr> 를 지운다.
   a:pPr 는 문단 맨 앞에만 올 수 있으므로, </a:r> 뒤에 나오는 a:pPr 는 모두 제거한다."""
import re, sys, zipfile, shutil, os
src = sys.argv[1]; tmp = src + ".tmp"
pat = re.compile(r'(</a:r>)<a:pPr\b[^>]*?(?:/>|>.*?</a:pPr>)', re.S)
n = 0
with zipfile.ZipFile(src) as zi, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
    for it in zi.infolist():
        data = zi.read(it.filename)
        if it.filename.startswith("ppt/slides/slide") and it.filename.endswith(".xml"):
            s = data.decode("utf-8"); s, k = pat.subn(r"\1", s); n += k; data = s.encode("utf-8")
        zo.writestr(it, data)
shutil.move(tmp, src); print("removed", n)
