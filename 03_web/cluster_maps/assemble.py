#!/usr/bin/env python3
"""assemble.py — data.json 을 template.html 에 넣어 한 파일짜리 HTML 을 만든다.
   usage: python3 assemble.py data.json out.html"""
import sys
from pathlib import Path
here = Path(__file__).parent
data = Path(sys.argv[1]).read_text(encoding="utf-8").replace("</", "<\\/")
Path(sys.argv[2]).write_text((here / "template.html").read_text(encoding="utf-8").replace("__DATA__", data), encoding="utf-8")
print(sys.argv[2], f"{Path(sys.argv[2]).stat().st_size / 1e6:.2f} MB")
