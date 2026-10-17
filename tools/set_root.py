"""Rewrite the workspace root hard-coded in the scripts.

The analysis was run in a workspace rooted at /home/claude. Run this once after cloning:

    python tools/set_root.py ~/watkins_work

Every '/home/claude/' prefix in .py/.js/.sh/.html files under this repo is replaced by the new root.
Run it again with another root to move again (it remembers the current root in tools/.root).
"""
import sys, pathlib
repo = pathlib.Path(__file__).resolve().parents[1]
state = repo / 'tools' / '.root'
old = state.read_text().strip() if state.exists() else '/home/claude/'
if len(sys.argv) != 2:
    sys.exit(f'usage: python tools/set_root.py NEW_ROOT   (current: {old})')
new = str(pathlib.Path(sys.argv[1]).expanduser().resolve()).rstrip('/') + '/'
n = 0
for p in repo.rglob('*'):
    if p.suffix in {'.py', '.js', '.sh', '.html'} and p.is_file() and p.name != 'set_root.py':
        s = p.read_text(encoding='utf-8')
        if old in s:
            p.write_text(s.replace(old, new), encoding='utf-8'); n += 1
state.write_text(new)
print(f'{old} -> {new}  ({n} files)')
