#!/usr/bin/env python3
"""make-prompt.py - writes bench/prompt.txt, the fixed prompt of bench/launch-bench.py and launch-bench-mlx.py: llama.cpp's
src/*.cpp at one commit, in sorted order, each after a "// ===== name =====" line, stopping after the file that takes the
text past --chars characters (61440 * 5, the old default cut). Byte for byte the text the bench used to build from the
checkout's own src/, so rows from before and after the pinned file compare.

The default commit is the one main's llama.cpp submodule pinned when the prompt was pinned. The published rows in
bench/results/results.jsonl name b613bde9d, which is not on GitHub; the files in the cut are identical at 200856db7 (the
pin committed with those rows) and at 1839b7817.

  bench/make-prompt.py                 # rewrite bench/prompt.txt
  bench/make-prompt.py --check         # exit 1 if bench/prompt.txt differs from what the commit gives
"""
import argparse, hashlib, io, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--commit', default='1839b78175d4f89f74023a2ced04f4e558abc4d4')
ap.add_argument('--repo', default=os.path.join(ROOT, 'llama.cpp'))
ap.add_argument('--chars', type=int, default=61440 * 5)
ap.add_argument('--out', default=os.path.join(ROOT, 'bench', 'prompt.txt'))
ap.add_argument('--check', action='store_true')
a = ap.parse_args()


def git(*args):
    r = subprocess.run(['git', '-C', a.repo, *args], capture_output=True)
    if r.returncode:
        sys.exit(f'git {" ".join(args)}: {r.stderr.decode().strip()}')
    return r.stdout


names = sorted(n for n in git('ls-tree', '--name-only', a.commit, 'src/').decode().split('\n') if n.endswith('.cpp'))
text = ''
for n in names:
    # the old bench read each file with open(f, errors='replace'): UTF-8, universal newlines
    text += f'\n// ===== {os.path.basename(n)} =====\n' + io.TextIOWrapper(io.BytesIO(git('show', f'{a.commit}:{n}')),
                                                                               encoding='utf-8', errors='replace').read()
    if len(text) > a.chars:
        break
else:
    sys.exit(f'only {len(text)} characters in src/*.cpp at {a.commit}, need more than {a.chars}')
data = text.encode('utf-8')
sha = hashlib.sha256(data).hexdigest()
if a.check:
    old = open(a.out, 'rb').read()
    print(f'{a.out}: sha256 {hashlib.sha256(old).hexdigest()}; {a.commit[:9]} gives {sha}')
    sys.exit(0 if old == data else 1)
with open(a.out, 'wb') as f:
    f.write(data)
print(f'{a.out}: {len(data)} bytes, sha256 {sha}, from llama.cpp {a.commit[:9]} ({names.index(n) + 1} files, through {n})')
