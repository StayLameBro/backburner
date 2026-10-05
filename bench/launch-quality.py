#!/usr/bin/env python3
"""launch-quality.py - does the iPhone change the model's output? Compares the greedy decode runs that bench/launch-bench.py
recorded (bench/results/runs.jsonl) between two configs at each depth:
  - identical: how many of the generated tokens match before the first difference (greedy, same prompt);
  - top-1 agreement and the largest probability gap of the top token over the positions both runs share (up to 64);
  - probs: how many of those positions had probabilities in both rows. Speculative decoding returns them for the first
    token only (#12), so 1 / 64 means the two columns before it rest on one token; fork-nospec gives all of them.
  - first token: the top-10 log-probs right after the prompt, which depend only on how the prompt was read.
Each side is the latest run of its config; with --a and --b the same config, A is the run before the latest. Every depth
prints both rows' run_id and prompt_sha256, and a warning if they are one run, read different prompts, or lack a hash.

  bench/launch-quality.py [--a fork-mac] [--b fork-phone]
"""
import argparse, json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--a', default='fork-mac')
ap.add_argument('--b', default='fork-phone')
ap.add_argument('--file', default=os.path.join(ROOT, 'bench', 'results', 'runs.jsonl'))
a = ap.parse_args()

if not os.path.exists(a.file):
    sys.exit(f'{a.file}: no such file (bench/launch-bench.py writes it; --file for another, e.g. bench/results/results.jsonl)')
runs = {}
for line in open(a.file):
    r = json.loads(line)
    if r.get('kind') == 'decode' and r.get('tokens') and r.get('note') != 'smoke':
        runs.setdefault((r['config'], r['depth']), []).append(r)   # in file order: the last is the latest run


def warn(msg):
    print(f'        !!!!! WARNING: {msg}')


print(f'{a.a} vs {a.b} (greedy, same prompt) from {a.file}')
print(f'{"depth":>7} {"identical":>12} {"top-1 same":>11} {"max |dp| top-1":>15} {"probs":>11} {"first-token top-10 overlap":>27}')
shown = 0
for d in sorted({k[1] for k in runs}):
    rows_a, rows_b = runs.get((a.a, d), []), runs.get((a.b, d), [])
    if a.a == a.b:
        rows_a = rows_a[:-1]
    if not rows_a or not rows_b:
        continue
    ra, rb = rows_a[-1], rows_b[-1]
    shown += 1
    ta, tb = ra['tokens'], rb['tokens']
    same = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    pa, pb = ra.get('probs') or [], rb.get('probs') or []
    agree = n = 0
    dp = 0.0
    for i in range(min(len(pa), len(pb), same + 1)):   # positions with the same history
        if not pa[i] or not pb[i]:
            continue
        n += 1
        agree += pa[i][0][0] == pb[i][0][0]
        lb = dict(map(tuple, pb[i]))
        if pa[i][0][0] in lb:
            dp = max(dp, abs(math.exp(pa[i][0][1]) - math.exp(lb[pa[i][0][0]])))
    ov = len({x[0] for x in pa[0]} & {x[0] for x in pb[0]}) if pa and pb and pa[0] and pb[0] else 0
    want = min(64, same + 1, len(ta), len(tb))   # positions with the same history, as far as probabilities are recorded
    print(f'{d:7d} {same:5d} / {min(len(ta), len(tb)):<5d} {agree:4d} / {n:<4d} {dp:15.4f} {n:4d} / {want:<4d} {ov:22d} / 10')
    ia, ib = ra.get('run_id'), rb.get('run_id')
    ha, hb = ra.get('prompt_sha256'), rb.get('prompt_sha256')
    for side, r, h in (('a', ra, ha), ('b', rb, hb)):
        print(f'        {side}: {r["config"]:<12} run {r.get("run_id") or "-":<36} prompt {(h or "-")[:16]}')
    if ia and ia == ib:
        warn(f'both rows come from the same run ({ia}): this compares a run with itself')
    if not ha or not hb:
        warn(f'no prompt_sha256 on {"either row" if not ha and not hb else "row " + ("a" if not ha else "b")} (written before '
             'bench/prompt.txt): nothing shows both runs read the same prompt')
    elif ha != hb:
        warn(f'the two runs read different prompts ({ha[:16]} vs {hb[:16]}): the comparison means nothing')
    if n < want:
        warn(f'probabilities on only {n} of {want} positions: "top-1 same" and "max |dp|" cover {n}, not {want} (speculative '
             'decoding returns them for the first token only, #12; fork-nospec has them all). "identical" covers every token.')
if not shown:
    print('no depth has decode rows for both' + (f' (two runs of {a.a})' if a.a == a.b else ''))
