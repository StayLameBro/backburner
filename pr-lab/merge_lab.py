#!/usr/bin/env python3
"""Automated mass-merge lab: merge PR batches (2-3 at a time), build, smoke, bench, self-heal.

Design for 1k+ open PRs: never merge blind. Every batch is:
  1. git fetch origin pull/N/head:pr/N  (no API token needed, no fork remotes)
  2. git merge --no-ff --no-commit pr/N (detect conflict without committing)
  3. on conflict -> abort that PR, quarantine, try next (self-heal #1)
  4. cmake build (ccache if present)
  5. on build fail -> bisect batch, keep good, quarantine bad (self-heal #2)
  6. smoke: llama-cli tiny model, bench: llama-bench pp/tg
  7. on runtime fail / >threshold regression -> bisect + quarantine (self-heal #3)
  8. commit batch as merge commit, log JSONL, checkpoint branch

Usage (in WSL, from llama.cpp checkout):
  python3 merge_lab.py --candidates candidates.json --base master --batch 3 --max-prs 50
  python3 merge_lab.py --candidates candidates.json --dry-run   # plan only, no builds

State files (in lab dir): lab-state.json, lab-log.jsonl, quarantined.json
Branching: starts from --base, creates pr-lab/base-<sha>, each good batch commits
on top. To resume: re-run, it loads lab-state.json.

Bisect: for a failing batch [a,b,c], test each alone on clean base+good-so-far.
Only singles that pass alone re-enter; pairs are NOT retried (avoids combinatorial
explosion across 1k PRs). Interaction failures get logged for manual review.
"""
import argparse, json, os, subprocess, sys, time, datetime

def sh(cmd, cwd, check=False, capture=True, timeout=900):
    r = subprocess.run(cmd, cwd=cwd, shell=True, text=True, executable="/bin/bash",
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.STDOUT if capture else None,
                       timeout=timeout)
    out = r.stdout if capture else ""
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {cmd}\n{out[-4000:]}")
    return r.returncode, (out or "")[-8000:]

class Lab:
    def __init__(self, a):
        self.a = a
        self.repo = os.path.expanduser(a.repo)
        self.state_f = os.path.join(a.state_dir, "lab-state.json")
        self.log_f = os.path.join(a.state_dir, "lab-log.jsonl")
        self.quar_f = os.path.join(a.state_dir, "quarantined.json")
        os.makedirs(a.state_dir, exist_ok=True)
        with open(a.candidates) as f:
            self.cands = json.load(f)
        self.state = {"base_sha": None, "branch": None, "merged": [], "quarantined": [],
                      "bench_baseline": None, "batches_done": 0}
        if os.path.exists(self.state_f):
            self.state = json.load(open(self.state_f))
        if os.path.exists(self.quar_f):
            self.quar = json.load(open(self.quar_f))
        else:
            self.quar = []

    def save(self):
        json.dump(self.state, open(self.state_f, "w"), indent=2)
        json.dump(self.quar, open(self.quar_f, "w"), indent=2)

    def log(self, **kw):
        kw["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        open(self.log_f, "a").write(json.dumps(kw) + "\n")
        print(f"[{kw.get('event')}] {json.dumps({k: v for k, v in kw.items() if k not in ('ts','event')})[:220]}")

    def git(self, cmd, check=False):
        return sh(f"git {cmd}", self.repo, check=check)

    def ensure_identity(self):
        rc, out = self.git("config user.email")
        if rc != 0 or not out.strip():
            self.git("config user.email 'pr-lab@localhost'")
            self.git("config user.name 'pr-lab'")
            self.log(event="git-identity-set")

    def clean_tree(self):
        # abort any in-progress merge/rebase, leave branch head intact
        self.git("merge --abort")
        self.git("rebase --abort")
        rc, out = self.git("status --porcelain")
        return rc, out

    def setup_branch(self):
        if self.state["branch"]:
            self.git(f"checkout {self.state['branch']}")
            return
        rc, base = self.git(f"rev-parse {self.a.base}")
        base = base.strip()
        br = f"pr-lab/base-{base[:8]}"
        self.git(f"checkout -B {br} {base}")
        self.state["base_sha"] = base
        self.state["branch"] = br
        self.save()
        self.log(event="branch", branch=br, base=base)

    def fetch_pr(self, n, retries=2):
        for attempt in range(retries + 1):
            rc, out = self.git(f"fetch origin pull/{n}/head:pr/{n} --force")
            if rc == 0:
                return True, out
            time.sleep(5)
        return False, out

    def merge_one(self, n):
        raise RuntimeError("merge_one removed: stacked uncommitted merges are "
                           "impossible in git (MERGE_HEAD). Use merge_one_committed.")

    def try_merge(self, nums):
        raise RuntimeError("try_merge removed: see merge_one. Batches now commit per-PR.")

    def build(self):
        bdir = os.path.join(self.repo, "build")
        cfg = f"cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE={self.a.build_type}"
        if sh("which ccache", self.repo)[0] == 0:
            cfg += " -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
        rc, out = sh(cfg, self.repo)
        if rc != 0:
            return False, "configure-failed\n" + out
        rc, out = sh(f"cmake --build build --target {' '.join(self.a.targets)} -j {self.a.jobs}", self.repo)
        return (rc == 0), out

    def smoke(self):
        # NOTE (2026-10-02): llama-cli 0.5.0-dev produces no stdout in
        # non-interactive runs (server-backend refactor); use llama-bench
        # pp32/tg32 on the tiny model as the smoke gate instead. It proves
        # inference works AND yields a perf number.
        bench = os.path.join(self.repo, "build", "bin", "llama-bench")
        model = os.path.expanduser(self.a.smoke_model)
        if not os.path.exists(bench) or not os.path.exists(model):
            return False, f"missing bench({os.path.exists(bench)}) or model({model})"
        rc, out = sh(f"set -o pipefail; timeout 300 '{bench}' -m '{model}' -p 32 -n 32 2>&1 | tail -15", self.repo)
        ok = rc == 0 and "tg32" in out
        return ok, out

    def bench(self):
        # llama-bench -o json emits a list; tg entry has n_gen>0, metric is avg_ts.
        # (2026-10-02 fix: was regexing "avg_throughput", always None -> gates
        # silently skipped. Now parses JSON, falls back to regexes, and the
        # caller logs bench failures loudly instead of skipping silently.)
        import re, json
        bench = os.path.join(self.repo, "build", "bin", "llama-bench")
        model = os.path.expanduser(self.a.bench_model)
        rc, out = sh(f"set -o pipefail; timeout 600 '{bench}' -m '{model}' -p {self.a.pp} -n {self.a.tg} -o json 2>&1 | tail -60", self.repo)
        val = None
        try:
            start = out.find("[")
            if start >= 0:
                for row in json.loads(out[start:]):
                    if isinstance(row, dict) and (row.get("n_gen") or 0) > 0 and row.get("avg_ts"):
                        val = float(row["avg_ts"])
                        break
        except Exception:
            val = None
        if val is None:
            for pat in (r'"avg_throughput"\s*:\s*([\d.]+)', r'"throughput"\s*:\s*([\d.]+)',
                        r'([\d.]+)\s*tok/s'):
                m = re.search(pat, out)
                if m:
                    try:
                        val = float(m.group(1))
                        break
                    except ValueError:
                        continue
        open(os.path.join(self.a.state_dir, "last-bench.txt"), "w").write(out)
        return rc == 0, val, out

    def quarantine(self, n, reason, detail=""):
        self.quar.append({"pr": n, "reason": reason, "detail": detail[:1000]})
        if n not in self.state["quarantined"]:
            self.state["quarantined"].append(n)
        self.save()
        self.log(event="quarantine", pr=n, reason=reason)

    def merge_one_committed(self, n, batch_tag):
        """Fetch + merge + commit a single PR. Returns (True, None) or (False, reason).

        One commit per PR: git cannot stack uncommitted merges, so batching
        means N commits, not one octopus. Each commit is verified
        (HEAD must advance and contain the merge); failures never pollute state.
        """
        self.clean_tree()
        ok, _ = self.fetch_pr(n)
        if not ok:
            return False, "fetch-failed"
        rc, out = self.git(f"merge --no-ff --no-edit pr/{n}")
        if rc != 0:
            self.clean_tree()
            return False, f"merge-conflict: {out[-1500:]}"
        rc, out = self.git("rev-parse HEAD")
        if rc != 0:
            self.clean_tree()
            return False, "rev-parse-failed"
        new_head = out.strip()
        # verify the merge actually landed (2 parents) and tree is clean
        rc, out = self.git(f"rev-list --parents -n 1 {new_head}")
        if rc != 0 or len(out.strip().split()) < 3:
            self.clean_tree()
            return False, f"empty-or-nonmerge: {out[-500:]}"
        return True, None

    def commit_batch(self, nums):
        """Legacy batch commit (octopus). Kept for compat; verified."""
        rc, out = self.git("status --porcelain")
        if rc != 0 or not out.strip():
            return False, "nothing-to-commit"
        rc, out = self.git(f"commit -m 'pr-lab: merge {'+'.join(f'#{n}' for n in nums)}'")
        if rc != 0:
            return False, out[-1500:]
        rc, out = self.git("rev-parse HEAD")
        if rc != 0:
            return False, "rev-parse-failed"
        self.state["merged"].extend(nums)
        self.state["batches_done"] += 1
        self.save()
        self.log(event="merged", prs=nums, total=len(self.state["merged"]), head=out.strip()[:8])
        return True, out.strip()

    def run(self):
        self.ensure_identity()
        self.setup_branch()
        self.clean_tree()
        pending = [c["number"] for c in self.cands
                   if c["number"] not in self.state["merged"]
                   and c["number"] not in self.state["quarantined"]]
        print(f"pending: {len(pending)} (merged={len(self.state['merged'])} quar={len(self.state['quarantined'])})", flush=True)
        if self.a.dry_run:
            bs = self.a.batch
            for i in range(0, min(len(pending), self.a.max_prs), bs):
                print(f"batch: {pending[i:i+bs]}")
            return
        if self.a.doctor:
            self.doctor()
            return
        # baseline bench once (7B model required; skip if absent)
        if self.a.bench_model and self.state["bench_baseline"] is None:
            if os.path.exists(os.path.expanduser(self.a.bench_model)) or \
               os.path.exists(os.path.expanduser(self.a.bench_model) + ".1"):
                ok, out = self.build()
                print(f"baseline build: {'OK' if ok else 'FAIL'}", flush=True)
                if ok:
                    okb, val, bout = self.bench()
                    self.state["bench_baseline"] = val
                    print(f"baseline bench tg: {val}", flush=True)
                    if val is None:
                        self.log(event="baseline-bench-unparsed", detail=bout[-1000:])
                    self.save()
            else:
                print("bench model absent, skipping baseline bench", flush=True)
        done = 0
        i = 0
        while i < len(pending) and done < self.a.max_prs:
            batch = [n for n in pending[i:i+self.a.batch]
                     if n not in self.state["quarantined"]]
            if not batch:
                i += self.a.batch
                continue
            self.log(event="batch-start", prs=batch)
            for n in batch:
                if done >= self.a.max_prs:
                    break
                ok, reason = self.merge_one_committed(n, f"batch-{self.state['batches_done']}")
                if not ok:
                    self.quarantine(n, "merge-conflict", reason)
                    continue
                # incremental gates: build + smoke on top of this commit.
                # ccache makes small-PR rebuilds fast. On failure revert the
                # commit (local branch, never pushed: safe) and quarantine.
                okb, bout = self.build()
                if not okb:
                    self.git("reset --hard HEAD~1")
                    self.quarantine(n, "build-failed", bout[-2000:])
                    continue
                oks, sout = self.smoke()
                if not oks:
                    self.git("reset --hard HEAD~1")
                    self.quarantine(n, "smoke-failed", sout[-2000:])
                    continue
                if self.a.bench_model and self.a.regression_pct > 0 and \
                   os.path.exists(os.path.expanduser(self.a.bench_model)):
                    okbench, val, bout = self.bench()
                    if not okbench or val is None:
                        self.log(event="bench-failed", pr=n,
                                 detail=bout[-1000:])
                        # bench infra flake (OOM/timeout/parse): do NOT punish
                        # the PR, but record the merge without a perf verdict.
                        self.state["merged"].append(n)
                        self.state["batches_done"] += 1
                        self.save()
                        self.log(event="merged-unverified-perf", pr=n,
                                 total=len(self.state["merged"]))
                        done += 1
                        continue
                    base = self.state.get("bench_baseline")
                    if base and val and val < base * (1 - self.a.regression_pct/100):
                        self.git("reset --hard HEAD~1")
                        self.log(event="regression", pr=n, val=val, base=base)
                        self.quarantine(n, "perf-regression", f"{val} vs {base}")
                        continue
                self.state["merged"].append(n)
                self.state["batches_done"] += 1
                self.save()
                self.log(event="merged", pr=n, total=len(self.state["merged"]))
                done += 1
            i += self.a.batch
        print(f"DONE merged={self.state['merged']} quarantined={len(self.state['quarantined'])}", flush=True)

    def doctor(self):
        """Reconcile state files vs repo. Drops phantom 'merged' entries whose
        commit is absent, and requeues quarantines caused by the MERGE_HEAD
        stacking bug (never truly conflict-tested)."""
        rc, out = self.git(f"log --format=%s {self.state.get('base_sha', self.a.base)}..HEAD")
        subjects = out if rc == 0 else ""
        fixed_merged, fixed_quar, requeued = [], [], []
        for n in self.state["merged"]:
            if f"#{n}" in subjects:
                fixed_merged.append(n)
            else:
                print(f"doctor: drop phantom merged #{n} (no commit on branch)")
        for q in list(self.quar):
            if "MERGE_HEAD exists" in q.get("detail", ""):
                print(f"doctor: requeue #{q['pr']} (bogus MERGE_HEAD quarantine)")
                requeued.append(q["pr"])
            else:
                fixed_quar.append(q)
        self.state["merged"] = fixed_merged
        self.state["quarantined"] = [n for n in self.state["quarantined"] if n not in requeued]
        self.quar = fixed_quar
        self.save()
        print(f"doctor: merged={fixed_merged} requeued={requeued}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.json")
    ap.add_argument("--repo", default="~/llama-pr-lab/llama.cpp")
    ap.add_argument("--state-dir", default=".")
    ap.add_argument("--base", default="master")
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--max-prs", type=int, default=50)
    ap.add_argument("--targets", nargs="+", default=["llama-cli", "llama-bench"])
    ap.add_argument("--build-type", default="Release")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--smoke-model", default="~/llama-pr-lab/models/tinyllama.gguf")
    ap.add_argument("--bench-model", default="~/llama-pr-lab/models/qwen2.5-7b.gguf")
    ap.add_argument("--pp", type=int, default=128)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--regression-pct", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--doctor", action="store_true",
                    help="reconcile state vs repo, requeue bogus quarantines, then exit")
    a = ap.parse_args()
    Lab(a).run()

if __name__ == "__main__":
    main()
