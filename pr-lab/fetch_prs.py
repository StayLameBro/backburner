#!/usr/bin/env python3
"""Fetch open llama.cpp PRs and triage to perf-relevant merge candidates.

Stage 1 (cheap, 1 API call per 100 PRs): list endpoint only, keyword score.
Stage 2 (expensive, 2 calls per PR): detail + files, only for top candidates.

Usage (in WSL):
  GH_TOKEN=xxx python3 fetch_prs.py --limit 1000 --top 100 --out candidates.json
  python3 fetch_prs.py --limit 200 --top 50 --out candidates.json   # unauth (60 req/hr)

Output: candidates.json sorted by score desc, each with number/title/score/reasons.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

API = "https://api.github.com/repos/ggml-org/llama.cpp"

def req(url, token):
    h = {"Accept": "application/vnd.github+json", "User-Agent": "llama-pr-lab"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    r = urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(r) as resp:
            return json.load(resp), resp.headers
    except urllib.error.HTTPError as e:
        if e.code == 403 and "rate" in str(e.headers).lower():
            reset = e.headers.get("X-RateLimit-Reset", "?")
            print(f"RATE LIMITED. Reset at {reset}. Set GH_TOKEN to raise limit.", file=sys.stderr)
        raise

def score_title(title, body, cfg):
    t = ((title or "") + " " + (body or "")).lower()
    hits = [k for k in cfg["perf_keywords"] if k.lower() in t]
    return len(hits), hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--top", type=int, default=100, help="how many go to stage-2 detail fetch")
    ap.add_argument("--out", default="candidates.json")
    ap.add_argument("--config", default="config.json")
    a = ap.parse_args()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    with open(os.path.join(os.path.dirname(__file__), a.config)) as f:
        cfg = json.load(f)

    all_prs = []
    page = 1
    while len(all_prs) < a.limit:
        url = f"{API}/pulls?state=open&per_page=100&page={page}"
        data, _ = req(url, token)
        if not data:
            break
        all_prs.extend(data)
        print(f"page {page}: got {len(data)} (total {len(all_prs)})")
        page += 1
        if len(data) < 100:
            break
        time.sleep(1 if token else 3)

    print(f"total open fetched: {len(all_prs)}", file=sys.stderr)

    staged = []
    for pr in all_prs:
        labels = [l["name"].lower() for l in pr.get("labels", [])]
        if any(x in labels for x in cfg["exclude_labels"]):
            continue
        if pr.get("draft"):
            continue  # skip drafts in auto-merge; review manually later
        s, hits = score_title(pr.get("title"), pr.get("body"), cfg)
        staged.append({"number": pr["number"], "title": pr["title"],
                       "user": pr["user"]["login"], "updated": pr["updated_at"],
                       "labels": labels, "score": s, "hits": hits,
                       "head": pr["head"]["sha"][:8]})
    staged.sort(key=lambda x: (-x["score"], x["updated"]))
    print(f"stage-1 survivors: {len(staged)} (drafts/excluded removed)")

    # Stage 2: enrich top-N with mergeable + files
    top = staged[:a.top]
    for i, c in enumerate(top):
        n = c["number"]
        try:
            detail, _ = req(f"{API}/pulls/{n}", token)
            c["mergeable"] = detail.get("mergeable")
            c["mergeable_state"] = detail.get("mergeable_state")
            c["additions"] = detail.get("additions"); c["deletions"] = detail.get("deletions")
            c["changed_files"] = detail.get("changed_files")
            files, _ = req(f"{API}/pulls/{n}/files?per_page=100", token)
            paths = [f["filename"] for f in files]
            c["files"] = paths[:20]
            c["touches_perf_path"] = any(p.startswith(pp.rstrip("/")) or pp.strip("/") in p for p in paths for pp in cfg["perf_paths"])
            if c["touches_perf_path"]:
                c["score"] += 2
                c["hits"] = c["hits"] + ["touches-perf-path"]
            if (c["additions"] or 0) + (c["deletions"] or 0) > 2000:
                c["score"] -= 1
                c["hits"] = c["hits"] + ["large-diff-penalty"]
            print(f"[{i+1}/{len(top)}] #{n} mergeable={c['mergeable_state']} files={c['changed_files']} score={c['score']}")
        except Exception as e:
            c["error"] = str(e)
            print(f"[{i+1}/{len(top)}] #{n} ERROR {e}")
        time.sleep(0.7 if token else 2.5)

    top.sort(key=lambda x: (-x["score"], x["updated"]))
    with open(a.out, "w") as f:
        json.dump(top, f, indent=2)
    print(f"wrote {a.out} with {len(top)} candidates")
    print("Top 15:")
    for c in top[:15]:
        print(f"  #{c['number']} s={c['score']} {c.get('mergeable_state')} {c['title'][:70]}")

if __name__ == "__main__":
    main()
