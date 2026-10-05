#!/usr/bin/env python3
"""check-sensitive.py - stop personal data, secrets and stray files from entering a commit or a push. Installed into the git
hooks (pre-commit, commit-msg, pre-push) by scripts/install-hooks.sh; tests: tests/security/check-sensitive-test.sh.

  check-sensitive.py --staged            the staged changes (pre-commit)
  check-sensitive.py --msg FILE          a commit message (commit-msg)
  check-sensitive.py --push              the commits a push would publish (pre-push; reads git's stdin)
  check-sensitive.py --all               every tracked file at HEAD (an audit)

What it refuses (exit 1, with file:line and the rule; the matched text is masked):
  secret        private keys, GitHub / Hugging Face / Anthropic / OpenAI / AWS / Slack / Google tokens, password assignments
  private       your own private strings (name, emails, team id, UDIDs, ...), kept only as keyed fingerprints in
                .git/info/private-hashes with the key in your Keychain: add them with --private-add
  home-path     /Users/<name>/ or /home/<name>/ (use ~ or $HOME)
  device-id     Apple device UDIDs / USB serials (00008xxx-...)
  team-id       a filled-in DEVELOPMENT_TEAM, or a bundle id with a team id appended
  device-name   "<name>'s iPhone / iPad / MacBook"
  email         any address except GitHub noreply, example.com/.org/.net, the .example/.invalid/.test TLDs, noreply@
                and localhost
  private-ip    a concrete LAN / link-local address (192.168.x.y, 10.x.y.z, 172.16-31.x.y, 169.254.x.y); write 169.254.x.y
  file          models and binaries (.gguf .safetensors .ipa .mlmodelc .xcarchive .o .a .dylib ...), signing material
                (.p12 .p8 .cer .mobileprovision .pem .key), .env, .DS_Store, files over 5 MB, symlinks leaving the repo
  attribution   Co-Authored-By / session / "Generated with" lines naming an AI assistant, in commit messages
  author        a commit whose author or committer email is not a GitHub noreply address (or one allowed with
                `git config --add backburner.allowedEmail ADDRESS`); GitHub's own committer address is accepted
A push skips commits already on a published project's main branch (PUBLISHED below): they are public, and a branch rebased
onto main would otherwise be refused for history from before these rules or for GitHub's merge commits (issue #8).
A line containing "sensitive-ok" is exempt from home-path, email, private-ip and device-name (say why on that line);
nothing exempts secret, private, device-id or team-id. Bypassing the hooks (--no-verify) defeats all of this: don't.
"""
import os, re, subprocess, sys

MAX_BYTES = 5 * 1024 * 1024

# patterns are assembled from pieces so this file never matches itself
def rx(*parts, flags=0):
    return re.compile("".join(parts), flags)

SECRET = [
    rx("-----BEGIN [A-Z ]*PRIV", "ATE KEY-----"),
    rx(r"\bgh[pousr]_", r"[A-Za-z0-9]{36,}\b"),
    rx(r"\bgithub_", r"pat_[A-Za-z0-9_]{40,}\b"),
    rx(r"\bhf_", r"[A-Za-z]{34,}\b"),
    rx(r"\bsk-", r"ant-[A-Za-z0-9_-]{20,}"),
    # OpenAI: project / service-account / admin keys put - and _ in the body;
    # legacy sk- keys stay alphanumeric-only so kebab-case text like sk-learn-… does not match.
    rx(r"\bsk-", r"(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}"),
    rx(r"\bsk-", r"(?!proj-|svcacct-|admin-)[A-Za-z0-9]{32,}\b"),
    rx(r"\b(AKIA|ASIA)", r"[0-9A-Z]{16}\b"),
    rx(r"\bxox", r"[baprs]-[A-Za-z0-9-]{10,}"),
    rx(r"\bAIza", r"[0-9A-Za-z_-]{35}\b"),
    rx(r"(?i)\b(pass(word|wd)?|secret|api[_-]?key|token)\s*[:=]\s*['\"][^'\"\s$]{8,}['\"]"),
]
HOME = rx(r"(/Users|/home)/(?!(you|me|user|USER|runner|Shared|name|NAME|example|\$|\{|<|\*|\.\.\.)[/\s\"'`:)]?)[A-Za-z0-9._-]+/")
DEVICE_ID = [rx(r"\b0000", r"[0-9A-F]{4}-[0-9A-F]{16}\b"), rx(r"\b0000", r"8[0-9A-F]{3}[0-9A-F]{16}\b")]
TEAM = [rx(r"DEVELOPMENT_TEAM\s*=\s*\"?", r"[A-Z0-9]{10}\b"), rx(r"\bapp\.backburner\.", r"[A-Z0-9]{10}\b"),
        rx(r"\bTeamIdentifier\b")]
DEVICE_NAME = rx(r"[A-Za-z]+(['’])s (iPhone|iPad|MacBook|Mac mini|iMac|Mac Studio)\b")
EMAIL = rx(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
EMAIL_OK = rx(r"^(.*@users\.noreply\.github\.com|.*@example\.(com|org|net)|.*\.(example|invalid|test)|noreply@.*|git@github\.com|.*@localhost)$",
              flags=re.I)
PRIVATE_IP = rx(r"\b(192\.168|10\.\d{1,3}|172\.(1[6-9]|2\d|3[01])|169\.254)\.\d{1,3}\.\d{1,3}\b")
ATTRIBUTION = rx(r"(?im)^\s*(co-authored-by:.*\b(claude|anthropic|copilot|chatgpt|openai|gemini|cursor)\b"
                 r"|claude-session:|.*generated with \[?(claude|copilot|cursor))")
BAD_EXT = (".gguf", ".pyc", ".safetensors", ".ipa", ".xcarchive", ".dsym", ".mlmodelc", ".mlpackage", ".o", ".a", ".dylib", ".so",
           ".p12", ".p8", ".cer", ".mobileprovision", ".provisionprofile", ".keychain", ".keychain-db", ".pem", ".key",
           ".env", ".sqlite", ".db", ".npy", ".pt", ".bin", ".zip", ".tar", ".gz", ".tgz")
BAD_NAME = (".ds_store", "id_rsa", "id_ed25519", ".netrc", "credentials", ".env")


def git(*a, input=None):
    return subprocess.run(["git", *a], capture_output=True, text=True, input=input, errors="replace").stdout


# ---- private strings, stored as keyed fingerprints -------------------------------------------------------------------
# .git/info/private-hashes holds one line per private string: "N:HEX", HEX = HMAC-SHA256(key, the string's normalized
# tokens joined by spaces), N = its token count. The key is 32 random bytes in your macOS login Keychain (service
# "backburner-private-patterns"), so the file alone reveals nothing: no names, emails or ids, only fingerprints. Scanning
# fingerprints every token of a line (and its pieces split at . @ _ + - %, and runs of up to 4 tokens) and compares.
#   check-sensitive.py --private-add      add strings (typed or piped, one per line; never echoed or stored in plain text)
#   check-sensitive.py --private-count    how many fingerprints there are
KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT = "backburner-private-patterns", "hmac-key"
TOKEN = re.compile(r"[a-z0-9@._+%-]+")
PIECE = re.compile(r"[a-z0-9]+")


def git_dir():
    d = git("rev-parse", "--git-common-dir").strip() or ".git"
    return os.path.abspath(d)


def normalize(text):
    return text.lower().replace("\u2019", "'")


def private_key(create=False):
    hx = os.environ.get("BB_PRIVATE_KEY_HEX")   # tests only (tests/security/check-sensitive-test.sh); never set it for real use
    if hx:
        return bytes.fromhex(hx)
    r = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT, "-w"],
                       capture_output=True, text=True)
    if r.returncode == 0 and len(r.stdout.strip()) == 64:
        return bytes.fromhex(r.stdout.strip())
    if not create:
        return None
    key = os.urandom(32)
    r = subprocess.run(["security", "add-generic-password", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT, "-w", key.hex()],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("check-sensitive: could not store a key in the Keychain: " + r.stderr.strip())
    return key


def fingerprint(key, tokens):
    import hmac, hashlib
    return hmac.new(key, " ".join(tokens).encode(), hashlib.sha256).hexdigest()


def load_private():
    path = os.path.join(git_dir(), "info", "private-hashes")
    want = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                n, h = line.split(":", 1)
                if n.isdigit():
                    want.setdefault(int(n), set()).add(h)
    if os.path.exists(os.path.join(git_dir(), "info", "private-patterns")):
        print("check-sensitive: .git/info/private-patterns is plain text; run check-sensitive.py --private-migrate", file=sys.stderr)
    if not want:
        return None, {}
    key = private_key()
    if key is None:
        print("check-sensitive: no Keychain key for the private fingerprints; private strings are NOT being checked", file=sys.stderr)
        return None, {}
    return key, want


PRIV_KEY, PRIV = load_private()


def private_hit(text):
    """the span of the first private string in `text`, or None"""
    if not PRIV:
        return None
    low = normalize(text)
    toks = [(m.group(0), m.start(), m.end()) for m in TOKEN.finditer(low)]
    single = PRIV.get(1, set())
    for t, a, b in toks:
        cands = {t, t.strip("._-%+@")} | {m.group(0) for m in PIECE.finditer(t)}
        if any(fingerprint(PRIV_KEY, [c]) in single for c in cands):
            return (a, b)
    # multi-token strings ("first last"): compare runs of plain pieces
    pieces = [(m.group(0), m.start(), m.end()) for m in PIECE.finditer(low)]
    for n, hs in PRIV.items():
        if n < 2:
            continue
        for i in range(len(pieces) - n + 1):
            run = pieces[i:i + n]
            if fingerprint(PRIV_KEY, [x[0] for x in run]) in hs:
                return (run[0][1], run[-1][2])
    return None


def entry_tokens(text):
    """a private string as stored: one token if it is one (an email, a UDID, a name), else its plain pieces"""
    low = normalize(text.strip())
    t = TOKEN.findall(low)
    return t if len(t) == 1 else PIECE.findall(low)


def private_add(lines):
    key = private_key(create=True)
    path = os.path.join(git_dir(), "info", "private-hashes")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    have = set(open(path).read().split()) if os.path.exists(path) else set()
    added = 0
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    for line in lines:
        if not line.strip() or line.strip().startswith("#"):
            continue
        toks = entry_tokens(line)
        if not toks or len(toks) > 4 or len(" ".join(toks)) < 3:
            print("check-sensitive: skipped an entry (empty, too short, or more than 4 words)", file=sys.stderr)
            continue
        rec = f"{len(toks)}:{fingerprint(key, toks)}"
        if rec not in have:
            os.write(fd, (rec + "\n").encode()); have.add(rec); added += 1
    os.close(fd)
    os.chmod(path, 0o600)
    print(f"check-sensitive: {added} fingerprint(s) added ({len(have)} in total) in {path}")


problems = []


def mask(line, span):
    a, b = span
    return (line[:a] + "*" * min(8, b - a) + line[b:]).strip()[:160]


def scan_line(where, line):
    exempt = "sensitive-ok" in line
    hit = private_hit(line)
    if hit:
        problems.append((where, "private", mask(line, hit)))
        return
    for r in SECRET:
        m = r.search(line)
        if m:
            problems.append((where, "secret", mask(line, m.span()))); return
    for r in DEVICE_ID:
        m = r.search(line)
        if m:
            problems.append((where, "device-id", mask(line, m.span()))); return
    for r in TEAM:
        m = r.search(line)
        if m:
            problems.append((where, "team-id", mask(line, m.span()))); return
    if exempt:
        return
    m = HOME.search(line)
    if m:
        problems.append((where, "home-path", mask(line, m.span()))); return
    m = DEVICE_NAME.search(line)
    if m:
        problems.append((where, "device-name", mask(line, m.span()))); return
    for m in EMAIL.finditer(line):
        if not EMAIL_OK.search(m.group(0)):
            problems.append((where, "email", mask(line, m.span()))); return
    for m in PRIVATE_IP.finditer(line):
        if m.group(0).split(".")[-1] in ("0", "255"):   # network / broadcast forms (169.254.0.0, 169.254.255.255) name no device
            continue
        problems.append((where, "private-ip", mask(line, m.span()))); return


def check_path(path, size=None, mode=None, link_target=None):
    low = path.lower()
    base = os.path.basename(low)
    if low.endswith(BAD_EXT) or any(s + "/" in low + "/" for s in (".xcarchive", ".dsym", ".mlmodelc", ".mlpackage")):
        problems.append((path, "file", "a model, binary, archive or signing file"))
    elif base in BAD_NAME or base.startswith(".env"):
        problems.append((path, "file", "a credentials / system file"))
    if size is not None and size > MAX_BYTES:
        problems.append((path, "file", f"{size / 1048576:.1f} MB (over {MAX_BYTES // 1048576} MB)"))
    if mode == "120000" and link_target is not None and (link_target.startswith("/") or link_target.startswith("~")
                                                         or ".." in link_target.split("/")):
        problems.append((path, "file", "a symlink that leaves the repository"))
    if private_hit(path):
        problems.append((path, "private", "the file name"))


def scan_diff(diff):
    """added lines of a unified diff (-U0)"""
    path, ln = None, 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            ln = int(m.group(1)) if m else 0
        elif line.startswith("+") and path:
            scan_line(f"{path}:{ln}", line[1:])
            ln += 1


def emails_ok(email):
    allowed = [e.strip().lower() for e in git("config", "--get-all", "backburner.allowedEmail").splitlines() if e.strip()]
    return email.lower().endswith("@users.noreply.github.com") or email.lower() in allowed


# GitHub's web merges and edits are committed as "GitHub <noreply@github.com>": an address that names no person
GITHUB_COMMITTER = "noreply@github.com"


def check_identity(who, email):
    if not emails_ok(email):
        problems.append((who, "author", "<" + email.split("@")[0][:2] + "***@" + email.split("@")[-1] + "> is not a GitHub noreply "
                         "address (git config user.email <id>+<user>@users.noreply.github.com, or allow it: "
                         "git config --add backburner.allowedEmail ADDRESS)"))


def staged():
    entries = git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR").split("\0")
    for p in filter(None, entries):
        ls = git("ls-files", "-s", "--", p).split()
        mode, blob = (ls[0], ls[1]) if len(ls) >= 2 else (None, None)
        size = int(git("cat-file", "-s", blob).strip() or 0) if blob else None
        target = git("cat-file", "-p", blob) if mode == "120000" else None
        check_path(p, size, mode, target)
    scan_diff(git("diff", "--cached", "-U0", "--no-color", "--no-ext-diff", "--diff-filter=ACMR"))
    ident = git("var", "GIT_AUTHOR_IDENT")
    m = re.search(r"<([^>]*)>", ident)
    if m:
        check_identity("author", m.group(1))
    ident = git("var", "GIT_COMMITTER_IDENT")
    m = re.search(r"<([^>]*)>", ident)
    if m:
        check_identity("committer", m.group(1))


def message(text, where="commit message"):
    body = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
    m = ATTRIBUTION.search(body)
    if m:
        problems.append((where, "attribution", m.group(0).strip()[:100]))
    for i, line in enumerate(body.splitlines(), 1):
        scan_line(f"{where}:{i}", line)


# Repos whose main branch is public history. A remote counts only by its URL (never its name), and only its main/master
# branch: commits there were checked when they were merged, or came from upstream llama.cpp. Add one with
# `git config --add backburner.publishedRepo URL`.
PUBLISHED = ("github.com/staylamebro/backburner", "github.com/staylamebro/backburner-llama.cpp", "github.com/ggml-org/llama.cpp")


def repo_id(url):
    u = re.sub(r"^[a-z][a-z0-9+.-]*://", "", url.strip().lower())   # https:// ssh://
    u = re.sub(r"^[^@/]+@", "", u)                                    # git@
    u = re.sub(r"^([^/:]+):(?!\d+/)", r"\1/", u)                      # git@github.com:owner/repo
    return re.sub(r"(\.git)?/*$", "", u)


def published_refs():
    ids = set(PUBLISHED) | {repo_id(u) for u in git("config", "--get-all", "backburner.publishedRepo").splitlines() if u.strip()}
    refs = []
    for line in git("config", "--get-regexp", r"^remote\..*\.url$").splitlines():
        key, _, url = line.partition(" ")
        if repo_id(url) not in ids:
            continue
        name = key[len("remote."):-len(".url")]
        for branch in ("main", "master"):
            ref = f"refs/remotes/{name}/{branch}"
            if git("rev-parse", "--verify", "-q", ref + "^{commit}").strip():
                refs.append(ref)
    return refs


def push():
    zero = "0" * 40
    published = published_refs()
    for line in sys.stdin.read().splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[1] == zero:
            continue
        local, remote = parts[1], parts[3]
        rng = [local, "--not", "--remotes", *published] if remote == zero else [local, "--not", remote, *published]
        for sha in filter(None, git("rev-list", *rng).split()):
            short = sha[:9]
            message(git("log", "-1", "--format=%B", sha), f"commit {short} message")
            ae, ce = git("log", "-1", "--format=%ae%n%ce", sha).split("\n")[:2]
            check_identity(f"commit {short} author", ae)
            if ce.lower() != GITHUB_COMMITTER:
                check_identity(f"commit {short} committer", ce)
            for row in filter(None, git("diff-tree", "-r", "-z", "--no-commit-id", "--diff-filter=ACMR", "--root", sha).split("\0:")):
                f = row.lstrip(":").split("\0")
                if len(f) < 2:
                    continue
                meta, p = f[0].split(), f[1]
                mode, blob = meta[1], meta[3]
                size = int(git("cat-file", "-s", blob).strip() or 0)
                target = git("cat-file", "-p", blob) if mode == "120000" else None
                check_path(p, size, mode, target)
            diff = git("show", "--format=", "-U0", "--no-color", "--no-ext-diff", "--diff-filter=ACMR", sha)
            n0 = len(problems)
            scan_diff(diff)
            problems[n0:] = [(f"commit {short} {w}", r, d) for w, r, d in problems[n0:]]


def audit():
    for p in filter(None, git("ls-files", "-z").split("\0")):
        ls = git("ls-files", "-s", "--", p).split()
        mode = ls[0] if ls else None
        target = os.readlink(p) if mode == "120000" and os.path.islink(p) else None
        size = os.path.getsize(p) if os.path.isfile(p) else None
        check_path(p, size, mode, target)
        if mode == "120000" or not os.path.isfile(p) or (size or 0) > MAX_BYTES:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    scan_line(f"{p}:{i}", line)
        except UnicodeDecodeError:
            pass


def main():
    a = sys.argv[1:]
    if a[:1] == ["--staged"]:
        staged()
    elif a[:1] == ["--msg"] and len(a) == 2:
        message(open(a[1], encoding="utf-8", errors="replace").read())
    elif a[:1] == ["--push"]:
        push()
    elif a[:1] == ["--all"]:
        audit()
    elif a[:1] == ["--private-add"]:
        if sys.stdin.isatty():
            import getpass
            print("Type private strings, one per line (not shown); an empty line ends.", file=sys.stderr)
            lines = []
            while True:
                v = getpass.getpass("> ")
                if not v:
                    break
                lines.append(v)
        else:
            lines = sys.stdin.read().splitlines()
        private_add(lines)
        return
    elif a[:1] == ["--private-migrate"]:
        # turn a plain-text .git/info/private-patterns into fingerprints, then overwrite and delete the plain file
        old = os.path.join(git_dir(), "info", "private-patterns")
        if not os.path.exists(old):
            sys.exit("check-sensitive: no plain-text private-patterns to migrate")
        lines = [l for l in open(old, encoding="utf-8", errors="replace").read().splitlines() if l.strip() and not l.startswith("#")]
        private_add(lines)
        with open(old, "r+b") as f:
            f.write(b"\0" * os.path.getsize(old)); f.flush(); os.fsync(f.fileno())
        os.remove(old)
        print("check-sensitive: the plain-text file was overwritten and deleted")
        return
    elif a[:1] == ["--private-count"]:
        print(sum(len(v) for v in PRIV.values()), "private fingerprints;", "checking" if PRIV else "not checking")
        return
    else:
        sys.exit(__doc__)
    if problems:
        print(f"check-sensitive: {len(problems)} problem(s); nothing was committed or pushed:", file=sys.stderr)
        for where, rule, detail in problems[:60]:
            print(f"  [{rule}] {where}: {detail}", file=sys.stderr)
        if len(problems) > 60:
            print(f"  ... and {len(problems) - 60} more", file=sys.stderr)
        print("Fix them (scripts/hooks/check-sensitive.py --help explains each rule). Do not bypass with --no-verify.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
