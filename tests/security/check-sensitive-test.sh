#!/bin/bash
# check-sensitive-test.sh - the commit / push hooks (scripts/hooks, scripts/install-hooks.sh) in a throwaway repo: every rule
# must block, a clean commit must pass, and pre-push must catch what a --no-verify commit let in. Exits non-zero on failure.
# Fake secrets are assembled from pieces so this file passes the scanner itself.
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)
T=$(mktemp -d "${TMPDIR:-/tmp}/bb-hooks.XXXXXX")
trap 'rm -rf "$T"' EXIT
pass=0; fail=0
ok()  { pass=$((pass + 1)); }
bad() { fail=$((fail + 1)); echo "FAIL $*" >&2; }

cd "$T"
git init -q repo && cd repo
git config user.name "Test" && git config user.email "1+test@users.noreply.github.com"
mkdir -p scripts/hooks && cp "$SRC"/scripts/hooks/* scripts/hooks/ && cp "$SRC/scripts/install-hooks.sh" scripts/
scripts/install-hooks.sh >/dev/null || bad "install-hooks"
# private strings: fingerprints only (a test key instead of the Keychain), and the file must not hold them in plain text
export BB_PRIVATE_KEY_HEX=$(printf '11%.0s' {1..32})
printf 'Zebulon Quarry\nquarry@mail.invalid\n' | python3 scripts/hooks/check-sensitive.py --private-add >/dev/null
PRIV="$(git rev-parse --git-common-dir)/info/private-hashes"
[ -f "$PRIV" ] && [ "$(stat -f %Lp "$PRIV")" = 600 ] && ok || bad "private-hashes created 600"
grep -qi 'zebulon\|quarry' "$PRIV" && bad "private-hashes holds plain text" || ok
[ "$(wc -l < "$PRIV" | tr -d ' ')" = 2 ] && ok || bad "two fingerprints stored"
echo "hello" > README && git add README && git commit -qm "first" && ok || bad "a clean commit is allowed"

# expect_block NAME FILE CONTENT  - committing CONTENT in FILE must fail
expect_block() {
  printf '%s\n' "$3" > "$2"; git add -- "$2"
  if git commit -qm "x" >/dev/null 2>&1; then bad "$1 was committed"; git reset -q --hard HEAD~1; else ok; fi
  git reset -q; rm -f -- "$2"
}
US="/Us""ers"
expect_block "a GitHub token"        a.txt "token = gh""p_$(printf 'a%.0s' {1..36})"
expect_block "a private key"         b.txt "-----BEGIN OPENSSH PRIV""ATE KEY-----"
expect_block "a Hugging Face token"  c.txt "HF=hf_""$(printf 'b%.0s' {1..34})"
expect_block "an Anthropic key"      d.txt "sk-""ant-api03-$(printf 'c%.0s' {1..30})"
expect_block "a password assignment" e.py  "password = '""hunter2hunter2'"
expect_block "a home path"           f.sh  "cd $US/zebulon/Projects"
expect_block "a device UDID"         g.md  "UDID=0000""8140-0123456789ABCDEF"
expect_block "a team id"             h.txt "DEVELOPMENT_TEAM = ""ABCDE12345"
expect_block "a device name"         i.md  "Zeb$(printf "%s" "'")s iPhone is plugged in"
expect_block "a personal email"      j.md  "mail zeb@""personal-mail.net"
expect_block "a LAN address"         k.md  "phone at 192.168.""1.15"
expect_block "a link-local address"  l.md  "phone at 169.254.""1.3"
expect_block "a private name"        m.md  "written by Zebulon  Quarry"
expect_block "a private email"       m2.md "contact: quarry@mail.invalid."
expect_block "a private file name"   zebulon_quarry.md "x"
expect_block "a model file"          n.gguf "GGUF"
expect_block "a signing file"        o.mobileprovision "x"
expect_block "a .env file"           .env  "X=1"
head -c 6000000 /dev/zero > big.dat; git add big.dat
git commit -qm big >/dev/null 2>&1 && { bad "a 6 MB file was committed"; git reset -q --hard HEAD~1; } || ok
git reset -q; rm -f big.dat
ln -s "$US/zebulon/secret" link; git add link
git commit -qm link >/dev/null 2>&1 && { bad "a symlink out of the repo was committed"; git reset -q --hard HEAD~1; } || ok
git reset -q; rm -f link

# allowed forms
printf '%s\n' "phone at 169.254.x.y (sensitive-ok: placeholder)" "cd ~/Projects" "ping 169.254.255.255" "mail noreply@""example.com" > ok.md
git add ok.md && git commit -qm "placeholders" >/dev/null 2>&1 && ok || bad "placeholders and ~ paths are allowed"

# commit messages
echo a > p.txt && git add p.txt
git commit -qm "$(printf 'fix\n\nCo-Authored-By: Cl''aude <noreply@anthropic.com>')" >/dev/null 2>&1 && { bad "an attribution line was committed"; git reset -q --hard HEAD~1; } || ok
git commit -qm "$(printf 'fix\n\nClaude''-Session: https://example.com/x')" >/dev/null 2>&1 && { bad "a session line was committed"; git reset -q --hard HEAD~1; } || ok
git commit -qm "fix p" >/dev/null 2>&1 && ok || bad "a plain message is allowed"

# author email
git config user.email "zeb@""personal-mail.net"
echo b > q.txt && git add q.txt
git commit -qm "q" >/dev/null 2>&1 && { bad "a personal author email was committed"; git reset -q --hard HEAD~1; } || ok
git config --add backburner.allowedEmail "zeb@""personal-mail.net"
git commit -qm "q" >/dev/null 2>&1 && ok || bad "an allowed email is accepted"
git config --unset-all backburner.allowedEmail; git config user.email "1+test@users.noreply.github.com"
git reset -q --hard HEAD~1   # (pre-push would rightly refuse that commit now that the address is no longer allowed)

# the listener lint runs in pre-commit when the checkout has it
mkdir -p scripts && printf '#!/bin/bash\nexit 1\n' > scripts/check-listeners.sh && chmod +x scripts/check-listeners.sh
echo c > r.txt && git add r.txt scripts/check-listeners.sh
git commit -qm "r" >/dev/null 2>&1 && { bad "pre-commit ignored a failing check-listeners.sh"; git reset -q --hard HEAD~1; } || ok
git reset -q; rm -f scripts/check-listeners.sh r.txt

# pre-push catches what --no-verify let in
git init -q --bare ../remote.git && git remote add origin ../remote.git
git push -q origin HEAD:main >/dev/null 2>&1 && ok || bad "a clean push is allowed"
printf '%s\n' "cd $US/zebulon/x" > s.sh && git add s.sh && git commit -q --no-verify -m "sneak" >/dev/null 2>&1
git push -q origin HEAD:main >/dev/null 2>&1 && bad "pre-push let a home path through" || ok
git reset -q --hard HEAD~1
git commit -q --no-verify --allow-empty -m "$(printf 'x\n\nCo-Authored-By: Cl''aude <noreply@anthropic.com>')" >/dev/null 2>&1
git push -q origin HEAD:main >/dev/null 2>&1 && bad "pre-push let an attribution line through" || ok
git reset -q --hard HEAD~1
git -c user.email="zeb@""personal-mail.net" commit -q --no-verify --allow-empty -m "y" >/dev/null 2>&1
git push -q origin HEAD:main >/dev/null 2>&1 && bad "pre-push let a personal author email through" || ok
git reset -q --hard HEAD~1
git push -q origin HEAD:feature >/dev/null 2>&1 && ok || bad "a clean new branch push is allowed"

# issue #8: a branch rebased onto the published main passes even though main holds commits these rules would refuse
# (history from before them, GitHub's merge commits); new commits on top are still checked, and an unpublished remote's
# history is not trusted. url.insteadOf points the real URLs at local repos, so the hook sees the URLs.
ATTR="$(printf 'old\n\nCo-Authored-By: Cl''aude <noreply@anthropic.com>')"
base=$(git rev-parse HEAD)
git switch -q -c pubhist "$base"
git -c user.email="zeb@""personal-mail.net" commit -q --no-verify --allow-empty -m "$ATTR" >/dev/null 2>&1
git switch -q -c side "$base" && git commit -q --no-verify --allow-empty -m side >/dev/null 2>&1 && git switch -q pubhist
GIT_AUTHOR_EMAIL="zeb@""personal-mail.net" GIT_COMMITTER_NAME=GitHub GIT_COMMITTER_EMAIL="noreply@github.com" \
  git merge -q --no-ff --no-verify -m "Merge pull request #7" side >/dev/null 2>&1
git init -q --bare ../pub.git && git -C ../pub.git fetch -q "$T/repo" pubhist:main
git config url."$T/pub.git".insteadOf "https://github.com/StayLameBro/backburner.git"
git remote add upstream "https://github.com/StayLameBro/backburner.git" && git fetch -q upstream
git switch -q -c work "$base" && echo w > w.txt && git add w.txt && git commit -qm work >/dev/null 2>&1
git push -q origin work >/dev/null 2>&1 && ok || bad "a clean new branch (work) is allowed"
git rebase -q upstream/main >/dev/null 2>&1
git push -q --force origin work >/dev/null 2>&1 && ok || bad "a branch rebased onto the published main was refused (issue #8)"
git commit -q --no-verify --allow-empty -m "$ATTR" >/dev/null 2>&1
git push -q --force origin work >/dev/null 2>&1 && bad "pre-push let an attribution line through on top of the published main" || ok
git reset -q --hard HEAD~1
GIT_COMMITTER_NAME=GitHub GIT_COMMITTER_EMAIL="noreply@github.com" git commit -q --no-verify --allow-empty -m "web edit" >/dev/null 2>&1
git push -q --force origin work >/dev/null 2>&1 && ok || bad "GitHub's committer address was refused"
GIT_AUTHOR_EMAIL="noreply@github.com" git commit -q --no-verify --allow-empty -m "odd author" >/dev/null 2>&1
git push -q --force origin work >/dev/null 2>&1 && bad "GitHub's address was accepted as an author" || ok
git reset -q --hard HEAD~1
git switch -q -c otherhist "$base"
git -c user.email="zeb@""personal-mail.net" commit -q --no-verify --allow-empty -m other >/dev/null 2>&1
git init -q --bare ../other.git && git -C ../other.git fetch -q "$T/repo" otherhist:main
git config url."$T/other.git".insteadOf "https://github.com/someone-else/backburner.git"
git remote add fork "https://github.com/someone-else/backburner.git" && git fetch -q fork
git switch -q work && git reset -q --hard fork/main
git push -q --force origin work >/dev/null 2>&1 && bad "an unpublished remote's history was trusted" || ok

echo "check-sensitive-test: $pass passed, $fail failed"
[ $fail = 0 ]
