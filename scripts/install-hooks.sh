#!/bin/bash
# install-hooks.sh - install the commit / push checks into this clone (every worktree and branch shares them):
#   pre-commit  staged changes: secrets, personal data, device ids, models / binaries, big files (scripts/hooks/check-sensitive.py),
#               and the phone app's network rules (scripts/check-listeners.sh)
#   commit-msg  the message: AI-assistant attribution lines, personal data
#   pre-push    every commit about to be published: all of the above, plus author / committer emails
# Your own private strings (name, personal emails, Apple team id, device UDIDs, device names): add them with
# `python3 scripts/hooks/check-sensitive.py --private-add`. They are stored only as keyed fingerprints in
# .git/info/private-hashes (never committed), with the key in your macOS Keychain, so no file holds them in plain text.
set -euo pipefail
cd "$(dirname "$0")/.."
HOOKS="$(git rev-parse --git-common-dir)/hooks"
mkdir -p "$HOOKS"
if [ -n "$(git config --get core.hooksPath || true)" ]; then
  echo "install-hooks: core.hooksPath is set ($(git config --get core.hooksPath)); these hooks would not run. Unset it first." >&2
  exit 1
fi
for f in check-sensitive.py pre-commit commit-msg pre-push; do
  install -m 0755 "scripts/hooks/$f" "$HOOKS/$f"
done
if [ -f "$(git rev-parse --git-common-dir)/info/private-patterns" ]; then
  python3 "$HOOKS/check-sensitive.py" --private-migrate   # an old plain-text list becomes fingerprints
fi
echo "install-hooks: pre-commit, commit-msg, pre-push installed in $HOOKS"
if [ "$(uname)" = Darwin ] && [ -t 1 ]; then
  echo "install-hooks: fingerprinting this Mac's names and your paired iPhones / iPads (devicectl: up to ~30 s the first time)"
  python3 "$HOOKS/check-sensitive.py" --private-add-devices || echo "install-hooks: that failed; run it later (below)"
fi
echo "install-hooks: add your private strings (name, emails, team id, device ids) with:"
echo "  python3 scripts/hooks/check-sensitive.py --private-add"
echo "  and after pairing a new iPhone / iPad: python3 scripts/hooks/check-sensitive.py --private-add-devices"
echo "  They are kept only as fingerprints in .git/info/private-hashes; the key is in your Keychain."
