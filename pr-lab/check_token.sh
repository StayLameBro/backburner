#!/usr/bin/env bash
# Validates the user-stored token (never carries the secret).
# Token lives at ~/.config/llama-pr-lab/token (mode 600), outside all repos.
f="$HOME/.config/llama-pr-lab/token"
test -s "$f" || { echo NO-FILE; exit 1; }
echo "file-bytes: $(wc -c < "$f")"
GH_TOKEN=$(cat "$f")
export GH_TOKEN
curl -s -H "Authorization: Bearer $GH_TOKEN" https://api.github.com/rate_limit \
  | python3 -c "import json,sys; d=json.load(sys.stdin); c=d['resources']['core']; print('limit:',c['limit'],'remaining:',c['remaining'])"
