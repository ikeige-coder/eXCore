#!/usr/bin/env bash
# Fetch the unquantized source GGUF and verify it against configs/sources.lock.json.
#
#   scripts/download_models.sh --url URL --out DIR [--name FILE] [--lock --repo R --revision V]
#
# Without --lock the file must match the pinned SHA-256 (a mismatch deletes nothing but fails).
# With --lock (first-time operator setup) the downloaded file's hash is recorded in the lock.
# Set HF_TOKEN for gated Hugging Face repositories. Downloads resume if interrupted.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL=""; OUT=""; NAME=""; LOCK=0; REPO=""; REVISION=""; LLAMA_COMMIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)      URL="${2:?}"; shift 2 ;;
    --out)      OUT="${2:?}"; shift 2 ;;
    --name)     NAME="${2:?}"; shift 2 ;;
    --lock)     LOCK=1; shift ;;
    --repo)     REPO="${2:?}"; shift 2 ;;
    --revision) REVISION="${2:?}"; shift 2 ;;
    --llama-cpp-commit) LLAMA_COMMIT="${2:?}"; shift 2 ;;
    -h|--help)  sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$OUT" ]] || { echo "--out is required" >&2; exit 2; }
mkdir -p "$OUT"

EXCORE=(python3 -m excore --config "$ROOT/configs/hpc_cpu.yaml")
LOCKFILE="$ROOT/configs/sources.lock.json"

if [[ -z "$NAME" ]]; then
  NAME="$(python3 - "$LOCKFILE" <<'PY'
import json, sys
files = json.load(open(sys.argv[1]))["sources"]["base"]["files"]
print(files[0]["name"] if files else "")
PY
)"
fi
[[ -n "$NAME" ]] || { [[ -n "$URL" ]] && NAME="$(basename "${URL%%\?*}")"; }
[[ -n "$NAME" ]] || { echo "cannot infer the file name; pass --name" >&2; exit 2; }

if [[ "$LOCK" == 0 ]] && "${EXCORE[@]}" sources verify --lock "$LOCKFILE" --root "$OUT" >/dev/null 2>&1; then
  echo "$OUT/$NAME is already present and verified"
  exit 0
fi

[[ -n "$URL" ]] || { echo "--url is required to download" >&2; exit 2; }
AUTH=()
[[ -n "${HF_TOKEN:-}" ]] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
echo "downloading $NAME"
curl -L --fail --retry 5 --retry-delay 5 -C - "${AUTH[@]}" -o "$OUT/$NAME.part" "$URL"
mv "$OUT/$NAME.part" "$OUT/$NAME"

if [[ "$LOCK" == 1 ]]; then
  ARGS=(sources pin --lock "$LOCKFILE" --root "$OUT" --file "$NAME")
  [[ -n "$REPO" ]] && ARGS+=(--repo "$REPO")
  [[ -n "$REVISION" ]] && ARGS+=(--revision "$REVISION")
  [[ -n "$LLAMA_COMMIT" ]] && ARGS+=(--llama-cpp-commit "$LLAMA_COMMIT")
  "${EXCORE[@]}" "${ARGS[@]}"
else
  "${EXCORE[@]}" sources verify --lock "$LOCKFILE" --root "$OUT" \
    || { echo "downloaded file does not match configs/sources.lock.json" >&2; exit 1; }
fi
echo "ready: $OUT/$NAME"
