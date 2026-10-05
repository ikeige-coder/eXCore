#!/usr/bin/env bash
# Build the pinned llama.cpp CPU tools eXCore needs: llama-quantize and llama-bench.
#
#   scripts/setup_bench.sh --commit <40-hex sha> [--prefix DIR] [--lock] [--with-python] [--jobs N]
#
# The commit must be given explicitly (or already pinned in configs/sources.lock.json):
# results are only comparable when everyone measures with the same llama.cpp build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="$ROOT/configs/sources.lock.json"
CONFIG="$ROOT/configs/hpc_cpu.yaml"
PREFIX="${EXCORE_HOME:-$ROOT/.excore}"
COMMIT=""
WRITE_LOCK=0
WITH_PYTHON=0
JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

usage() { sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit)      COMMIT="${2:?--commit needs a value}"; shift 2 ;;
    --prefix)      PREFIX="${2:?--prefix needs a value}"; shift 2 ;;
    --jobs)        JOBS="${2:?--jobs needs a value}"; shift 2 ;;
    --lock)        WRITE_LOCK=1; shift ;;
    --with-python) WITH_PYTHON=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for tool in git cmake python3; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done
command -v c++ >/dev/null || command -v g++ >/dev/null || command -v clang++ >/dev/null \
  || { echo "missing a C++ compiler (install build-essential or clang)" >&2; exit 1; }

if [[ -z "$COMMIT" ]]; then
  COMMIT="$(python3 - "$LOCK_FILE" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["sources"]["llama_cpp"].get("commit") or "")
PY
)"
fi
if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "no llama.cpp commit pinned: pass --commit <full 40-character sha> (add --lock to record it)" >&2
  exit 2
fi

# CMake flags from the track config; ARM uses the native NEON build, x86 the pinned AVX2 build.
ARCH="$(uname -m)"
if [[ "$ARCH" == "x86_64" || "$ARCH" == "amd64" ]]; then
  FLAGS="$(python3 - "$CONFIG" <<'PY'
import sys, yaml
flags = yaml.safe_load(open(sys.argv[1]))["runtime"]["cmake_flags"]
print(" ".join(f"-D{f}" for f in flags))
PY
)"
else
  FLAGS="-DGGML_NATIVE=ON -DGGML_CUDA=OFF -DGGML_METAL=OFF"
fi
FLAGS="$FLAGS -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release"

SRC="$PREFIX/src/llama.cpp"
BUILD="$PREFIX/build"
mkdir -p "$SRC" "$PREFIX/bin"
if [[ ! -d "$SRC/.git" ]]; then
  git -C "$SRC" init -q
  git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp
fi
git -C "$SRC" fetch -q --depth 1 origin "$COMMIT"
git -C "$SRC" checkout -q --detach FETCH_HEAD
[[ "$(git -C "$SRC" rev-parse HEAD)" == "$COMMIT" ]] || { echo "checked-out commit does not match $COMMIT" >&2; exit 1; }

echo "building llama.cpp $COMMIT for $ARCH with: $FLAGS"
# shellcheck disable=SC2086
cmake -S "$SRC" -B "$BUILD" $FLAGS >/dev/null
cmake --build "$BUILD" --config Release -j "$JOBS" --target llama-quantize llama-bench

for bin in llama-quantize llama-bench; do
  found="$(find "$BUILD" -type f -name "$bin" -perm -u+x | head -n1)"
  [[ -n "$found" ]] || { echo "build finished but $bin was not produced" >&2; exit 1; }
  install -m 0755 "$found" "$PREFIX/bin/$bin"
done
echo "$COMMIT" > "$PREFIX/llama.cpp.commit"

if [[ "$WRITE_LOCK" == 1 ]]; then
  python3 - "$LOCK_FILE" "$COMMIT" <<'PY'
import json, sys
path, commit = sys.argv[1], sys.argv[2]
lock = json.load(open(path))
lock["sources"]["llama_cpp"]["commit"] = commit
open(path, "w").write(json.dumps(lock, indent=2) + "\n")
print(f"recorded llama.cpp {commit} in {path}")
PY
fi

if [[ "$WITH_PYTHON" == 1 ]]; then
  echo "installing llama-cpp-python from source (CPU only)"
  CMAKE_ARGS="$FLAGS" python3 -m pip install --no-binary llama-cpp-python --force-reinstall llama-cpp-python
  python3 -c "import llama_cpp; print('llama-cpp-python', llama_cpp.__version__)"
  echo "NOTE: the bindings bundle their own llama.cpp. Accuracy numbers come from that bundled version," >&2
  echo "      while quantization and benchmarking use $COMMIT. Keep both on the same release." >&2
fi

cat <<MSG

done. Add these to the validator's environment:
  export EXCORE_LLAMA_QUANTIZE="$PREFIX/bin/llama-quantize"
  export EXCORE_LLAMA_BENCH="$PREFIX/bin/llama-bench"
MSG
