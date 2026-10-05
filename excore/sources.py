"""Hash-locked sources: the original unquantized model and the llama.cpp checkout.

``configs/sources.lock.json`` pins the source GGUF by SHA-256 and llama.cpp by commit.
The evaluator calls ``require_pinned`` before every run, so results can only come from
the exact weights and tools the lock names.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Sequence

SOURCES_SCHEMA = "excore/sources@1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


class SourcesError(RuntimeError):
    pass


def load_lock(path: str | Path, *, track: str | None = None) -> dict:
    try:
        lock = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourcesError(f"{path}: unreadable lock file ({exc})") from None
    if lock.get("schema") != SOURCES_SCHEMA:
        raise SourcesError(f"{path}: not an {SOURCES_SCHEMA} file")
    if track is not None and lock.get("track") != track:
        raise SourcesError(f"{path}: lock is for track {lock.get('track')!r}, not {track!r}")
    return lock


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unpinned_reasons(lock: dict) -> list[str]:
    out = []
    base = lock.get("sources", {}).get("base", {})
    files = base.get("files") or []
    if not files:
        out.append("base model has no pinned files")
    for f in files:
        if not HEX64.fullmatch(str(f.get("sha256") or "")):
            out.append(f"{f.get('name', '?')}: sha256 is not pinned")
    if not COMMIT.fullmatch(str(lock.get("sources", {}).get("llama_cpp", {}).get("commit") or "")):
        out.append("llama.cpp commit is not pinned")
    return out


def is_pinned(lock: dict) -> bool:
    return not unpinned_reasons(lock)


def verify_sources(lock: dict, root: str | Path, *, cache: str | Path | None = None) -> list[str]:
    """Problems found (empty = verified). ``cache`` skips re-hashing files whose size/mtime are unchanged."""
    problems = unpinned_reasons(lock)
    if problems:
        return problems
    root = Path(root)
    seen: dict = {}
    if cache and Path(cache).exists():
        try:
            seen = json.loads(Path(cache).read_text())
        except (OSError, json.JSONDecodeError):
            seen = {}
    new_seen = {}
    for f in lock["sources"]["base"]["files"]:
        p = root / f["name"]
        if not p.is_file():
            problems.append(f"{f['name']}: file not found in {root}")
            continue
        st = p.stat()
        if f.get("size") is not None and st.st_size != int(f["size"]):
            problems.append(f"{f['name']}: size {st.st_size} != locked {f['size']}")
            continue
        key = f"{st.st_size}:{st.st_mtime_ns}:{f['sha256']}"
        if seen.get(f["name"]) != key:
            if sha256_file(p) != f["sha256"]:
                problems.append(f"{f['name']}: sha256 does not match the lock")
                continue
        new_seen[f["name"]] = key
    if cache and not problems:
        Path(cache).write_text(json.dumps(new_seen))
    return problems


def require_pinned(lock_path: str | Path, root: str | Path, *, track: str | None = None,
                   cache: str | Path | None = None) -> dict:
    lock = load_lock(lock_path, track=track)
    problems = verify_sources(lock, root, cache=cache)
    if problems:
        raise SourcesError("sources are not verified: " + "; ".join(problems))
    return lock


def source_gguf(lock: dict, root: str | Path) -> Path:
    files = lock["sources"]["base"]["files"]
    if not files:
        raise SourcesError("lock has no base files")
    return Path(root) / files[0]["name"]


def pin(lock_path: str | Path, root: str | Path, *, repo: str | None, revision: str | None,
        files: Sequence[str], llama_cpp_commit: str | None = None) -> dict:
    """Hash ``files`` under ``root`` and write them into the lock."""
    lock = load_lock(lock_path)
    root = Path(root)
    entries = []
    for name in files:
        p = root / name
        if not p.is_file():
            raise SourcesError(f"{p} does not exist")
        entries.append({"name": name, "size": p.stat().st_size, "sha256": sha256_file(p)})
    base = lock["sources"]["base"]
    base.update(repo=repo, revision=revision, files=entries)
    if llama_cpp_commit:
        if not COMMIT.fullmatch(llama_cpp_commit):
            raise SourcesError("llama.cpp commit must be a full 40-character hash")
        lock["sources"]["llama_cpp"]["commit"] = llama_cpp_commit
    if is_pinned(lock):
        lock["note"] = "Pinned. Re-pin deliberately: every result is only comparable within one lock."
    tmp = Path(str(lock_path) + ".tmp")
    tmp.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, lock_path)
    return lock
