"""Accuracy drift: Reference-Partition KL-divergence (RP-KL) on CPU.

For every scored position the reference model's next-token distribution ``p`` is
partitioned into its top-k tokens plus one *tail* bucket (everything else). A candidate
is scored by the KL divergence between the two distributions coarsened to that same
partition::

    RP-KL = sum_i p_i log(p_i / q_i)  +  p_tail log(p_tail / q_tail)

where ``q`` is the candidate's distribution over the *reference's* top-k tokens.
Because it is a KL divergence of a coarsening, RP-KL is never larger than the full KL,
needs only k numbers per position to store, and is compared position by position.

Reference distributions are computed once from the unquantized model and stored with
the exact token windows, so candidates are scored on identical tokens without
re-tokenizing. All model access goes through the small ``LogitsBackend`` protocol.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

REF_SCHEMA = "excore/reference@1"
CORPUS_SCHEMA = "excore/corpus@1"
LOCK_NAME = "corpus.lock.json"
EPS = 1e-12
_ROW_CHUNK = 16


class AccuracyError(RuntimeError):
    pass


class LogitsBackend(Protocol):
    """The llama.cpp hook. ``excore.eval.backend.LlamaCppBackend`` is the real one."""

    vocab_size: int

    def tokenize(self, text: str) -> list[int]: ...

    def bos_token(self) -> int: ...

    def logits(self, window: np.ndarray, start: int) -> np.ndarray:
        """float32 (len(window) - start, vocab); row i = logits after ``window[: start + i + 1]``."""
        ...


# -- corpus ---------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def corpus_files(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("*.txt"))


def lock_corpus(root: str | Path) -> dict:
    """Write ``corpus.lock.json`` for the current contents of ``root``."""
    root = Path(root)
    files = {p.name: _sha256_file(p) for p in corpus_files(root)}
    lock = {"schema": CORPUS_SCHEMA, "files": files}
    (root / LOCK_NAME).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def load_corpus(root: str | Path, *, require_lock: bool = True) -> tuple[list[tuple[str, str]], str]:
    """Return ([(name, text)], digest). With a lock, any missing/extra/changed file is an error."""
    root = Path(root)
    files = corpus_files(root)
    hashes = {p.name: _sha256_file(p) for p in files}
    lock_path = root / LOCK_NAME
    if require_lock:
        if not lock_path.exists():
            raise AccuracyError(f"{root}: {LOCK_NAME} is missing")
        locked = json.loads(lock_path.read_text(encoding="utf-8")).get("files", {})
        if not locked:
            raise AccuracyError(f"{root}: corpus is not populated (see {root}/README.md)")
        missing, extra = sorted(set(locked) - set(hashes)), sorted(set(hashes) - set(locked))
        changed = sorted(n for n in set(locked) & set(hashes) if locked[n] != hashes[n])
        if missing or extra or changed:
            raise AccuracyError(
                f"{root}: corpus does not match its lock (missing={missing}, extra={extra}, changed={changed})"
            )
    if not files:
        raise AccuracyError(f"{root}: no .txt files")
    digest = hashlib.sha256("\n".join(f"{n}:{h}" for n, h in sorted(hashes.items())).encode()).hexdigest()
    texts = [(p.name, p.read_text(encoding="utf-8")) for p in files]
    return texts, digest


def make_windows(backend: LogitsBackend, texts: Sequence[tuple[str, str]], ctx: int, max_tokens: int) -> np.ndarray:
    """Tokenize the corpus and cut it into BOS-prefixed windows of exactly ``ctx`` tokens."""
    stream: list[int] = []
    for _, text in texts:
        stream.extend(backend.tokenize(text))
    stream = stream[:max_tokens]
    step = ctx - 1
    n = len(stream) // step
    if n == 0:
        raise AccuracyError(f"corpus has {len(stream)} tokens, fewer than one window of {ctx}")
    windows = np.empty((n, ctx), dtype=np.int32)
    windows[:, 0] = backend.bos_token()
    windows[:, 1:] = np.asarray(stream[: n * step], dtype=np.int32).reshape(n, step)
    return windows


# -- numerics -------------------------------------------------------------------


def logsumexp_rows(logits: np.ndarray) -> np.ndarray:
    out = np.empty(logits.shape[0], dtype=np.float64)
    for i in range(0, logits.shape[0], _ROW_CHUNK):
        x = logits[i : i + _ROW_CHUNK].astype(np.float64)
        m = x.max(axis=1, keepdims=True)
        out[i : i + _ROW_CHUNK] = m[:, 0] + np.log(np.exp(x - m).sum(axis=1))
    return out


def top_k_rows(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic top-k per row: (ids int32, log-probs float32), best first, ties by token id."""
    n, v = logits.shape
    k = min(k, v)
    lse = logsumexp_rows(logits)
    ids = np.empty((n, k), dtype=np.int32)
    logp = np.empty((n, k), dtype=np.float32)
    for i in range(n):
        x = logits[i]
        thr = np.partition(x, v - k)[v - k]
        above = np.flatnonzero(x > thr)
        ties = np.flatnonzero(x == thr)[: k - len(above)]
        sel = np.concatenate([above, ties])
        order = np.lexsort((sel, -x[sel].astype(np.float64)))
        sel = sel[order]
        ids[i] = sel
        logp[i] = (x[sel].astype(np.float64) - lse[i]).astype(np.float32)
    return ids, logp


def rp_kl_rows(logits: np.ndarray, ref_ids: np.ndarray, ref_logp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RP-KL per row (float64) and whether the candidate's argmax equals the reference's."""
    n = logits.shape[0]
    kl = np.empty(n, dtype=np.float64)
    top1 = np.empty(n, dtype=bool)
    for i in range(0, n, _ROW_CHUNK):
        sl = slice(i, i + _ROW_CHUNK)
        x = logits[sl]
        lse = logsumexp_rows(x)
        picked = np.take_along_axis(x, ref_ids[sl].astype(np.int64), axis=1).astype(np.float64)
        q = np.maximum(np.exp(picked - lse[:, None]), EPS)
        lp = ref_logp[sl].astype(np.float64)
        p = np.exp(lp)
        p_tail = np.maximum(1.0 - p.sum(axis=1), 0.0)
        q_tail = np.maximum(1.0 - q.sum(axis=1), EPS)
        head = (p * (lp - np.log(q))).sum(axis=1)
        tail = np.where(p_tail > EPS, p_tail * (np.log(np.maximum(p_tail, EPS)) - np.log(q_tail)), 0.0)
        kl[sl] = np.maximum(head + tail, 0.0)
        top1[sl] = x.argmax(axis=1) == ref_ids[sl, 0]
    return kl, top1


# -- reference store --------------------------------------------------------------


@dataclass
class Reference:
    windows: np.ndarray      # (n_windows, ctx) int32
    ids: np.ndarray          # (n_positions, k) int32
    logprobs: np.ndarray     # (n_positions, k) float32
    meta: dict

    @property
    def ctx(self) -> int:
        return int(self.meta["ctx"])

    @property
    def scored_start(self) -> int:
        return int(self.meta["scored_start"])

    @property
    def per_window(self) -> int:
        return self.ctx - self.scored_start

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(self.meta, sort_keys=True).encode())
        for a in (self.windows, self.ids, self.logprobs):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            np.savez(fh, windows=self.windows, ids=self.ids, logprobs=self.logprobs,
                     meta=np.array(json.dumps(self.meta, sort_keys=True)))

    @classmethod
    def load(cls, path: str | Path) -> "Reference":
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                ref = cls(z["windows"], z["ids"], z["logprobs"], meta)
        except (OSError, ValueError, KeyError) as exc:
            raise AccuracyError(f"{path}: unreadable reference file ({exc})") from None
        if meta.get("schema") != REF_SCHEMA:
            raise AccuracyError(f"{path}: not an {REF_SCHEMA} file")
        n_pos = ref.windows.shape[0] * ref.per_window
        if ref.ids.shape != (n_pos, int(meta["k"])) or ref.logprobs.shape != ref.ids.shape:
            raise AccuracyError(f"{path}: reference arrays are inconsistent")
        return ref

    def verify(self, *, corpus_digest: str | None = None, model_id: str | None = None) -> None:
        if corpus_digest is not None and self.meta.get("corpus_digest") != corpus_digest:
            raise AccuracyError("reference was built from a different corpus")
        if model_id is not None and self.meta.get("model_id") != model_id:
            raise AccuracyError("reference was built from a different source model")


Progress = Callable[[int, int], None]


def build_reference(
    backend: LogitsBackend,
    windows: np.ndarray,
    *,
    k: int,
    corpus_digest: str,
    model_id: str,
    progress: Progress | None = None,
) -> Reference:
    n_windows, ctx = windows.shape
    start = ctx // 2
    ids, logp = [], []
    for w in range(n_windows):
        logits = _checked_logits(backend, windows[w], start)
        i, lp = top_k_rows(logits, k)
        ids.append(i)
        logp.append(lp)
        if progress:
            progress(w + 1, n_windows)
    meta = {
        "schema": REF_SCHEMA, "k": int(min(k, backend.vocab_size)), "ctx": int(ctx),
        "scored_start": int(start), "vocab_size": int(backend.vocab_size),
        "corpus_digest": corpus_digest, "model_id": model_id, "n_windows": int(n_windows),
    }
    return Reference(windows.astype(np.int32), np.concatenate(ids), np.concatenate(logp), meta)


def _checked_logits(backend: LogitsBackend, window: np.ndarray, start: int) -> np.ndarray:
    logits = backend.logits(window, start)
    expected = (len(window) - start, backend.vocab_size)
    if logits.shape != expected:
        raise AccuracyError(f"backend returned logits of shape {logits.shape}, expected {expected}")
    if not np.isfinite(logits).all():
        raise AccuracyError("model produced NaN or Inf logits")
    return logits


# -- scoring --------------------------------------------------------------------


@dataclass(frozen=True)
class AccuracyResult:
    rp_kl: float                       # mean over scored positions (lower = closer to the reference)
    stderr: float                      # standard error from per-window means
    top1_agreement: float              # share of positions where argmax matches the reference
    n_positions: int
    per_position: np.ndarray           # float32, for paired comparisons
    window_means: np.ndarray           # float64
    reference_digest: str

    def to_dict(self) -> dict:
        return {"rp_kl": self.rp_kl, "stderr": self.stderr, "top1_agreement": self.top1_agreement,
                "n_positions": self.n_positions, "reference": self.reference_digest}


def score(backend: LogitsBackend, ref: Reference, *, progress: Progress | None = None) -> AccuracyResult:
    if backend.vocab_size != int(ref.meta["vocab_size"]):
        raise AccuracyError("candidate vocabulary size differs from the reference")
    n_windows, per = ref.windows.shape[0], ref.per_window
    kl_all = np.empty(n_windows * per, dtype=np.float64)
    top1_all = np.empty(n_windows * per, dtype=bool)
    for w in range(n_windows):
        sl = slice(w * per, (w + 1) * per)
        logits = _checked_logits(backend, ref.windows[w], ref.scored_start)
        kl_all[sl], top1_all[sl] = rp_kl_rows(logits, ref.ids[sl], ref.logprobs[sl])
        if progress:
            progress(w + 1, n_windows)
    means = kl_all.reshape(n_windows, per).mean(axis=1)
    se = float(means.std(ddof=1) / np.sqrt(n_windows)) if n_windows > 1 else float("nan")
    return AccuracyResult(float(kl_all.mean()), se, float(top1_all.mean()), kl_all.size,
                          kl_all.astype(np.float32), means, ref.digest())


@dataclass(frozen=True)
class PairedComparison:
    delta: float          # base - candidate; positive means the candidate is closer to the reference
    stderr: float         # paired, from per-window differences
    rel_gain: float       # delta / base
    significant: bool     # |delta| > z * stderr


def compare(base: AccuracyResult, cand: AccuracyResult, *, z: float = 2.0) -> PairedComparison:
    """Paired comparison on identical windows. Raises if they were scored against different references."""
    if base.reference_digest != cand.reference_digest:
        raise AccuracyError("results were scored against different references")
    d = base.window_means - cand.window_means
    delta = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    rel = delta / base.rp_kl if base.rp_kl > 0 else 0.0
    return PairedComparison(delta, se, rel, bool(np.isfinite(se) and abs(delta) > z * se))
