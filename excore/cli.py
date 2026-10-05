"""Command line: ``excore check | build | audit | sources | plots | reference``.

Miners need only ``check`` (is my manifest legal, and about how big is the model?).
Operators use the rest. Evaluator subprocesses call ``build`` and ``audit`` inside the sandbox.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

from .build import BuildError, build, manifest_quantizer_names
from .config import load_config
from .eval.accuracy import AccuracyError, build_reference, load_corpus, make_windows
from .frontier.core_frontier import FrontierError, Space, Spectrum
from .frontier.plots import write_plots
from .gguf import GGUFError
from .manifest import ManifestError, load_manifest
from .model.qwen38_cpu import ModelSpec
from .quantizers import QuantizerError, ensure_loaded
from .screen import screen
from .sources import SourcesError, load_lock, pin, verify_sources
from .validate import audit

KNOWN_ERRORS = (ManifestError, BuildError, QuantizerError, GGUFError, SourcesError, AccuracyError, FrontierError)


def _setup(args):
    cfg = load_config(args.config)
    return cfg, ModelSpec.from_mapping(cfg["model"])


def cmd_check(args) -> int:
    cfg, spec = _setup(args)
    m = load_manifest(args.manifest, spec, expected_track=cfg["track"])
    r = screen(m, spec, cfg)
    print(f"manifest   {m.name}   candidate {r.candidate_id}")
    print(f"units      {r.units}   formats: " + ", ".join(f"{k} x{v}" for k, v in r.formats.items()))
    print(f"size       ~{r.file_gib:.1f} GiB file, ~{r.est_ram_gib:.1f} GiB RAM (estimate), "
          f"{r.avg_bits_per_weight:.2f} bits/weight on searchable units")
    if m.rules:
        print(f"rules      {len(m.rules)} ({r.default_units} units stay on the default)")
        for i in r.unused_rules:
            rule = m.rules[i]
            layers = f" layers {rule.layers[0]}..{rule.layers[-1]}" if rule.layers else ""
            print(f"warning: rule {i} (match {rule.match!r}{layers} -> {rule.format}) has no effect: "
                  "later rules override every unit it matches", file=sys.stderr)
    if not r.within_ceiling:
        print(f"over the {r.ceiling_gib:.1f} GiB RAM ceiling: this recipe would fail the RAM gate", file=sys.stderr)
        return 3
    print("ok: the manifest is legal")
    return 0


def cmd_build(args) -> int:
    cfg, spec = _setup(args)
    m = load_manifest(args.manifest, spec, expected_track=cfg["track"])
    res = build(m, args.source, args.out, spec=spec, cfg=cfg, llama_quantize=args.llama_quantize)
    print(f"built {res.output} ({res.size_bytes / 1024 ** 3:.2f} GiB) candidate {res.candidate_id} sha256 {res.sha256}")
    return 0


def cmd_audit(args) -> int:
    cfg, spec = _setup(args)
    m = load_manifest(args.manifest, spec, expected_track=cfg["track"])
    ensure_loaded(manifest_quantizer_names(m))
    rep = audit(args.candidate, args.source, m, spec, cfg=cfg, replay=args.replay, sample=args.sample)
    if args.json:
        print(json.dumps({"ok": rep.ok, "findings": [f.__dict__ for f in rep.findings], "stats": rep.stats}))
    else:
        print(rep.summary())
        for f in rep.findings:
            print(f"  {f}")
    return 0 if rep.ok else 1


def cmd_sources(args) -> int:
    cfg = load_config(args.config)
    lock_path = Path(args.lock)
    if args.action == "pin":
        lock = pin(lock_path, args.root, repo=args.repo, revision=args.revision, files=args.file,
                   llama_cpp_commit=args.llama_cpp_commit)
        print(f"pinned {len(lock['sources']['base']['files'])} file(s) in {lock_path}")
        return 0
    lock = load_lock(lock_path, track=cfg["track"])
    problems = verify_sources(lock, args.root)
    for p in problems:
        print(f"  {p}", file=sys.stderr)
    print("sources verified" if not problems else "sources NOT verified")
    return 0 if not problems else 1


def cmd_plots(args) -> int:
    cfg = load_config(args.config)
    sp = Spectrum.load(args.spectrum, Space.from_config(cfg), track=cfg["track"])
    for p in write_plots(sp, args.out_dir):
        print(p)
    return 0


def reference_build(cfg: dict, model: str, corpus: str, out: str, model_id: str, *,
                    backend_factory: Callable | None = None, require_lock: bool = True) -> Path:
    """Build and save reference distributions from the unquantized model on a corpus."""
    acc = cfg["accuracy"]
    if backend_factory is None:
        from .eval.backend import LlamaCppBackend

        backend_factory = lambda: LlamaCppBackend(model, n_ctx=int(acc["ctx"]), n_threads=int(cfg["limits"]["threads"]),
                                                  n_batch=int(cfg["limits"]["batch_size"]))
    texts, digest = load_corpus(corpus, require_lock=require_lock)
    backend = backend_factory()
    try:
        windows = make_windows(backend, texts, int(acc["ctx"]), int(acc["max_tokens"]))
        ref = build_reference(backend, windows, k=int(acc["top_k"]), corpus_digest=digest, model_id=model_id,
                              progress=lambda i, n: print(f"\rwindow {i}/{n}", end="", file=sys.stderr))
    finally:
        close = getattr(backend, "close", None)
        if close:
            close()
    print(file=sys.stderr)
    ref.save(out)
    return Path(out)


def cmd_reference(args) -> int:
    cfg = load_config(args.config)
    path = reference_build(cfg, args.model, args.corpus, args.out, args.model_id, require_lock=not args.no_lock)
    print(f"wrote {path}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="excore", description="eXCore: CPU quantization search engine")
    p.add_argument("--config", default=None, help="track config (default: configs/hpc_cpu.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="validate a manifest and estimate the model size")
    c.add_argument("manifest")
    c.set_defaults(fn=cmd_check)

    b = sub.add_parser("build", help="compile a manifest into a candidate GGUF")
    b.add_argument("manifest")
    b.add_argument("--source", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--llama-quantize", default=None)
    b.set_defaults(fn=cmd_build)

    a = sub.add_parser("audit", help="audit a candidate GGUF against the source and manifest")
    a.add_argument("manifest")
    a.add_argument("--candidate", required=True)
    a.add_argument("--source", required=True)
    a.add_argument("--replay", choices=["regenerable", "sample", "none"], default="regenerable")
    a.add_argument("--sample", type=int, default=8)
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_audit)

    s = sub.add_parser("sources", help="pin or verify the hash-locked sources")
    s.add_argument("action", choices=["pin", "verify"])
    s.add_argument("--lock", default="configs/sources.lock.json")
    s.add_argument("--root", required=True, help="directory holding the model files")
    s.add_argument("--file", action="append", default=[], help="file to pin (repeatable)")
    s.add_argument("--repo")
    s.add_argument("--revision")
    s.add_argument("--llama-cpp-commit")
    s.set_defaults(fn=cmd_sources)

    pl = sub.add_parser("plots", help="render the spectrum charts")
    pl.add_argument("--spectrum", default="results/core_spectrum/spectrum.json")
    pl.add_argument("--out-dir", default="results/core_spectrum/plots")
    pl.set_defaults(fn=cmd_plots)

    r = sub.add_parser("reference", help="build reference distributions from the unquantized model")
    r.add_argument("--model", required=True)
    r.add_argument("--corpus", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model-id", required=True, help="e.g. the base model's sha256")
    r.add_argument("--no-lock", action="store_true", help="for the private holdout, which has no lock file")
    r.set_defaults(fn=cmd_reference)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        return args.fn(args)
    except KNOWN_ERRORS as exc:
        if isinstance(exc, ManifestError):
            print("manifest problems:", file=sys.stderr)
            for prob in exc.problems:
                print(f"  - {prob}", file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
