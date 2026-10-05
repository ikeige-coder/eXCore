"""Manifest parser: turns a miner's YAML into a per-unit assignment map.

A manifest is a short, ordered list of rules over a default::

    schema: excore/manifest@1
    track: CPU-35B
    name: mlp-q4-attn-q8
    default: Q5_K
    rules:
      - match: "L*.mlp"
        layers: "8-55"
        format: Q4_K
      - match: "L*.attn.*"
        format: Q8_0

Later rules override earlier ones. The parser is deliberately strict: unknown keys,
duplicate keys, YAML aliases, rules that match nothing, and illegal
unit/format/quantizer combinations are all errors, because a manifest is untrusted input.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml

from . import SCHEMA_MANIFEST, TRACK_DEFAULT
from .model.qwen38_cpu import ModelSpec, Unit, build_inventory
from .precision import (
    DEFAULT_QUANTIZER,
    PrecisionError,
    check_assignment,
    resolve_format,
)

MAX_BYTES = 64 * 1024
MAX_RULES = 512
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
PATTERN_RE = re.compile(r"^[A-Za-z0-9_.*?\[\]-]{1,64}$")
_TOP_KEYS = {"schema", "track", "name", "description", "default", "default_quantizer", "rules"}
_RULE_KEYS = {"match", "layers", "format", "quantizer"}


class ManifestError(ValueError):
    """Raised with every problem found, not just the first."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


# -- strict YAML --------------------------------------------------------------


class _StrictLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.events.AliasEvent):
            event = self.peek_event()
            raise yaml.composer.ComposerError(
                None, None, "YAML aliases are not allowed in manifests", event.start_mark
            )
        return super().compose_node(parent, index)


def _construct_unique_mapping(loader: _StrictLoader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            dup = key in seen
        except TypeError:
            raise yaml.constructor.ConstructorError(
                None, None, "unhashable mapping key", key_node.start_mark
            ) from None
        if dup:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


# -- data model ---------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    match: str
    layers: tuple[int, ...] | None   # None = all layers (and lm_head)
    format: str                      # canonical tensor type
    quantizer: str


@dataclass(frozen=True)
class Assignment:
    format: str
    quantizer: str
    rule: int                        # index of the rule that set it, -1 for the default


@dataclass(frozen=True)
class Manifest:
    schema: str
    track: str
    name: str
    description: str
    default_format: str
    default_quantizer: str
    rules: tuple[Rule, ...]

    def expand(self, spec: ModelSpec) -> dict[str, Assignment]:
        """Resolve rules into one Assignment per unit. Raises ManifestError."""
        return expand(self, spec)

    def candidate_id(self, spec: ModelSpec) -> str:
        """Stable id from what actually runs; the name and rule layout do not matter."""
        return candidate_id(self.track, self.expand(spec))


# -- parsing ------------------------------------------------------------------


def parse_layers(value: Any, num_layers: int) -> tuple[int, ...]:
    """'0-55', '0-3,8,10-12', 7, or [1, 2, 3] -> sorted unique layer indices."""
    if isinstance(value, bool):
        raise ValueError("layers must be an int, a list of ints, or a range string")
    if isinstance(value, int):
        parts: list[Any] = [value]
    elif isinstance(value, list):
        parts = value
    elif isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            raise ValueError("layers is empty")
    else:
        raise ValueError("layers must be an int, a list of ints, or a range string")
    out: set[int] = set()
    for p in parts:
        if isinstance(p, bool):
            raise ValueError("layers must contain integers")
        if isinstance(p, int):
            lo = hi = p
        elif isinstance(p, str):
            m = re.fullmatch(r"(\d+)(?:-(\d+))?", p)
            if not m:
                raise ValueError(f"bad layer range {p!r}")
            lo = int(m.group(1))
            hi = int(m.group(2)) if m.group(2) is not None else lo
        else:
            raise ValueError("layers must contain integers or range strings")
        if lo > hi:
            raise ValueError(f"layer range {lo}-{hi} is reversed")
        if lo < 0 or hi >= num_layers:
            raise ValueError(f"layers {lo}-{hi} outside 0-{num_layers - 1}")
        out.update(range(lo, hi + 1))
    return tuple(sorted(out))


def parse_manifest(text: str, spec: ModelSpec, *, expected_track: str = TRACK_DEFAULT) -> Manifest:
    """Parse and structurally validate manifest text. Raises ManifestError."""
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise ManifestError([f"manifest exceeds {MAX_BYTES} bytes"])
    try:
        raw = yaml.load(text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise ManifestError([f"invalid YAML: {exc}"]) from None
    if not isinstance(raw, dict):
        raise ManifestError(["manifest must be a YAML mapping"])

    problems: list[str] = []
    for k in sorted(set(raw) - _TOP_KEYS, key=str):
        problems.append(f"unknown top-level key {k!r}")

    if raw.get("schema") != SCHEMA_MANIFEST:
        problems.append(f"schema must be {SCHEMA_MANIFEST!r}")
    if raw.get("track") != expected_track:
        problems.append(f"track must be {expected_track!r}")
    name = raw.get("name")
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        problems.append("name must match [a-z0-9][a-z0-9._-]{0,63}")
        name = "invalid"
    description = raw.get("description", "")
    if not isinstance(description, str) or len(description) > 500:
        problems.append("description must be a string of at most 500 characters")
        description = ""

    default_q = raw.get("default_quantizer", DEFAULT_QUANTIZER)
    if not isinstance(default_q, str):
        problems.append("default_quantizer must be a string")
        default_q = DEFAULT_QUANTIZER

    default_fmt = ""
    try:
        if "default" not in raw:
            raise PrecisionError("'default' format is required")
        default_fmt = resolve_format(raw["default"]).name
    except PrecisionError as exc:
        problems.append(f"default: {exc}")

    rules: list[Rule] = []
    raw_rules = raw.get("rules", [])
    if not isinstance(raw_rules, list):
        problems.append("rules must be a list")
        raw_rules = []
    if len(raw_rules) > MAX_RULES:
        problems.append(f"too many rules ({len(raw_rules)} > {MAX_RULES})")
        raw_rules = raw_rules[:MAX_RULES]
    for i, r in enumerate(raw_rules):
        if not isinstance(r, dict):
            problems.append(f"rule {i}: must be a mapping")
            continue
        for k in sorted(set(r) - _RULE_KEYS, key=str):
            problems.append(f"rule {i}: unknown key {k!r}")
        match = r.get("match")
        if not isinstance(match, str) or not PATTERN_RE.fullmatch(match):
            problems.append(f"rule {i}: match must be a glob over unit names (e.g. 'L*.mlp')")
            continue
        layers = None
        if "layers" in r:
            try:
                layers = parse_layers(r["layers"], spec.num_layers)
            except ValueError as exc:
                problems.append(f"rule {i}: {exc}")
                continue
        try:
            if "format" not in r:
                raise PrecisionError("'format' is required")
            fmt = resolve_format(r["format"]).name
        except PrecisionError as exc:
            problems.append(f"rule {i}: {exc}")
            continue
        q = r.get("quantizer", default_q)
        if not isinstance(q, str):
            problems.append(f"rule {i}: quantizer must be a string")
            continue
        rules.append(Rule(match, layers, fmt, q))

    if problems:
        raise ManifestError(problems)
    return Manifest(
        schema=SCHEMA_MANIFEST,
        track=expected_track,
        name=name,
        description=description,
        default_format=default_fmt,
        default_quantizer=default_q,
        rules=tuple(rules),
    )


def load_manifest(path: str | Path, spec: ModelSpec, *, expected_track: str = TRACK_DEFAULT) -> Manifest:
    p = Path(path)
    try:
        if p.stat().st_size > MAX_BYTES:
            raise ManifestError([f"{p.name}: manifest exceeds {MAX_BYTES} bytes"])
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError([f"cannot read manifest {str(path)!r}: {exc.strerror or exc}"]) from None
    except UnicodeDecodeError:
        raise ManifestError([f"{p.name}: manifest is not valid UTF-8 text"]) from None
    return parse_manifest(text, spec, expected_track=expected_track)


# -- expansion ----------------------------------------------------------------


def _rule_targets(rule: Rule, units: tuple[Unit, ...]) -> list[Unit]:
    layers = set(rule.layers) if rule.layers is not None else None
    out = []
    for u in units:
        if not fnmatchcase(u.name, rule.match):
            continue
        if layers is not None and u.layer not in layers:
            continue
        out.append(u)
    return out


def expand(manifest: Manifest, spec: ModelSpec) -> dict[str, Assignment]:
    units = build_inventory(spec)
    by_name = {u.name: u for u in units}
    problems: list[str] = []

    default_fmt = resolve_format(manifest.default_format)
    result: dict[str, Assignment] = {
        u.name: Assignment(default_fmt.name, manifest.default_quantizer, -1) for u in units
    }

    for i, rule in enumerate(manifest.rules):
        targets = _rule_targets(rule, units)
        if not targets:
            where = f" layers {rule.layers[0]}..{rule.layers[-1]}" if rule.layers else ""
            problems.append(f"rule {i}: match {rule.match!r}{where} matches no units")
            continue
        fmt = resolve_format(rule.format)
        for u in targets:
            msg = check_assignment(u, fmt, rule.quantizer)
            if msg:
                problems.append(f"rule {i}: {msg}")
            else:
                result[u.name] = Assignment(fmt.name, rule.quantizer, i)

    # The default only has to be legal for units that still use it after every rule.
    default_problems: set[str] = set()
    for name, a in result.items():
        if a.rule == -1:
            u = by_name[name]
            msg = check_assignment(u, default_fmt, manifest.default_quantizer)
            if msg:
                default_problems.add(f"default: {msg.split(': ', 1)[1]} (first seen on {u.kind} units)")
    problems = sorted(default_problems) + problems

    if problems:
        raise ManifestError(problems)
    return result


def candidate_id(track: str, assignments: dict[str, Assignment]) -> str:
    """16 hex chars of sha256 over the canonical (unit -> format@quantizer) map."""
    canon = {n: f"{a.format}@{a.quantizer}" for n, a in sorted(assignments.items())}
    blob = json.dumps({"track": track, "units": canon}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
