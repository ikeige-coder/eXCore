"""Cheap pre-submission screening: is a manifest legal, and roughly how big is the model it builds?

Miners run this locally (``excore check``) before opening a PR. The size is an estimate,
not a measurement: searchable units are exact (parameters x bits per weight); everything
frozen is approximated by the token embeddings plus a small fixed allowance. The embeddings are
memory-mapped and only the rows in use become resident, so they count toward the file size but
not toward the RAM estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .build import manifest_quantizer_names
from .manifest import Manifest
from .model.qwen38_cpu import ModelSpec, build_inventory
from .precision import resolve_format
from .quantizers import ensure_loaded

GIB = 1024 ** 3
FROZEN_ALLOWANCE = 96 << 20        # norms, state parameters, small tensors (these are resident)
RUNTIME_OVERHEAD_GIB = 1.5         # KV cache + compute buffers, rough


@dataclass(frozen=True)
class ScreenReport:
    candidate_id: str
    units: int
    formats: Mapping[str, int] = field(default_factory=dict)      # format -> number of units
    searchable_gib: float = 0.0
    frozen_gib: float = 0.0
    embedding_gib: float = 0.0
    avg_bits_per_weight: float = 0.0
    ceiling_gib: float = 0.0
    rule_units: tuple[int, ...] = ()      # units each rule still owns once every later rule has been applied
    default_units: int = 0                # units left on the manifest default

    @property
    def unused_rules(self) -> tuple[int, ...]:
        """Indices of rules that end up owning no unit because later rules override all of theirs."""
        return tuple(i for i, n in enumerate(self.rule_units) if n == 0)

    @property
    def file_gib(self) -> float:
        return self.searchable_gib + self.frozen_gib

    @property
    def est_ram_gib(self) -> float:
        return self.file_gib - self.embedding_gib + RUNTIME_OVERHEAD_GIB

    @property
    def within_ceiling(self) -> bool:
        return self.est_ram_gib <= self.ceiling_gib


def screen(manifest: Manifest, spec: ModelSpec, cfg: Mapping) -> ScreenReport:
    """Raises ManifestError (or QuantizerError for an unknown quantizer) if the manifest is illegal."""
    ensure_loaded(manifest_quantizer_names(manifest))
    assignments = manifest.expand(spec)
    units = build_inventory(spec)
    formats: dict[str, int] = {}
    total_bytes = 0.0
    total_params = 0
    rule_units = [0] * len(manifest.rules)
    default_units = 0
    for u in units:
        a = assignments[u.name]
        if a.rule < 0:
            default_units += 1
        else:
            rule_units[a.rule] += 1
        fmt = resolve_format(a.format)
        formats[fmt.name] = formats.get(fmt.name, 0) + 1
        total_bytes += fmt.bytes_for(u.params)
        total_params += u.params
    embedding = spec.vocab_size * spec.hidden_size * 2
    frozen = embedding + FROZEN_ALLOWANCE
    return ScreenReport(
        candidate_id=manifest.candidate_id(spec),
        units=len(units),
        formats=dict(sorted(formats.items())),
        searchable_gib=total_bytes / GIB,
        frozen_gib=frozen / GIB,
        embedding_gib=embedding / GIB,
        avg_bits_per_weight=total_bytes * 8 / total_params,
        ceiling_gib=float(cfg["limits"]["max_peak_rss_gib"]),
        rule_units=tuple(rule_units),
        default_units=default_units,
    )
