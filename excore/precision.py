"""CPU precision space: the legal GGUF execution formats and who can produce them.

llama.cpp stores each tensor in one *tensor type* (Q4_K, Q5_K, Q6_K, Q8_0, ...).
Names such as ``Q4_K_M`` are *file types*: a recipe that mixes tensor types. Because a
manifest assigns formats per unit, the ``_M``/``_S``/``_L`` spellings are accepted
as aliases for their base tensor type.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model.qwen38_cpu import ATTN, GDN, LM_HEAD, MLP, Unit


class PrecisionError(ValueError):
    pass


@dataclass(frozen=True)
class Format:
    name: str
    bits_per_weight: float   # includes block scales/mins
    block_size: int          # elements per block along in_features
    is_float: bool = False

    def bytes_for(self, params: int) -> float:
        return params * self.bits_per_weight / 8.0


FORMATS: dict[str, Format] = {
    f.name: f
    for f in (
        Format("Q2_K", 2.625, 256),
        Format("Q3_K", 3.4375, 256),
        Format("Q4_0", 4.5, 32),
        Format("Q4_1", 5.0, 32),
        Format("Q4_K", 4.5, 256),
        Format("Q5_0", 5.5, 32),
        Format("Q5_1", 6.0, 32),
        Format("Q5_K", 5.5, 256),
        Format("Q6_K", 6.5625, 256),
        Format("Q8_0", 8.5, 32),
        Format("F16", 16.0, 1, is_float=True),
        Format("BF16", 16.0, 1, is_float=True),
    )
}

# File-type spellings -> the tensor type that executes.
ALIASES: dict[str, str] = {
    "Q3_K_S": "Q3_K", "Q3_K_M": "Q3_K", "Q3_K_L": "Q3_K",
    "Q4_K_S": "Q4_K", "Q4_K_M": "Q4_K",
    "Q5_K_S": "Q5_K", "Q5_K_M": "Q5_K",
}

_ALL = frozenset(FORMATS)
_HEAD = frozenset({"Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16", "BF16"})

# Which formats a miner may assign to each kind of unit.
LEGAL_BY_KIND: dict[str, frozenset[str]] = {
    GDN: _ALL,
    ATTN: _ALL,
    MLP: _ALL,
    LM_HEAD: _HEAD,
}

# Which formats each quantizer can emit. Plugins add themselves via register_quantizer.
QUANTIZER_SUPPORT: dict[str, frozenset[str]] = {
    "runtime": _ALL,                                             # native llama.cpp encoders
    "rtn": frozenset({"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"}),  # round-to-nearest block formats
}
DEFAULT_QUANTIZER = "runtime"


def register_quantizer(name: str, formats: set[str] | frozenset[str]) -> None:
    """Called by quantizer plugins (excore.quantizers) when they load."""
    canon = frozenset(resolve_format(f).name for f in formats)
    if name in QUANTIZER_SUPPORT and QUANTIZER_SUPPORT[name] != canon:
        raise PrecisionError(f"quantizer {name!r} already registered with different formats")
    QUANTIZER_SUPPORT[name] = canon


def resolve_format(name: str) -> Format:
    """Case-insensitive lookup; accepts file-type aliases like Q4_K_M."""
    if not isinstance(name, str):
        raise PrecisionError(f"format must be a string, got {type(name).__name__}")
    key = name.strip().upper()
    key = ALIASES.get(key, key)
    try:
        return FORMATS[key]
    except KeyError:
        raise PrecisionError(
            f"unknown format {name!r}; legal: {', '.join(sorted(FORMATS))}"
        ) from None


def check_assignment(unit: Unit, fmt: Format, quantizer: str) -> str | None:
    """Return a human-readable problem, or None if (unit, format, quantizer) is legal."""
    if fmt.name not in LEGAL_BY_KIND[unit.kind]:
        return f"{unit.name}: {fmt.name} is not a legal format for {unit.kind} units"
    support = QUANTIZER_SUPPORT.get(quantizer)
    if support is None:
        return f"{unit.name}: unknown quantizer {quantizer!r}"
    if fmt.name not in support:
        return f"{unit.name}: quantizer {quantizer!r} cannot produce {fmt.name}"
    for t in unit.tensors:
        if t.in_features % fmt.block_size:
            return (
                f"{unit.name}: {t.gguf_name} row length {t.in_features} is not a "
                f"multiple of the {fmt.name} block size {fmt.block_size}"
            )
    return None


def unit_bytes(unit: Unit, fmt: Format) -> float:
    return fmt.bytes_for(unit.params)


def estimate_searchable_bytes(units, formats: dict[str, str]) -> float:
    """Bytes occupied by the searchable units under ``formats`` (unit name -> format name)."""
    return sum(unit_bytes(u, resolve_format(formats[u.name])) for u in units)
