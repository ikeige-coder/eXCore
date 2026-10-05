"""CPU model config: the searchable layers/units of a hybrid GDN/attention track.

A *unit* is the smallest piece whose storage format a miner may choose. The default
eXCore-v1 track (35B, 84 layers) has 358 units:

    63 Gated-DeltaNet layers x 3 units (qkv, z, out)  = 189
    21 full-attention layers x 4 units (q, k, v, o)   =  84
    84 MLP blocks x 1 unit (gate+up+down together)    =  84
    lm_head                                           =   1

(The archived base-engine 27B track has 273: 144 + 64 + 64 + 1.)

Everything else in the source file (embeddings, norms, conv/state parameters, ...) is
frozen: it is whatever tensor is not owned by a unit, and it must stay byte-identical
to the hash-locked source (see ``owned_tensor_names``).

All dimensions come from ``configs/hpc_cpu.yaml``; nothing in this module is
hard-wired to one model, so a new track only needs a new config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Unit kinds
GDN = "gdn"
ATTN = "attn"
MLP = "mlp"
LM_HEAD = "lm_head"

GDN_ROLES = ("qkv", "z", "out")
ATTN_ROLES = ("q", "k", "v", "o")

# GGUF tensor names (llama.cpp convention) for each unit role.
_GDN_TENSORS = {"qkv": "attn_qkv", "z": "attn_gate", "out": "ssm_out"}
_ATTN_TENSORS = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output"}
_MLP_TENSORS = ("ffn_gate", "ffn_up", "ffn_down")

@dataclass(frozen=True)
class TensorSpec:
    """One weight matrix. ``in_features`` is the axis the block formats pack along."""

    gguf_name: str
    out_features: int
    in_features: int

    @property
    def params(self) -> int:
        return self.out_features * self.in_features


@dataclass(frozen=True)
class Unit:
    name: str                      # "L7.attn.q", "L12.mlp", "lm_head"
    layer: int | None              # None for lm_head
    kind: str                      # gdn | attn | mlp | lm_head
    role: str | None               # qkv / z / out / q / k / v / o / None
    tensors: tuple[TensorSpec, ...]

    @property
    def params(self) -> int:
        return sum(t.params for t in self.tensors)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_layers: int
    attention_every: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    gdn_key_heads: int
    gdn_key_head_dim: int
    gdn_value_heads: int
    gdn_value_head_dim: int

    @classmethod
    def from_mapping(cls, m: Mapping) -> "ModelSpec":
        try:
            attn, gdn = m["attn"], m["gdn"]
            spec = cls(
                name=str(m["name"]),
                num_layers=int(m["num_layers"]),
                attention_every=int(m["attention_every"]),
                hidden_size=int(m["hidden_size"]),
                intermediate_size=int(m["intermediate_size"]),
                vocab_size=int(m["vocab_size"]),
                num_heads=int(attn["num_heads"]),
                num_kv_heads=int(attn["num_kv_heads"]),
                head_dim=int(attn["head_dim"]),
                gdn_key_heads=int(gdn["num_key_heads"]),
                gdn_key_head_dim=int(gdn["key_head_dim"]),
                gdn_value_heads=int(gdn["num_value_heads"]),
                gdn_value_head_dim=int(gdn["value_head_dim"]),
            )
        except KeyError as exc:
            raise ValueError(f"model config is missing key {exc}") from None
        spec.validate()
        return spec

    def validate(self) -> None:
        for field in (
            "num_layers", "attention_every", "hidden_size", "intermediate_size",
            "vocab_size", "num_heads", "num_kv_heads", "head_dim",
            "gdn_key_heads", "gdn_key_head_dim", "gdn_value_heads", "gdn_value_head_dim",
        ):
            if getattr(self, field) <= 0:
                raise ValueError(f"model config: {field} must be positive")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("model config: num_heads must be a multiple of num_kv_heads")

    # -- layer layout -------------------------------------------------------

    def layer_kind(self, layer: int) -> str:
        """GDN or ATTN for a layer index."""
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"layer {layer} out of range 0..{self.num_layers - 1}")
        return ATTN if (layer + 1) % self.attention_every == 0 else GDN

    @property
    def attention_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_layers) if self.layer_kind(i) == ATTN)

    @property
    def gdn_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_layers) if self.layer_kind(i) == GDN)

    # -- tensor shapes ------------------------------------------------------

    def _gdn_shapes(self) -> dict[str, tuple[int, int]]:
        qk = self.gdn_key_heads * self.gdn_key_head_dim
        v = self.gdn_value_heads * self.gdn_value_head_dim
        return {
            "qkv": (2 * qk + v, self.hidden_size),
            "z": (v, self.hidden_size),
            "out": (self.hidden_size, v),
        }

    def _attn_shapes(self) -> dict[str, tuple[int, int]]:
        q = self.num_heads * self.head_dim
        kv = self.num_kv_heads * self.head_dim
        return {
            "q": (q, self.hidden_size),
            "k": (kv, self.hidden_size),
            "v": (kv, self.hidden_size),
            "o": (self.hidden_size, q),
        }


def build_inventory(spec: ModelSpec) -> tuple[Unit, ...]:
    """Enumerate every searchable unit, in a stable order."""
    units: list[Unit] = []
    gdn_shapes, attn_shapes = spec._gdn_shapes(), spec._attn_shapes()
    for layer in range(spec.num_layers):
        if spec.layer_kind(layer) == GDN:
            for role in GDN_ROLES:
                out_f, in_f = gdn_shapes[role]
                t = TensorSpec(f"blk.{layer}.{_GDN_TENSORS[role]}.weight", out_f, in_f)
                units.append(Unit(f"L{layer}.gdn.{role}", layer, GDN, role, (t,)))
        else:
            for role in ATTN_ROLES:
                out_f, in_f = attn_shapes[role]
                t = TensorSpec(f"blk.{layer}.{_ATTN_TENSORS[role]}.weight", out_f, in_f)
                units.append(Unit(f"L{layer}.attn.{role}", layer, ATTN, role, (t,)))
        h, i = spec.hidden_size, spec.intermediate_size
        mlp = (
            TensorSpec(f"blk.{layer}.ffn_gate.weight", i, h),
            TensorSpec(f"blk.{layer}.ffn_up.weight", i, h),
            TensorSpec(f"blk.{layer}.ffn_down.weight", h, i),
        )
        units.append(Unit(f"L{layer}.mlp", layer, MLP, None, mlp))
    head = TensorSpec("output.weight", spec.vocab_size, spec.hidden_size)
    units.append(Unit("lm_head", None, LM_HEAD, None, (head,)))
    return tuple(units)


def owned_tensor_names(units: tuple[Unit, ...]) -> frozenset[str]:
    """GGUF tensor names that belong to a searchable unit. Every other tensor is frozen."""
    return frozenset(t.gguf_name for u in units for t in u.tensors)
