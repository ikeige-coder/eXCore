import numpy as np
import pytest

from excore.config import load_config, load_model_spec
from excore.gguf import GGMLType, WriteEntry, f32_to_bf16, pack_metadata, write_gguf
from excore.model.qwen38_cpu import ModelSpec, build_inventory


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def spec():
    return load_model_spec()


TINY = {
    "name": "tiny",
    "num_layers": 4,
    "attention_every": 2,
    "hidden_size": 256,
    "intermediate_size": 512,
    "vocab_size": 256,
    "attn": {"num_heads": 2, "num_kv_heads": 1, "head_dim": 128},
    "gdn": {"num_key_heads": 2, "key_head_dim": 64, "num_value_heads": 2, "value_head_dim": 128},
}


@pytest.fixture(scope="session")
def tiny_spec():
    return ModelSpec.from_mapping(TINY)


@pytest.fixture(scope="session")
def tiny_cfg(cfg):
    c = dict(cfg)
    c["model"] = TINY
    c["limits"] = {**cfg["limits"], "threads": 2, "build_timeout_s": 60}
    return c


def _bf16(rng, dims, scale=0.05):
    n = int(np.prod(dims))
    return f32_to_bf16(rng.normal(0, scale, n).astype(np.float32)).tobytes()


def _f32(rng, n):
    return rng.normal(1.0, 0.1, n).astype("<f4").tobytes()


@pytest.fixture(scope="session")
def tiny_source(tmp_path_factory, tiny_spec):
    """A small BF16 GGUF with unit tensors plus frozen embeddings/norms/state tensors."""
    rng = np.random.default_rng(1234)
    entries = []

    def add(name, dims, gt, data):
        entries.append(WriteEntry(name, tuple(dims), gt, lambda d=data: [d]))

    h = tiny_spec.hidden_size
    add("token_embd.weight", (h, tiny_spec.vocab_size), GGMLType.BF16, _bf16(rng, (h, tiny_spec.vocab_size)))
    owned = {t.gguf_name: t for u in build_inventory(tiny_spec) for t in u.tensors}
    for layer in range(tiny_spec.num_layers):
        add(f"blk.{layer}.attn_norm.weight", (h,), GGMLType.F32, _f32(rng, h))
        if tiny_spec.layer_kind(layer) == "gdn":
            add(f"blk.{layer}.ssm_a", (2,), GGMLType.F32, _f32(rng, 2))
        else:
            add(f"blk.{layer}.attn_q_norm.weight", (128,), GGMLType.F32, _f32(rng, 128))
        for name, t in owned.items():
            if name.startswith(f"blk.{layer}."):
                add(name, (t.in_features, t.out_features), GGMLType.BF16,
                    _bf16(rng, (t.in_features, t.out_features)))
    add("output_norm.weight", (h,), GGMLType.F32, _f32(rng, h))
    t = owned["output.weight"]
    add("output.weight", (t.in_features, t.out_features), GGMLType.BF16, _bf16(rng, (h, tiny_spec.vocab_size)))

    meta, kv = pack_metadata([
        ("general.architecture", "string", "qwen3next"),
        ("general.name", "string", "tiny"),
        ("tokenizer.ggml.tokens", "strings", [f"t{i}" for i in range(50)]),
    ])
    path = tmp_path_factory.mktemp("src") / "tiny-bf16.gguf"
    write_gguf(path, metadata_raw=meta, kv_count=kv, alignment=32, entries=entries)
    return path
