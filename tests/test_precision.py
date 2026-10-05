import pytest

from excore.model.qwen38_cpu import build_inventory
from excore.precision import (
    PrecisionError,
    check_assignment,
    estimate_searchable_bytes,
    register_quantizer,
    resolve_format,
)


def test_aliases_resolve_to_tensor_types():
    assert resolve_format("Q4_K_M").name == "Q4_K"
    assert resolve_format("q5_k_m").name == "Q5_K"
    assert resolve_format("Q8_0").bits_per_weight == 8.5


def test_unknown_format_rejected():
    with pytest.raises(PrecisionError):
        resolve_format("Q9_Z")


def test_lm_head_restricted(spec):
    head = build_inventory(spec)[-1]
    assert check_assignment(head, resolve_format("Q2_K"), "runtime")
    assert check_assignment(head, resolve_format("Q6_K"), "runtime") is None


def test_rtn_cannot_make_k_quants(spec):
    unit = build_inventory(spec)[0]
    assert "cannot produce" in check_assignment(unit, resolve_format("Q4_K"), "rtn")
    assert check_assignment(unit, resolve_format("Q4_0"), "rtn") is None


def test_unknown_quantizer(spec):
    unit = build_inventory(spec)[0]
    assert "unknown quantizer" in check_assignment(unit, resolve_format("Q4_K"), "nope")


def test_register_quantizer_is_idempotent_but_not_overwritable():
    register_quantizer("test-plugin", {"Q4_0"})
    register_quantizer("test-plugin", {"Q4_0"})
    with pytest.raises(PrecisionError):
        register_quantizer("test-plugin", {"Q8_0"})


def test_size_estimate_orders_formats(spec):
    units = build_inventory(spec)
    q4 = estimate_searchable_bytes(units, {u.name: "Q4_K" if u.kind != "lm_head" else "Q6_K" for u in units})
    q8 = estimate_searchable_bytes(units, {u.name: "Q8_0" for u in units})
    assert q4 < q8
    assert 18e9 < q4 < 24e9
