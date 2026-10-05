import json

import numpy as np
import pytest

from synthetic import SyntheticBackend, sample_texts

from excore.eval.accuracy import (
    AccuracyError,
    Reference,
    build_reference,
    compare,
    load_corpus,
    lock_corpus,
    make_windows,
    rp_kl_rows,
    score,
    top_k_rows,
)

CTX, K = 64, 32


@pytest.fixture(scope="module")
def ref():
    b = SyntheticBackend()
    windows = make_windows(b, sample_texts(), CTX, 5000)
    return build_reference(b, windows, k=K, corpus_digest="c" * 8, model_id="m")


def full_kl(p_logits, q_logits):
    def sm(x):
        x = x.astype(np.float64)
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    p, q = sm(p_logits), sm(q_logits)
    return (p * (np.log(p) - np.log(q))).sum(-1)


def test_top_k_is_deterministic_with_ties():
    row = np.zeros((1, 10), dtype=np.float32)
    row[0, [2, 5, 7, 9]] = 3.0
    ids, lp = top_k_rows(row, 3)
    assert ids[0].tolist() == [2, 5, 7]
    assert np.allclose(lp[0], lp[0][0])


def test_make_windows(ref):
    assert ref.windows.shape[1] == CTX and (ref.windows[:, 0] == 0).all()
    assert ref.ids.shape == (ref.windows.shape[0] * (CTX - CTX // 2), K)


def test_identical_model_scores_zero(ref):
    r = score(SyntheticBackend(), ref)
    assert r.rp_kl < 1e-9 and r.top1_agreement == 1.0
    assert r.n_positions == ref.ids.shape[0]


def test_rp_kl_grows_with_noise(ref):
    vals = [score(SyntheticBackend(noise=n), ref).rp_kl for n in (0.1, 0.5, 1.5)]
    assert vals[0] < vals[1] < vals[2]
    assert score(SyntheticBackend(noise=1.5), ref).top1_agreement < 1.0


def test_rp_kl_is_bounded_by_full_kl_and_exact_at_full_k():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(12, 50)).astype(np.float32) * 2
    b = a + rng.normal(size=a.shape).astype(np.float32)
    ids, lp = top_k_rows(a, 8)
    rp, _ = rp_kl_rows(b, ids, lp)
    assert (rp <= full_kl(a, b) + 1e-6).all() and (rp > 0).all()
    ids, lp = top_k_rows(a, 50)
    rp_full, _ = rp_kl_rows(b, ids, lp)
    assert np.allclose(rp_full, full_kl(a, b), atol=1e-5)


def test_reference_roundtrip_and_digest(ref, tmp_path):
    p = tmp_path / "ref.npz"
    ref.save(p)
    back = Reference.load(p)
    assert back.digest() == ref.digest() and back.meta == ref.meta
    back.ids[0, 0] += 1
    assert back.digest() != ref.digest()


def test_reference_verify_and_bad_files(ref, tmp_path):
    ref.verify(corpus_digest="c" * 8, model_id="m")
    with pytest.raises(AccuracyError, match="different corpus"):
        ref.verify(corpus_digest="x")
    with pytest.raises(AccuracyError, match="different source model"):
        ref.verify(model_id="other")
    (tmp_path / "junk.npz").write_bytes(b"nope")
    with pytest.raises(AccuracyError, match="unreadable"):
        Reference.load(tmp_path / "junk.npz")


def test_nan_logits_and_vocab_mismatch_rejected(ref):
    class Broken(SyntheticBackend):
        def logits(self, window, start):
            z = super().logits(window, start)
            z[0, 0] = np.nan
            return z

    with pytest.raises(AccuracyError, match="NaN"):
        score(Broken(), ref)
    with pytest.raises(AccuracyError, match="vocabulary"):
        score(SyntheticBackend(vocab=300), ref)


def test_paired_comparison(ref):
    far, near = score(SyntheticBackend(noise=1.0), ref), score(SyntheticBackend(noise=0.2), ref)
    c = compare(far, near)
    assert c.delta > 0 and c.significant and 0 < c.rel_gain < 1
    assert not compare(near, near).significant
    other = build_reference(SyntheticBackend(seed=5), ref.windows, k=K, corpus_digest="c" * 8, model_id="m")
    with pytest.raises(AccuracyError, match="different references"):
        compare(far, score(SyntheticBackend(seed=5), other))


def test_corpus_lock(tmp_path):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "b.txt").write_text("second file", encoding="utf-8")
    with pytest.raises(AccuracyError, match="missing"):
        load_corpus(tmp_path)
    (tmp_path / "corpus.lock.json").write_text(json.dumps({"files": {}}))
    with pytest.raises(AccuracyError, match="not populated"):
        load_corpus(tmp_path)
    lock_corpus(tmp_path)
    texts, digest = load_corpus(tmp_path)
    assert [n for n, _ in texts] == ["a.txt", "b.txt"] and len(digest) == 64
    (tmp_path / "a.txt").write_text("hello w0rld", encoding="utf-8")
    with pytest.raises(AccuracyError, match="changed=\\['a.txt'\\]"):
        load_corpus(tmp_path)
    lock_corpus(tmp_path)
    (tmp_path / "c.txt").write_text("sneaky", encoding="utf-8")
    with pytest.raises(AccuracyError, match="extra=\\['c.txt'\\]"):
        load_corpus(tmp_path)
    assert load_corpus(tmp_path, require_lock=False)[0]
