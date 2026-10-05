import datetime as dt

import pytest

from synthetic import SyntheticBackend, sample_texts

from excore.eval.accuracy import AccuracyError, build_reference, make_windows
from excore.eval.holdout import epoch_now, gain, holdout_verdict, list_shards, load_holdout, select_shard

KW = dict(min_transfer=0.5, tolerance=0.005)


@pytest.mark.parametrize(
    "pub, hold, passed",
    [
        ((1.0, 0.8), (1.0, 0.85), True),      # gain 0.20 public, 0.15 holdout: carries over
        ((1.0, 0.8), (1.0, 0.98), False),     # overfit: only 0.02 survives
        ((1.0, 1.0), (1.0, 1.003), True),     # no claimed gain, tiny holdout slip is within tolerance
        ((1.0, 1.0), (1.0, 1.02), False),     # no claimed gain, but clearly worse on unseen text
        ((1.0, 1.05), (1.0, 1.04), True),     # worse publicly (-5%), holdout agrees
        ((1.0, 1.05), (1.0, 1.09), False),    # worse publicly, much worse on holdout
    ],
)
def test_verdict_cases(pub, hold, passed):
    v = holdout_verdict(*pub, *hold, **KW)
    assert v.passed is passed, v


def test_gain_handles_zero_incumbent():
    assert gain(0.0, 0.1) == 0.0 and gain(2.0, 1.0) == 0.5


def test_epoch_is_iso_week():
    assert epoch_now(dt.date(2026, 10, 1)) == 202640


@pytest.fixture()
def shards(tmp_path):
    b = SyntheticBackend()
    for n in (0, 1, 2):
        d = tmp_path / f"shard-{n}"
        d.mkdir()
        w = make_windows(b, sample_texts(seed=n), 64, 3000)
        build_reference(b, w, k=8, corpus_digest=f"s{n}", model_id="m").save(d / "reference.npz")
    (tmp_path / "shard-9").mkdir()            # no reference file: ignored
    (tmp_path / "notes").mkdir()
    return tmp_path


def test_shard_rotation(shards):
    assert [p.name for p in list_shards(shards)] == ["shard-0", "shard-1", "shard-2"]
    assert [select_shard(shards, e).name for e in (0, 1, 2, 3, 202640)] == [
        "shard-0", "shard-1", "shard-2", "shard-0", f"shard-{202640 % 3}"]
    ref, name = load_holdout(shards, epoch=1)
    assert name == "shard-1" and ref.meta["corpus_digest"] == "s1"


def test_shard_config_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("EXCORE_HOLDOUT_DIR", raising=False)
    with pytest.raises(AccuracyError, match="not configured"):
        select_shard()
    with pytest.raises(AccuracyError, match="no holdout shards"):
        select_shard(tmp_path)
    monkeypatch.setenv("EXCORE_HOLDOUT_DIR", str(tmp_path))
    with pytest.raises(AccuracyError, match="no holdout shards"):
        select_shard()
