"""Verdict mapping from P(fake) — no uncertain→ORIGINAL bias."""

from services.pipeline import verdict_from_lstm_probability


def test_high_prob_fake():
    is_fake, conf, method = verdict_from_lstm_probability(0.92, 0.5)
    assert is_fake is True
    assert abs(conf - 0.92) < 1e-9
    assert method == "lstm_local"


def test_low_prob_real():
    is_fake, conf, method = verdict_from_lstm_probability(0.12, 0.5)
    assert is_fake is False
    assert abs(conf - 0.88) < 1e-9


def test_borderline_below_default_threshold_is_real():
    """P(fake) barely above 0.5 but below thr=0.55 → ORIGINAL, not FAKE."""
    is_fake, conf, _ = verdict_from_lstm_probability(0.501, 0.55)
    assert is_fake is False
    assert abs(conf - 0.499) < 1e-9


def test_strict_threshold_respects_higher_thr():
    is_fake, conf, _ = verdict_from_lstm_probability(0.52, 0.55)
    assert is_fake is False
    assert abs(conf - 0.48) < 1e-9

    is_fake2, conf2, _ = verdict_from_lstm_probability(0.56, 0.55)
    assert is_fake2 is True
    assert abs(conf2 - 0.56) < 1e-9


def test_confidence_capped_at_0_99():
    is_fake, conf, _ = verdict_from_lstm_probability(0.999, 0.5)
    assert is_fake is True
    assert conf == 0.99
