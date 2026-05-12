from services.analysis.heuristics import (
    infer_category,
    infer_summary,
    infer_translation_quality,
)


def test_infer_translation_quality_good():
    kn = "ಕ್ರಿಕೆಟ್ ಪಂದ್ಯ"
    en = "The cricket match was exciting and competitive."
    assert infer_translation_quality(kn, en) == "good"


def test_infer_translation_quality_poor_empty():
    assert infer_translation_quality("abc", "") == "poor"


def test_infer_category_sports():
    cat, conf = infer_category("India won the cricket world cup final")
    assert cat == "sports"
    assert conf > 0.5


def test_infer_category_other():
    cat, _ = infer_category("random text without keywords xyz")
    assert cat == "other"


def test_infer_summary_truncation():
    long_en = "word " * 200
    s = infer_summary(long_en, max_len=50)
    assert len(s) <= 51
    assert s.endswith("…")
