from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple

from config.settings import Settings
from services.analysis.heuristics import (
    infer_category,
    infer_summary,
    infer_translation_quality,
)
from services.classification.lstm_service import LSTMNewsClassifier
from services.translation.nllb_translator import NLLBTranslator, get_nllb_translator

logger = logging.getLogger(__name__)


class KannadaNewsAnalyzer:
    """
    Local pipeline: Kannada → English (NLLB) → LSTM fake probability + heuristic metadata.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        translator: Optional[NLLBTranslator] = None,
        translator_factory: Optional[Callable[[], NLLBTranslator]] = None,
    ):
        self.settings = settings or Settings()
        self._translator = translator
        self._translator_factory = translator_factory or (
            lambda: get_nllb_translator(self.settings)
        )
        self._lstm = LSTMNewsClassifier(
            self.settings.LSTM_MODEL_PATH,
            self.settings.TOKENIZER_PATH,
            self.settings.MODEL_CONFIG_PATH,
        )

    def _translator_instance(self) -> NLLBTranslator:
        if self._translator is not None:
            return self._translator
        return self._translator_factory()

    def translate_kannada_to_english(self, kannada_text: str) -> str:
        return self._translator_instance().translate(kannada_text)

    def warmup(self) -> None:
        if self.settings.WARMUP_TRANSLATOR:
            self._translator_instance().warmup()

    def combined_analysis(self, kannada_text: str) -> Tuple[dict[str, Any], str]:
        english_text = self.translate_kannada_to_english(kannada_text)
        # LSTM must see Kannada — same distribution as training (not English ASCII).
        lstm_prob, lstm_status = self._lstm.predict_fake_probability(kannada_text)
        logger.debug("LSTM status=%s P(fake)=%.4f", lstm_status, lstm_prob)

        thr = float(self.settings.LSTM_FAKE_THRESHOLD)
        margin = float(self.settings.LSTM_UNCERTAIN_MARGIN)
        if abs(lstm_prob - 0.5) <= margin:
            is_fake = False
            confidence = 0.5 + abs(lstm_prob - 0.5)
            method = "lstm_local_uncertain_original"
        else:
            is_fake = lstm_prob > thr
            confidence = lstm_prob if is_fake else 1.0 - lstm_prob
            method = "lstm_local"

        category, cat_conf = infer_category(english_text)
        summary = infer_summary(english_text)
        tq = infer_translation_quality(kannada_text, english_text)

        prediction = {
            "is_fake": bool(is_fake),
            "confidence": float(min(max(confidence, 0.0), 0.99)),
            "category": category,
            "category_confidence": float(cat_conf),
            "summary": summary,
            "analysis_method": method,
            "translation_quality": tq,
        }
        return prediction, english_text
