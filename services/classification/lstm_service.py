from __future__ import annotations

import logging
import sys
import types
from typing import Any, Tuple

logger = logging.getLogger(__name__)


def register_keras_preprocessing_shim() -> None:
    """Old tokenizer.pkl files reference keras_preprocessing.text.Tokenizer."""
    if "keras_preprocessing.text" in sys.modules:
        return
    import tensorflow as tf

    kp = types.ModuleType("keras_preprocessing")
    kpt = types.ModuleType("keras_preprocessing.text")
    kpt.Tokenizer = tf.keras.preprocessing.text.Tokenizer
    kp.text = kpt
    sys.modules["keras_preprocessing"] = kp
    sys.modules["keras_preprocessing.text"] = kpt


def load_tokenizer_pickle(path: str) -> Any:
    import pickle

    register_keras_preprocessing_shim()
    with open(path, "rb") as f:
        tok = pickle.load(f)
    # Legacy pickles (keras_preprocessing) lack `analyzer` required by Keras 3 Tokenizer.
    if not hasattr(tok, "analyzer"):
        tok.analyzer = None
    return tok


def load_legacy_lstm_h5(model_path: str):
    """
    Load HDF5 models saved with older Keras / TF where layer configs include
    fields that Keras 3 rejects (e.g. SpatialDropout1D + trainable + noise_shape;
    LSTM + time_major).
    """
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, SpatialDropout1D
    from tensorflow.keras.models import load_model

    class SpatialDropout1DCompat(SpatialDropout1D):
        @classmethod
        def from_config(cls, config):
            c = dict(config)
            c.pop("trainable", None)
            c.pop("noise_shape", None)
            kwargs: dict[str, Any] = {"rate": float(c.get("rate", 0.0))}
            if c.get("seed") is not None:
                kwargs["seed"] = c["seed"]
            if c.get("name"):
                kwargs["name"] = c["name"]
            if c.get("dtype"):
                kwargs["dtype"] = c["dtype"]
            return cls(**kwargs)

    class LSTMCompat(LSTM):
        @classmethod
        def from_config(cls, config):
            c = dict(config)
            c.pop("time_major", None)
            return super().from_config(c)

    custom_objects = {
        "SpatialDropout1D": SpatialDropout1DCompat,
        "LSTM": LSTMCompat,
    }
    kw = {"compile": False, "custom_objects": custom_objects}
    try:
        model = load_model(model_path, safe_mode=False, **kw)
    except TypeError:
        model = load_model(model_path, **kw)
    logger.info("Loaded LSTM model from %s", model_path)
    return model


class LSTMNewsClassifier:
    """English-text LSTM fake-news scorer (probability of fake)."""

    def __init__(self, model_path: str, tokenizer_path: str, config_path: str):
        import pickle

        self._model = load_legacy_lstm_h5(model_path)
        self._tokenizer = load_tokenizer_pickle(tokenizer_path)
        with open(config_path, "rb") as f:
            self._config = pickle.load(f)

    @staticmethod
    def preprocess_english(text: str) -> str:
        import re

        text = str(text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = " ".join(text.split())
        return text.strip().lower()

    def predict_fake_probability(self, english_text: str) -> Tuple[float, str]:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        if self._model is None:
            return 0.5, "Model not available"
        cleaned = self.preprocess_english(english_text)
        if len(cleaned) < 5:
            return 0.5, "Text too short for reliable prediction"
        sequence = self._tokenizer.texts_to_sequences([cleaned])
        if not sequence or len(sequence[0]) == 0:
            return 0.5, "Text not recognizable by model"
        padded = pad_sequences(
            sequence,
            maxlen=self._config["max_len"],
            padding="post",
            truncating="post",
        )
        prediction = self._model.predict(padded, verbose=0)[0][0]
        return float(prediction), "LSTM prediction successful"
