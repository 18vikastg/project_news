import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-change-me-in-production")
    DATABASE_PATH = os.environ.get(
        "NEWS_DB_PATH", str(BASE_DIR / "news_analysis.db")
    )

    LSTM_MODEL_PATH = os.environ.get(
        "LSTM_MODEL_PATH", str(BASE_DIR / "kannada_news_lstm_model.h5")
    )
    TOKENIZER_PATH = os.environ.get(
        "TOKENIZER_PATH", str(BASE_DIR / "tokenizer.pkl")
    )
    MODEL_CONFIG_PATH = os.environ.get(
        "MODEL_CONFIG_PATH", str(BASE_DIR / "model_config.pkl")
    )

    NLLB_MODEL_ID = os.environ.get(
        "NLLB_MODEL_ID", "facebook/nllb-200-distilled-600M"
    )
    NLLB_SRC_LANG = os.environ.get("NLLB_SRC_LANG", "kan_Knda")
    NLLB_TGT_LANG = os.environ.get("NLLB_TGT_LANG", "eng_Latn")

    TRANSLATION_TIMEOUT = float(os.environ.get("TRANSLATION_TIMEOUT", "120"))
    TRANSLATION_CACHE_SIZE = int(os.environ.get("TRANSLATION_CACHE_SIZE", "128"))
    MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "8000"))
    CHUNK_MAX_CHARS = int(os.environ.get("CHUNK_MAX_CHARS", "400"))
    WARMUP_TRANSLATOR = os.environ.get("WARMUP_TRANSLATOR", "0") == "1"

    TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "4"))
