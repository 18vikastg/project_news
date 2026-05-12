# Kannada fake news detection (local pipeline)

This project runs **fully locally**: Kannada text is translated with **Meta NLLB** (`facebook/nllb-200-distilled-600M` by default) via Hugging Face `transformers`, then scored with a **TensorFlow LSTM** trained on your dataset artifacts.

## Requirements

- Python 3.10+
- Several GB free disk space for the translation model (HF cache)
- **RAM**: roughly **8–16 GB** recommended if both PyTorch (NLLB) and TensorFlow (LSTM) run on the same host CPU; **GPU** reduces translation latency.

### PyTorch

Install a CPU or CUDA build of PyTorch that matches your platform from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). Then:

```bash
cd PROJECT_NEWS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `pip install torch` is slow or fails, install `torch` first from the official index, then `pip install -r requirements.txt` while omitting the `torch` line.

## Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `SECRET_KEY` | dev placeholder | Flask session security |
| `NEWS_DB_PATH` | `<project>/news_analysis.db` | SQLite database file |
| `LSTM_MODEL_PATH` | `kannada_news_lstm_model.h5` | LSTM weights |
| `TOKENIZER_PATH` | `tokenizer.pkl` | Keras tokenizer pickle |
| `MODEL_CONFIG_PATH` | `model_config.pkl` | `max_len` and training metadata |
| `NLLB_MODEL_ID` | `facebook/nllb-200-distilled-600M` | HF model id |
| `NLLB_SRC_LANG` | `kan_Knda` | NLLB source language code |
| `NLLB_TGT_LANG` | `eng_Latn` | NLLB target language code |
| `TRANSLATION_TIMEOUT` | `120` | Seconds per translation chunk |
| `TRANSLATION_CACHE_SIZE` | `128` | LRU cache entries (0 disables) |
| `MAX_INPUT_CHARS` | `8000` | Hard cap on input length |
| `CHUNK_MAX_CHARS` | `400` | Chunk size for long articles |
| `WARMUP_TRANSLATOR` | `0` | Set to `1` to translate a short string on startup |
| `TORCH_NUM_THREADS` | `4` | CPU threads for PyTorch |
| `PORT` | `5000` | Dev server port |
| `FLASK_DEBUG` | `0` | Set to `1` for debug mode |

## Train / inference alignment

The bundled `tokenizer.pkl` was produced with legacy **`keras_preprocessing.text.Tokenizer`**. The app registers a small import shim so it loads under modern TensorFlow.

Training defaults in `training.py` use **Kannada** text (`USE_TRANSLATION = False`). The **web app** translates to **English** and runs the LSTM with English-only preprocessing. For best accuracy when training with English labels, set `USE_TRANSLATION = True` in `training.py` (uses the same local NLLB path; slow on CPU) or pre-translate your CSV offline.

## Local run

```bash
export SECRET_KEY="change-me"
python app.py
```

Open `http://127.0.0.1:5000`. Register a user, log in, paste Kannada news, and run **Analyze**.

Use the **same Python** you installed dependencies into (for example `source .venv/bin/activate` then `python app.py`). If you run `python app.py` with a different environment (e.g. a global Jupyter Python), you may get mismatched TensorFlow/Keras versions and model load failures.

- **Health**: `GET /health` — JSON with `predictor_ready` when the analyzer initialized (LSTM + translator stack).

## Troubleshooting

| Symptom | Cause / fix |
|--------|----------------|
| `Translation service temporarily unavailable` or `/api/translate` **503** | `analyzer` failed to start. Check the **first** error in the server log when the app loads. |
| `RuntimeError: Already borrowed` (tokenizers / NLLB) | Fast tokenizer is not thread-safe. Use the latest `nllb_translator.py` (single inference lock, no `ThreadPoolExecutor` around tokenize). |
| `'Tokenizer' object has no attribute 'analyzer'` | Legacy `tokenizer.pkl` from Keras 2 / keras_preprocessing. Fixed by patching `analyzer=None` after load in `lstm_service.load_tokenizer_pickle`. |
| `SpatialDropout1D.__init__() got an unexpected keyword argument 'trainable'` | Old `.h5` saved with Keras 2–style configs. This repo loads it with a **compat shim** (`load_legacy_lstm_h5`). Upgrade to the latest `lstm_service.py` and restart. |
| `Unrecognized keyword arguments passed to LSTM: {'time_major': False}` | Same: legacy config field removed in Keras 3; handled by **`LSTMCompat`** in `load_legacy_lstm_h5`. |
| CUDA / `cuInit` errors in the log | Normal on machines without a working NVIDIA driver; inference falls back to **CPU**. |
| `predictor_ready: false` on `/health` | Missing `kannada_news_lstm_model.h5` / tokenizer / config paths, or model load still failing — fix paths or see log. |

## CLI (no web UI)

```bash
python final.py --text "ನಿಮ್ಮ ಸುದ್ದಿ ಇಲ್ಲಿ"
```

## Tests

```bash
pytest -q
```

Tests use mocks for heavy models by default.

## Deployment notes

- Run behind **Gunicorn** or **Waitress** with a reverse proxy (TLS).
- Use a strong `SECRET_KEY` and restrict file permissions on `NEWS_DB_PATH`.
- Prefer **one worker process per GPU**; CPU-only hosts may use multiple workers if memory allows (each worker loads its own models).
- Pre-download models in your image or run `huggingface-cli download facebook/nllb-200-distilled-600M` for air-gapped installs; set `HF_HOME` as needed.

## Performance

- Enable `WARMUP_TRANSLATOR=1` in production to pay the first-load cost at boot.
- Increase `TRANSLATION_CACHE_SIZE` for repeated snippets (bounded RAM).
- Consider 8-bit loading or **CTranslate2** conversion later for faster CPU inference.

## Security note

Older versions of this repo embedded cloud API keys in source. **Rotate any keys that were ever committed** and rely on environment variables only.
# project_news
