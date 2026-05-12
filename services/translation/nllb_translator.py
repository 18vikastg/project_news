from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


def _install_torch_thread_env(num_threads: int) -> None:
    try:
        import torch

        torch.set_num_threads(max(1, num_threads))
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, num_threads)))
    except ImportError:
        pass


class NLLBTranslator:
    """
    Lazy-loaded Kannada → English translator using a local Hugging Face seq2seq model.
    Thread-safe singleton usage via module-level factory (see get_nllb_translator).
    """

    def __init__(
        self,
        model_id: str,
        src_lang: str,
        tgt_lang: str,
        timeout_sec: float = 120.0,
        cache_size: int = 128,
        max_input_chars: int = 8000,
        chunk_max_chars: int = 400,
        torch_num_threads: int = 4,
    ):
        self.model_id = model_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.timeout_sec = timeout_sec
        self.cache_size = max(0, cache_size)
        self.max_input_chars = max_input_chars
        self.chunk_max_chars = max(64, chunk_max_chars)
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._device = None
        _install_torch_thread_env(torch_num_threads)
        self._cache: OrderedDict[str, str] = OrderedDict()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.info("Loading NLLB model %s (first use may download weights)...", self.model_id)
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            logger.info("NLLB loaded on device=%s", device)

    def _forced_bos_token_id(self) -> int:
        tok = self._tokenizer
        if hasattr(tok, "lang_code_to_id") and self.tgt_lang in tok.lang_code_to_id:
            return int(tok.lang_code_to_id[self.tgt_lang])
        tid = tok.convert_tokens_to_ids(self.tgt_lang)
        if isinstance(tid, int) and tid >= 0:
            return tid
        raise ValueError(f"Cannot resolve target language token id for {self.tgt_lang!r}")

    def _tokenize_chunk(self, text: str):
        tok = self._tokenizer
        kwargs = {"return_tensors": "pt", "truncation": True, "max_length": 512}
        try:
            return tok(text, src_lang=self.src_lang, **kwargs)
        except TypeError:
            return tok(text, **kwargs)

    def _translate_chunk_impl(self, text: str) -> str:
        import torch

        self._ensure_loaded()
        tok = self._tokenizer
        model = self._model
        inputs = self._tokenize_chunk(text)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        bos = self._forced_bos_token_id()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                forced_bos_token_id=bos,
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
            )
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        return (decoded[0] or "").strip()

    @staticmethod
    def _split_chunks(text: str, max_chars: int) -> list[str]:
        text = text.strip()
        if len(text) <= max_chars:
            return [text] if text else []
        parts: list[str] = []
        boundary = re.compile(r"[\n।\.!?]\s*")
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            segment = text[start:end]
            if end < len(text):
                last_break = None
                for m in boundary.finditer(segment):
                    last_break = m
                if last_break is not None:
                    end = start + last_break.end()
                    segment = text[start:end]
            segment = segment.strip()
            if segment:
                parts.append(segment)
            advance = end - start
            start = end if advance > 0 else start + 1
        return parts

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        if self.cache_size <= 0:
            return None
        val = self._cache.pop(key, None)
        if val is not None:
            self._cache[key] = val
        return val

    def _cache_set(self, key: str, value: str) -> None:
        if self.cache_size <= 0:
            return
        self._cache.pop(key, None)
        self._cache[key] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def translate(self, kannada_text: str) -> str:
        raw = (kannada_text or "").strip()
        if not raw:
            return ""
        if len(raw) > self.max_input_chars:
            raw = raw[: self.max_input_chars]
        ck = self._cache_key(raw)
        hit = self._cache_get(ck)
        if hit is not None:
            return hit
        # One lock for the whole translate: HF fast tokenizers (Rust) are not thread-safe,
        # and NLLB mutates tokenizer state; never interleave chunks or concurrent requests.
        with self._infer_lock:
            hit = self._cache_get(ck)
            if hit is not None:
                return hit
            chunks = self._split_chunks(raw, self.chunk_max_chars)
            if not chunks:
                return ""
            outs: list[str] = []
            for ch in chunks:
                t = ch.strip()
                if not t:
                    continue
                outs.append(self._translate_chunk_impl(t))
            result = " ".join(outs).strip()
            self._cache_set(ck, result)
            return result

    def warmup(self, sample: str = "ಸುದ್ದಿ") -> None:
        try:
            self.translate(sample)
            logger.info("Translator warmup completed")
        except Exception as e:
            logger.warning("Translator warmup failed: %s", e)


_translator_singleton: Optional[NLLBTranslator] = None
_singleton_lock = threading.Lock()


def get_nllb_translator(settings) -> NLLBTranslator:
    global _translator_singleton
    with _singleton_lock:
        if _translator_singleton is None:
            _translator_singleton = NLLBTranslator(
                model_id=settings.NLLB_MODEL_ID,
                src_lang=settings.NLLB_SRC_LANG,
                tgt_lang=settings.NLLB_TGT_LANG,
                timeout_sec=settings.TRANSLATION_TIMEOUT,
                cache_size=settings.TRANSLATION_CACHE_SIZE,
                max_input_chars=settings.MAX_INPUT_CHARS,
                chunk_max_chars=settings.CHUNK_MAX_CHARS,
                torch_num_threads=settings.TORCH_NUM_THREADS,
            )
        return _translator_singleton
