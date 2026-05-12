from services.translation.nllb_translator import NLLBTranslator


def test_split_chunks_respects_max():
    tr = NLLBTranslator(
        model_id="dummy",
        src_lang="kan_Knda",
        tgt_lang="eng_Latn",
        cache_size=0,
        chunk_max_chars=20,
    )
    text = "a" * 100
    chunks = tr._split_chunks(text, 20)
    assert len(chunks) >= 4
    assert all(len(c) <= 20 for c in chunks)


def test_split_chunks_preserves_short():
    tr = NLLBTranslator("x", "kan_Knda", "eng_Latn", cache_size=0)
    assert tr._split_chunks("hello", 100) == ["hello"]
