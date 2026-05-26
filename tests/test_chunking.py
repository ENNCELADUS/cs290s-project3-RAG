from rag_collection.chunking import chunk_text, normalize_text


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("a   b\n\n\nc") == "a b\n\nc"


def test_chunk_text_splits_long_paragraphs() -> None:
    chunks = chunk_text("a" * 2500, max_chars=1000, overlap=100)

    assert len(chunks) == 3
    assert all(len(chunk) <= 1000 for chunk in chunks)
