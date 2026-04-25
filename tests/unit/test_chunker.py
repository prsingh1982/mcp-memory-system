from memory_core.ingestion.chunking import TextChunker


def test_text_chunker_splits_long_text_with_overlap():
    chunker = TextChunker(max_chars=80, overlap_chars=10, min_chunk_chars=20)
    text = ("This is a test sentence. " * 20).strip()

    chunks = chunker.chunk(document_id="doc-1", text=text)

    assert len(chunks) > 1
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].sequence_index == 0
    assert chunks[1].sequence_index == 1
    assert chunks[0].char_end > chunks[1].char_start


def test_text_chunker_returns_empty_for_blank_text():
    chunker = TextChunker()

    chunks = chunker.chunk(document_id="doc-1", text="   ")

    assert chunks == []
