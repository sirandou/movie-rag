from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain.schema import Document

from src.langchain.chunk import chunk_documents


@pytest.fixture
def lc_docs():
    return [
        Document(
            page_content="Sentence one about movies. Sentence two about films.",
            metadata={"movie_title": "Film A", "genres": "Drama"},
        ),
        Document(
            page_content="Another sentence about cinema. More text here.",
            metadata={"movie_title": "Film B", "genres": "Action"},
        ),
    ]


def _mock_st():
    """Patch SentenceTransformer so MovieReviewChunker.__init__ doesn't download models."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts: np.random.rand(len(texts), 4).astype(
        "float32"
    )
    return mock_model


# --- return type ---


def test_returns_list_of_documents(lc_docs):
    with patch("src.data.chunk.SentenceTransformer", return_value=_mock_st()):
        result = chunk_documents(lc_docs, "sentence")
    assert isinstance(result, list)
    assert all(isinstance(d, Document) for d in result)


def test_output_count_at_least_input_count(lc_docs):
    with patch("src.data.chunk.SentenceTransformer", return_value=_mock_st()):
        result = chunk_documents(lc_docs, "sentence")
    assert len(result) >= len(lc_docs)


# --- content ---


def test_chunks_have_page_content(lc_docs):
    with patch("src.data.chunk.SentenceTransformer", return_value=_mock_st()):
        result = chunk_documents(lc_docs, "sentence")
    assert all(isinstance(d.page_content, str) and d.page_content for d in result)


def test_chunks_have_metadata(lc_docs):
    with patch("src.data.chunk.SentenceTransformer", return_value=_mock_st()):
        result = chunk_documents(lc_docs, "sentence")
    assert all(isinstance(d.metadata, dict) for d in result)


def test_original_metadata_preserved(lc_docs):
    with patch("src.data.chunk.SentenceTransformer", return_value=_mock_st()):
        result = chunk_documents(lc_docs, "sentence")
    genres = {d.metadata.get("genres") for d in result}
    assert "Drama" in genres
    assert "Action" in genres


def test_chunk_metadata_added(lc_docs):
    with patch("src.data.chunk.SentenceTransformer", return_value=_mock_st()):
        result = chunk_documents(lc_docs, "sentence")
    for doc in result:
        assert "doc_id" in doc.metadata
        assert "chunk_id" in doc.metadata
        assert "chunking_strategy" in doc.metadata


# --- strategy ---


def test_fixed_strategy_works(lc_docs):
    with patch("src.data.chunk.SentenceTransformer", return_value=_mock_st()):
        result = chunk_documents(lc_docs, "fixed", {"chunk_size": 50, "overlap": 10})
    assert len(result) > 0


def test_invalid_strategy_raises(lc_docs):
    with patch("src.data.chunk.SentenceTransformer", return_value=_mock_st()):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_documents(lc_docs, "nonexistent")


# --- empty input ---


def test_empty_input_returns_empty():
    with patch("src.data.chunk.SentenceTransformer", return_value=_mock_st()):
        result = chunk_documents([], "sentence")
    assert result == []
