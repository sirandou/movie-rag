from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.chunk import MovieReviewChunker, chunk


@pytest.fixture
def chunker():
    """MovieReviewChunker with SentenceTransformer mocked out."""
    with patch("src.data.chunk.SentenceTransformer") as mock_st:
        mock_model = MagicMock()

        def fake_encode(texts):
            return np.random.rand(len(texts), 4).astype("float32")

        mock_model.encode.side_effect = fake_encode
        mock_st.return_value = mock_model
        yield MovieReviewChunker()


# --- chunk_fixed_size_tokens ---


def test_fixed_chunk_returns_list(chunker):
    result = chunker.chunk_fixed_size_tokens("Hello world. " * 50)
    assert isinstance(result, list)
    assert len(result) > 0


def test_fixed_chunk_all_items_are_strings(chunker):
    result = chunker.chunk_fixed_size_tokens("Hello world. " * 50)
    assert all(isinstance(c, str) for c in result)


def test_fixed_chunk_short_text_is_single_chunk(chunker):
    result = chunker.chunk_fixed_size_tokens("Short text.", chunk_size=200)
    assert len(result) == 1


def test_fixed_chunk_respects_chunk_size(chunker):
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    text = "word " * 500
    result = chunker.chunk_fixed_size_tokens(text, chunk_size=100, overlap=0)
    for c in result:
        assert len(enc.encode(c)) <= 100


# --- chunk_by_sentences ---


def test_sentence_chunk_returns_list(chunker):
    result = chunker.chunk_by_sentences("This is sentence one. This is sentence two.")
    assert isinstance(result, list)
    assert len(result) > 0


def test_sentence_chunk_review_text(chunker):
    text = (
        "Movie title: Test\nReview: Great film.\nReview: Loved it.\nReview: Must watch."
    )
    result = chunker.chunk_by_sentences(text, sentences_per_chunk=2)
    assert len(result) >= 1
    assert all(isinstance(c, str) for c in result)


def test_sentence_chunk_groups_sentences(chunker):
    sentences = [f"Sentence {i}." for i in range(10)]
    text = " ".join(sentences)
    result = chunker.chunk_by_sentences(text, sentences_per_chunk=5)
    assert len(result) == 2


# --- chunk_by_semantic_similarity ---


def test_semantic_chunk_returns_list(chunker):
    text = "The movie was great. The acting was superb. The plot was engaging."
    result = chunker.chunk_by_semantic_similarity(text)
    assert isinstance(result, list)
    assert len(result) >= 1


def test_semantic_chunk_all_strings(chunker):
    text = "First sentence here. Second sentence here. Third one too."
    result = chunker.chunk_by_semantic_similarity(text)
    assert all(isinstance(c, str) for c in result)


def test_semantic_chunk_empty_like_text_returns_original(chunker):
    text = "ok"
    result = chunker.chunk_by_semantic_similarity(text)
    assert len(result) >= 1


# --- chunk() factory ---


def test_chunk_factory_fixed_strategy(chunker):
    docs = [
        {
            "page_content": "Some content about a film. " * 30,
            "metadata": {"movie_title": "Test Movie"},
        }
    ]
    with patch("src.data.chunk.SentenceTransformer"):
        with patch("src.data.chunk.MovieReviewChunker", return_value=chunker):
            result = chunk("fixed", docs, {"chunk_size": 50, "overlap": 10})

    assert isinstance(result, list)
    assert len(result) > 0


def test_chunk_factory_sentence_strategy(chunker):
    docs = [
        {
            "page_content": "Sentence one. Sentence two. Sentence three.",
            "metadata": {"movie_title": "Test Movie"},
        }
    ]
    with patch("src.data.chunk.SentenceTransformer"):
        with patch("src.data.chunk.MovieReviewChunker", return_value=chunker):
            result = chunk("sentence", docs)

    assert isinstance(result, list)
    assert len(result) > 0


def test_chunk_factory_unknown_strategy_raises(chunker):
    docs = [{"page_content": "text", "metadata": {"movie_title": "Test"}}]
    with patch("src.data.chunk.SentenceTransformer"):
        with patch("src.data.chunk.MovieReviewChunker", return_value=chunker):
            with pytest.raises(ValueError, match="Unknown chunking strategy"):
                chunk("unknown", docs)


def test_chunk_adds_movie_title_prefix(chunker):
    docs = [
        {
            "page_content": "Content without title.",
            "metadata": {"movie_title": "My Movie"},
        }
    ]
    with patch("src.data.chunk.SentenceTransformer"):
        with patch("src.data.chunk.MovieReviewChunker", return_value=chunker):
            result = chunk("sentence", docs)

    assert all("Movie title: My Movie" in c["text"] for c in result)


def test_chunk_output_has_required_metadata_fields(chunker):
    docs = [
        {
            "page_content": "Content here.",
            "metadata": {"movie_title": "Film", "genres": "Drama"},
        }
    ]
    with patch("src.data.chunk.SentenceTransformer"):
        with patch("src.data.chunk.MovieReviewChunker", return_value=chunker):
            result = chunk("sentence", docs)

    for c in result:
        assert "doc_id" in c["metadata"]
        assert "chunk_id" in c["metadata"]
        assert "chunking_strategy" in c["metadata"]
        assert c["metadata"]["chunking_strategy"] == "sentence"
        assert "genres" in c["metadata"]  # original metadata preserved
