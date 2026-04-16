from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.utils.embeddings import EmbeddingModel


# =============================================================================
# sentence-transformers provider
# =============================================================================


@pytest.fixture
def st_model():
    """EmbeddingModel backed by a mocked SentenceTransformer."""
    mock_st = MagicMock()
    mock_st.get_sentence_embedding_dimension.return_value = 384
    mock_st.encode.side_effect = lambda texts, **kw: np.ones(
        (len(texts), 384), dtype="float32"
    )
    with patch("src.utils.embeddings.SentenceTransformer", return_value=mock_st):
        model = EmbeddingModel("all-MiniLM-L6-v2", "sentence-transformers")
    return model


def test_st_get_dimension(st_model):
    assert st_model.get_dimension() == 384


def test_st_encode_list_shape(st_model):
    result = st_model.encode(["hello", "world"])
    assert result.shape == (2, 384)


def test_st_encode_single_string_shape(st_model):
    result = st_model.encode("single text")
    assert result.shape == (1, 384)


def test_st_encode_returns_float32(st_model):
    result = st_model.encode(["text"])
    assert result.dtype == np.float32


def test_st_repr(st_model):
    r = repr(st_model)
    assert "sentence-transformers" in r
    assert "384" in r


# =============================================================================
# openai provider
# =============================================================================


@pytest.fixture
def openai_model():
    """EmbeddingModel backed by a mocked OpenAI client."""
    mock_client = MagicMock()

    def fake_create(input, model):
        items = [MagicMock(embedding=[0.1] * 1536) for _ in input]
        response = MagicMock()
        response.data = items
        return response

    mock_client.embeddings.create.side_effect = fake_create

    with patch("src.utils.embeddings.OpenAI", return_value=mock_client):
        model = EmbeddingModel("text-embedding-3-small", "openai")
    return model, mock_client


def test_openai_dimension(openai_model):
    model, _ = openai_model
    assert model.get_dimension() == 1536


def test_openai_encode_shape(openai_model):
    model, _ = openai_model
    result = model.encode(["a", "b", "c"])
    assert result.shape == (3, 1536)


def test_openai_encode_returns_float32(openai_model):
    model, _ = openai_model
    result = model.encode(["text"])
    assert result.dtype == np.float32


def test_openai_encode_single_string(openai_model):
    model, _ = openai_model
    result = model.encode("single")
    assert result.shape == (1, 1536)


def test_openai_large_model_dimension():
    mock_client = MagicMock()

    def fake_create(input, model):
        items = [MagicMock(embedding=[0.0] * 3072) for _ in input]
        response = MagicMock()
        response.data = items
        return response

    mock_client.embeddings.create.side_effect = fake_create
    with patch("src.utils.embeddings.OpenAI", return_value=mock_client):
        model = EmbeddingModel("text-embedding-3-large", "openai")
    assert model.get_dimension() == 3072


def test_openai_unknown_model_raises():
    with patch("src.utils.embeddings.OpenAI"):
        with pytest.raises(ValueError, match="Unknown OpenAI model"):
            EmbeddingModel("gpt-4-turbo", "openai")


# =============================================================================
# invalid provider
# =============================================================================


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        EmbeddingModel("some-model", "cohere")


# =============================================================================
# batching (openai)
# =============================================================================


def test_openai_batching_calls_api_multiple_times():
    """encode() should flush mid-batch when token estimate exceeds max_tokens_per_batch=250_000."""
    call_count = 0

    def fake_create(input, model):
        nonlocal call_count
        call_count += 1
        items = [MagicMock(embedding=[0.0] * 1536) for _ in input]
        response = MagicMock()
        response.data = items
        return response

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = fake_create

    with patch("src.utils.embeddings.OpenAI", return_value=mock_client):
        with patch("src.utils.embeddings.time.sleep"):  # skip sleep
            model = EmbeddingModel("text-embedding-3-small", "openai")
            # 600k chars → ~150k tokens each; combined 300k > 250k limit → forces a flush
            long_texts = ["x" * 600_000, "y" * 600_000]
            result = model.encode(long_texts)

    assert call_count >= 2  # batched into multiple API calls
    assert result.shape == (2, 1536)
