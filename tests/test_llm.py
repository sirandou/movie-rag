from unittest.mock import MagicMock, patch

import pytest

from src.utils.llm import SimpleLLM


@pytest.fixture
def mock_openai():
    """Patch OpenAI so no real API calls are made."""
    mock_client = MagicMock()
    with patch("src.utils.llm.OpenAI", return_value=mock_client):
        llm = SimpleLLM(model="gpt-4o-mini")
    return llm, mock_client


def _setup_response(mock_client, content: str):
    response = MagicMock()
    response.choices[0].message.content = content
    mock_client.chat.completions.create.return_value = response


@pytest.fixture
def retrieved_chunks():
    return [
        (
            {"text": "Inception is a mind-bending thriller.", "metadata": {"movie_title": "Inception"}},
            0.92,
        ),
        (
            {"text": "The Matrix explores simulated reality.", "metadata": {"movie_title": "The Matrix"}},
            0.85,
        ),
    ]


# --- response structure ---


def test_returns_dict(mock_openai, retrieved_chunks):
    llm, client = mock_openai
    _setup_response(client, "Great sci-fi picks.")
    result = llm.generate_llm_answer("Recommend sci-fi", retrieved_chunks)
    assert isinstance(result, dict)


def test_result_has_answer_key(mock_openai, retrieved_chunks):
    llm, client = mock_openai
    _setup_response(client, "Inception and The Matrix.")
    result = llm.generate_llm_answer("Recommend sci-fi", retrieved_chunks)
    assert result["answer"] == "Inception and The Matrix."


def test_result_has_query_key(mock_openai, retrieved_chunks):
    llm, client = mock_openai
    _setup_response(client, "Some answer.")
    result = llm.generate_llm_answer("My question", retrieved_chunks)
    assert result["query"] == "My question"


def test_result_has_sources(mock_openai, retrieved_chunks):
    llm, client = mock_openai
    _setup_response(client, "Answer.")
    result = llm.generate_llm_answer("Q", retrieved_chunks)
    assert "sources" in result
    assert len(result["sources"]) == len(retrieved_chunks)


def test_result_has_num_chunks_used(mock_openai, retrieved_chunks):
    llm, client = mock_openai
    _setup_response(client, "Answer.")
    result = llm.generate_llm_answer("Q", retrieved_chunks)
    assert result["num_chunks_used"] == len(retrieved_chunks)


# --- source structure ---


def test_sources_have_rank(mock_openai, retrieved_chunks):
    llm, client = mock_openai
    _setup_response(client, "Answer.")
    sources = llm.generate_llm_answer("Q", retrieved_chunks)["sources"]
    assert sources[0]["rank"] == 1
    assert sources[1]["rank"] == 2


def test_sources_have_score(mock_openai, retrieved_chunks):
    llm, client = mock_openai
    _setup_response(client, "Answer.")
    sources = llm.generate_llm_answer("Q", retrieved_chunks)["sources"]
    assert sources[0]["score"] == 0.92
    assert sources[1]["score"] == 0.85


def test_sources_have_movie_title(mock_openai, retrieved_chunks):
    llm, client = mock_openai
    _setup_response(client, "Answer.")
    sources = llm.generate_llm_answer("Q", retrieved_chunks)["sources"]
    assert sources[0]["movie"] == "Inception"
    assert sources[1]["movie"] == "The Matrix"


def test_sources_text_truncated_to_150(mock_openai):
    llm, client = mock_openai
    _setup_response(client, "Answer.")
    long_text = "word " * 100
    chunks = [({"text": long_text, "metadata": {"movie_title": "X"}}, 0.5)]
    sources = llm.generate_llm_answer("Q", chunks)["sources"]
    assert sources[0]["text"].endswith("...")
    assert len(sources[0]["text"]) <= 155  # 150 chars + "..."


# --- unknown movie title ---


def test_sources_unknown_movie_when_no_title(mock_openai):
    llm, client = mock_openai
    _setup_response(client, "Answer.")
    chunks = [({"text": "Some text.", "metadata": {}}, 0.5)]
    sources = llm.generate_llm_answer("Q", chunks)["sources"]
    assert sources[0]["movie"] == "Unknown"


# --- empty chunks ---


def test_empty_chunks(mock_openai):
    llm, client = mock_openai
    _setup_response(client, "No info.")
    result = llm.generate_llm_answer("Q", [])
    assert result["num_chunks_used"] == 0
    assert result["sources"] == []


# --- API is called once ---


def test_api_called_once(mock_openai, retrieved_chunks):
    llm, client = mock_openai
    _setup_response(client, "Answer.")
    llm.generate_llm_answer("Q", retrieved_chunks)
    assert client.chat.completions.create.call_count == 1


def test_temperature_passed_to_api(mock_openai, retrieved_chunks):
    llm, client = mock_openai
    _setup_response(client, "Answer.")
    llm.generate_llm_answer("Q", retrieved_chunks, temperature=0.2)
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.2
