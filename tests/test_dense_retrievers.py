from unittest.mock import patch

import numpy as np
import pytest

from src.retrievers.dense_retriever import FaissDenseRetriever
from src.retrievers.in_memory_dense_retriever import InMemoryDenseRetriever


# --- InMemoryDenseRetriever ---


@pytest.fixture
def in_memory_retriever(fake_embedding_cls):
    with patch("src.retrievers.in_memory_dense_retriever.EmbeddingModel", fake_embedding_cls):
        r = InMemoryDenseRetriever()
    return r


@pytest.fixture
def loaded_in_memory(in_memory_retriever, sample_docs):
    in_memory_retriever.add_documents(sample_docs)
    return in_memory_retriever


def test_in_memory_add_documents_stores_chunks(loaded_in_memory, sample_docs):
    assert len(loaded_in_memory.chunks) == len(sample_docs)


def test_in_memory_add_documents_generates_embeddings(loaded_in_memory, sample_docs):
    assert loaded_in_memory.embeddings is not None
    assert loaded_in_memory.embeddings.shape[0] == len(sample_docs)


def test_in_memory_add_documents_empty_raises(in_memory_retriever):
    with pytest.raises(ValueError, match="No documents provided"):
        in_memory_retriever.add_documents([])


def test_in_memory_search_returns_k_results(loaded_in_memory):
    results = loaded_in_memory.search("action film", k=3)
    assert len(results) == 3


def test_in_memory_search_returns_tuples(loaded_in_memory):
    results = loaded_in_memory.search("romantic comedy", k=2)
    for doc, score in results:
        assert isinstance(doc, dict)
        assert isinstance(score, float)


def test_in_memory_search_scores_in_range(loaded_in_memory):
    results = loaded_in_memory.search("film", k=5)
    for _, score in results:
        assert -1.0 <= score <= 1.0


def test_in_memory_search_sorted_descending(loaded_in_memory):
    results = loaded_in_memory.search("action", k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


# --- FaissDenseRetriever ---


@pytest.fixture
def faiss_retriever(fake_embedding_cls):
    with patch("src.retrievers.dense_retriever.EmbeddingModel", fake_embedding_cls):
        r = FaissDenseRetriever()
    return r


@pytest.fixture
def loaded_faiss(faiss_retriever, sample_docs):
    faiss_retriever.add_documents(sample_docs)
    return faiss_retriever


def test_faiss_add_documents_builds_index(loaded_faiss, sample_docs):
    assert loaded_faiss.index is not None
    assert loaded_faiss.index.ntotal == len(sample_docs)


def test_faiss_add_documents_stores_chunks(loaded_faiss, sample_docs):
    assert len(loaded_faiss.chunks) == len(sample_docs)


def test_faiss_add_documents_empty_raises(faiss_retriever):
    with pytest.raises(ValueError, match="No documents provided"):
        faiss_retriever.add_documents([])


def test_faiss_search_returns_k_results(loaded_faiss):
    results = loaded_faiss.search("action film", k=3)
    assert len(results) == 3


def test_faiss_search_returns_tuples(loaded_faiss):
    results = loaded_faiss.search("space science", k=2)
    for doc, score in results:
        assert isinstance(doc, dict)
        assert isinstance(score, float)


def test_faiss_search_before_add_raises(faiss_retriever):
    with pytest.raises(ValueError, match="No index created"):
        faiss_retriever.search("query")


def test_faiss_save_load_roundtrip(loaded_faiss, sample_docs, tmp_path, fake_embedding_cls):
    loaded_faiss.save(str(tmp_path))

    with patch("src.retrievers.dense_retriever.EmbeddingModel", fake_embedding_cls):
        new_retriever = FaissDenseRetriever()
        new_retriever.load(str(tmp_path))

    assert len(new_retriever.chunks) == len(sample_docs)
    assert new_retriever.index.ntotal == len(sample_docs)

    original = loaded_faiss.search("action", k=3)
    restored = new_retriever.search("action", k=3)
    assert [r[0]["text"] for r in original] == [r[0]["text"] for r in restored]


def test_faiss_repr_after_add(loaded_faiss, sample_docs):
    r = repr(loaded_faiss)
    assert "FaissDenseRetriever" in r
    assert f"docs={len(sample_docs)}" in r
