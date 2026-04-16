from unittest.mock import patch

import pytest

from src.retrievers.dense_retriever import FaissDenseRetriever
from src.retrievers.factory import create_text_retriever, find_retriever_type
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.sparse_retriever import BM25SparseRetriever


# --- create_text_retriever ---


def test_create_sparse_retriever():
    retriever = create_text_retriever({"type": "sparse"})
    assert isinstance(retriever, BM25SparseRetriever)


def test_create_sparse_retriever_custom_params():
    retriever = create_text_retriever({"type": "sparse", "k1": 2.0, "b": 0.5})
    assert retriever.k1 == 2.0
    assert retriever.b == 0.5


def test_create_dense_retriever(fake_embedding_cls):
    with patch("src.retrievers.dense_retriever.EmbeddingModel", fake_embedding_cls):
        retriever = create_text_retriever({"type": "dense"})
    assert isinstance(retriever, FaissDenseRetriever)


def test_create_hybrid_retriever_default(fake_embedding_cls):
    with (
        patch("src.retrievers.dense_retriever.EmbeddingModel", fake_embedding_cls),
        patch("src.retrievers.in_memory_dense_retriever.EmbeddingModel", fake_embedding_cls),
    ):
        retriever = create_text_retriever({"type": "hybrid"})
    assert isinstance(retriever, HybridRetriever)


def test_create_default_is_hybrid(fake_embedding_cls):
    with (
        patch("src.retrievers.dense_retriever.EmbeddingModel", fake_embedding_cls),
        patch("src.retrievers.in_memory_dense_retriever.EmbeddingModel", fake_embedding_cls),
    ):
        retriever = create_text_retriever()
    assert isinstance(retriever, HybridRetriever)


def test_create_none_cfg_is_hybrid(fake_embedding_cls):
    with (
        patch("src.retrievers.dense_retriever.EmbeddingModel", fake_embedding_cls),
        patch("src.retrievers.in_memory_dense_retriever.EmbeddingModel", fake_embedding_cls),
    ):
        retriever = create_text_retriever(None)
    assert isinstance(retriever, HybridRetriever)


def test_create_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown retriever type"):
        create_text_retriever({"type": "unknown"})


def test_create_hybrid_sparse_only_strategy():
    retriever = create_text_retriever({"type": "hybrid", "hybrid_strategy": "sparse"})
    assert isinstance(retriever, HybridRetriever)
    assert retriever.strategy == "sparse"


def test_create_hybrid_custom_alpha(fake_embedding_cls):
    with (
        patch("src.retrievers.dense_retriever.EmbeddingModel", fake_embedding_cls),
        patch("src.retrievers.in_memory_dense_retriever.EmbeddingModel", fake_embedding_cls),
    ):
        retriever = create_text_retriever({"type": "hybrid", "hybrid_alpha": 0.8})
    assert retriever.hybrid_alpha == 0.8


# --- find_retriever_type ---


def test_find_type_sparse():
    r = BM25SparseRetriever()
    assert find_retriever_type(r) == "sparse"


def test_find_type_dense(fake_embedding_cls):
    with patch("src.retrievers.dense_retriever.EmbeddingModel", fake_embedding_cls):
        r = FaissDenseRetriever()
    assert find_retriever_type(r) == "dense"


def test_find_type_hybrid():
    r = create_text_retriever({"type": "hybrid", "hybrid_strategy": "sparse"})
    assert find_retriever_type(r) == "hybrid"


def test_find_type_unknown_raises():
    class WeirdRetriever:
        pass

    with pytest.raises(ValueError, match="Unknown retriever type"):
        find_retriever_type(WeirdRetriever())
