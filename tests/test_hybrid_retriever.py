from unittest.mock import patch

import pytest

from src.retrievers.hybrid_retriever import HybridRetriever


# --- helpers ---


def _make_hybrid(strategy, fake_embedding_cls=None, dense_backend="faiss", alpha=0.5):
    if fake_embedding_cls and strategy in ("dense", "hybrid"):
        with patch("src.retrievers.dense_retriever.EmbeddingModel", fake_embedding_cls):
            return HybridRetriever(
                strategy=strategy,
                dense_backend=dense_backend,
                hybrid_alpha=alpha,
            )
    return HybridRetriever(strategy=strategy, hybrid_alpha=alpha)


# --- Sparse-only strategy (no embedding mock needed) ---


@pytest.fixture
def sparse_hybrid(sample_docs):
    r = HybridRetriever(strategy="sparse")
    r.add_documents(sample_docs)
    return r


def test_sparse_strategy_has_no_dense_retriever():
    r = HybridRetriever(strategy="sparse")
    assert r.dense_retriever is None
    assert r.sparse_retriever is not None


def test_sparse_strategy_add_documents(sparse_hybrid, sample_docs):
    assert len(sparse_hybrid.chunks) == len(sample_docs)


def test_sparse_strategy_search_returns_k(sparse_hybrid):
    results = sparse_hybrid.search("action film", k=3)
    assert len(results) == 3


def test_sparse_strategy_search_returns_tuples(sparse_hybrid):
    for doc, score in sparse_hybrid.search("horror", k=2):
        assert isinstance(doc, dict)
        assert isinstance(score, float)


# --- Dense-only strategy ---


@pytest.fixture
def dense_hybrid(fake_embedding_cls, sample_docs):
    with patch("src.retrievers.dense_retriever.EmbeddingModel", fake_embedding_cls):
        r = HybridRetriever(strategy="dense", dense_backend="faiss")
        r.add_documents(sample_docs)
    return r


def test_dense_strategy_has_no_sparse_retriever(dense_hybrid):
    assert dense_hybrid.sparse_retriever is None
    assert dense_hybrid.dense_retriever is not None


def test_dense_strategy_search_returns_k(dense_hybrid):
    results = dense_hybrid.search("action film", k=3)
    assert len(results) == 3


# --- Full hybrid strategy ---


@pytest.fixture
def full_hybrid(fake_embedding_cls, sample_docs):
    with patch("src.retrievers.dense_retriever.EmbeddingModel", fake_embedding_cls):
        r = HybridRetriever(strategy="hybrid", dense_backend="faiss")
        r.add_documents(sample_docs)
    return r


def test_hybrid_strategy_has_both_retrievers(full_hybrid):
    assert full_hybrid.dense_retriever is not None
    assert full_hybrid.sparse_retriever is not None


def test_hybrid_strategy_search_returns_k(full_hybrid):
    results = full_hybrid.search("action film", k=3)
    assert len(results) == 3


def test_hybrid_strategy_results_sorted_descending(full_hybrid):
    results = full_hybrid.search("film", k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_strategy_returns_dicts(full_hybrid):
    for doc, score in full_hybrid.search("space", k=3):
        assert "text" in doc
        assert "metadata" in doc


def test_unknown_strategy_raises():
    r = HybridRetriever(strategy="sparse")
    r.strategy = "unknown"
    r.add_documents([{"text": "test", "metadata": {"doc_id": 0, "chunk_id": 0}}])
    with pytest.raises(ValueError, match="Unknown strategy"):
        r.search("test")


# --- switch_strategy ---


def test_switch_strategy_sparse_to_dense(fake_embedding_cls, sample_docs):
    r = HybridRetriever(strategy="sparse")
    r.add_documents(sample_docs)

    with patch("src.retrievers.dense_retriever.EmbeddingModel", fake_embedding_cls):
        r.switch_strategy("dense")

    assert r.strategy == "dense"
    assert r.dense_retriever is not None
    results = r.search("action", k=2)
    assert len(results) == 2


def test_switch_strategy_sparse_to_hybrid(fake_embedding_cls, sample_docs):
    r = HybridRetriever(strategy="sparse")
    r.add_documents(sample_docs)

    with patch("src.retrievers.dense_retriever.EmbeddingModel", fake_embedding_cls):
        r.switch_strategy("hybrid")

    assert r.strategy == "hybrid"
    results = r.search("film", k=3)
    assert len(results) == 3


# --- set_hybrid_weight ---


def test_set_hybrid_weight_updates_alpha():
    r = HybridRetriever(strategy="sparse")
    r.set_hybrid_weight(0.9)
    assert r.hybrid_alpha == 0.9


def test_set_hybrid_weight_zero():
    r = HybridRetriever(strategy="sparse")
    r.set_hybrid_weight(0.0)
    assert r.hybrid_alpha == 0.0


# --- save / load roundtrip (sparse strategy) ---


def test_save_load_sparse_roundtrip(sample_docs, tmp_path):
    r = HybridRetriever(strategy="sparse")
    r.add_documents(sample_docs)
    r.save(str(tmp_path))

    r2 = HybridRetriever(strategy="sparse")
    r2.load(str(tmp_path))

    assert r2.strategy == "sparse"
    original = r.search("horror", k=2)
    loaded = r2.search("horror", k=2)
    assert [res[0]["text"] for res in original] == [res[0]["text"] for res in loaded]


# --- repr ---


def test_repr_contains_strategy():
    r = HybridRetriever(strategy="sparse")
    assert "HybridRetriever" in repr(r)
    assert "sparse" in repr(r)
