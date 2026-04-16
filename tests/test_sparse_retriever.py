import pytest

from src.retrievers.sparse_retriever import BM25SparseRetriever


@pytest.fixture
def retriever():
    return BM25SparseRetriever()


@pytest.fixture
def loaded_retriever(retriever, sample_docs):
    retriever.add_documents(sample_docs)
    return retriever


# --- _tokenize ---


def test_tokenize_lowercases(retriever):
    tokens = retriever._tokenize("Hello WORLD")
    assert all(t == t.lower() for t in tokens)


def test_tokenize_removes_punctuation(retriever):
    tokens = retriever._tokenize("Hello, world!")
    assert "," not in tokens
    assert "!" not in tokens


def test_tokenize_removes_short_tokens(retriever):
    # Filter: len(t) > 2, so 1-2 char tokens are dropped, 3+ char tokens survive
    tokens = retriever._tokenize("a an the film")
    assert "a" not in tokens
    assert "an" not in tokens
    assert "the" in tokens   # 3 chars — kept
    assert "film" in tokens


def test_tokenize_returns_list(retriever):
    result = retriever._tokenize("Some text here")
    assert isinstance(result, list)


# --- add_documents ---


def test_add_documents_builds_index(retriever, sample_docs):
    retriever.add_documents(sample_docs)
    assert retriever.bm25 is not None
    assert len(retriever.chunks) == len(sample_docs)


def test_add_documents_empty_raises(retriever):
    with pytest.raises(ValueError, match="No documents provided"):
        retriever.add_documents([])


def test_add_documents_tokenizes_corpus(retriever, sample_docs):
    retriever.add_documents(sample_docs)
    assert len(retriever.tokenized_corpus) == len(sample_docs)


# --- search ---


def test_search_returns_k_results(loaded_retriever):
    results = loaded_retriever.search("action film", k=3)
    assert len(results) == 3


def test_search_returns_tuples(loaded_retriever):
    results = loaded_retriever.search("horror", k=2)
    for doc, score in results:
        assert isinstance(doc, dict)
        assert isinstance(score, float)


def test_search_before_add_raises(retriever):
    with pytest.raises(ValueError, match="No index created"):
        retriever.search("query")


def test_search_relevance(loaded_retriever):
    results = loaded_retriever.search("horror suspense", k=5)
    top_doc = results[0][0]
    assert "horror" in top_doc["text"].lower() or "suspense" in top_doc["text"].lower()


def test_search_scores_are_non_negative(loaded_retriever):
    results = loaded_retriever.search("film", k=5)
    for _, score in results:
        assert score >= 0.0


def test_search_k_larger_than_corpus(loaded_retriever, sample_docs):
    results = loaded_retriever.search("movie", k=100)
    assert len(results) == len(sample_docs)


# --- save / load ---


def test_save_load_roundtrip(loaded_retriever, sample_docs, tmp_path):
    loaded_retriever.save(str(tmp_path))

    new_retriever = BM25SparseRetriever()
    new_retriever.load(str(tmp_path))

    assert len(new_retriever.chunks) == len(sample_docs)
    assert new_retriever.k1 == loaded_retriever.k1
    assert new_retriever.b == loaded_retriever.b

    original_results = loaded_retriever.search("action", k=3)
    loaded_results = new_retriever.search("action", k=3)
    assert [r[0]["text"] for r in original_results] == [r[0]["text"] for r in loaded_results]


# --- __repr__ ---


def test_repr_before_add(retriever):
    r = repr(retriever)
    assert "BM25SparseRetriever" in r
    assert "docs=0" in r


def test_repr_after_add(loaded_retriever, sample_docs):
    r = repr(loaded_retriever)
    assert f"docs={len(sample_docs)}" in r
