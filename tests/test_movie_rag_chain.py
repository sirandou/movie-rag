from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document
from src.langchain.chains.movie_rag import MovieRAGChain

from src.langchain.retrieval.retrievers import TextRetrieverWrapper
from src.retrievers.sparse_retriever import BM25SparseRetriever


# =============================================================================
# TextRetrieverWrapper
# =============================================================================


@pytest.fixture
def loaded_bm25(sample_docs):
    r = BM25SparseRetriever()
    r.add_documents(sample_docs)
    return r


@pytest.fixture
def wrapper(loaded_bm25):
    return TextRetrieverWrapper(loaded_bm25, k=3)


@pytest.fixture
def langchain_docs(sample_docs):
    return [
        Document(page_content=d["text"], metadata=d["metadata"]) for d in sample_docs
    ]


def test_wrapper_add_converts_to_custom_format(langchain_docs):
    fresh_bm25 = BM25SparseRetriever()
    w = TextRetrieverWrapper(fresh_bm25, k=3)
    w.add_documents(langchain_docs)

    assert len(fresh_bm25.chunks) == len(langchain_docs)
    for chunk in fresh_bm25.chunks:
        assert "text" in chunk
        assert "metadata" in chunk


def test_wrapper_get_relevant_documents_returns_langchain_docs(wrapper):
    results = wrapper._get_relevant_documents("action film")
    assert all(isinstance(d, Document) for d in results)


def test_wrapper_respects_k(wrapper):
    results = wrapper._get_relevant_documents("horror movie")
    assert len(results) == 3


def test_wrapper_includes_score_in_metadata(wrapper):
    results = wrapper._get_relevant_documents("action")
    for doc in results:
        assert "score" in doc.metadata
        assert isinstance(doc.metadata["score"], float)


def test_wrapper_page_content_matches_source(wrapper, sample_docs):
    results = wrapper._get_relevant_documents("documentary nature wildlife")
    texts = {d["text"] for d in sample_docs}
    for doc in results:
        assert doc.page_content in texts


def test_wrapper_preserves_original_metadata(langchain_docs):
    fresh_bm25 = BM25SparseRetriever()
    w = TextRetrieverWrapper(fresh_bm25, k=3)
    w.add_documents(langchain_docs)
    results = w._get_relevant_documents("film")
    for doc in results:
        assert "movie_title" in doc.metadata


# =============================================================================
# MovieRAGChain
# =============================================================================


@pytest.fixture
def fake_lc_docs(sample_docs):
    return [
        Document(page_content=d["text"], metadata=d["metadata"]) for d in sample_docs
    ]


@pytest.fixture
def base_chain_kwargs():
    return {
        "plots_path": "fake_plots.csv",
        "use_custom_retriever": True,
        "use_custom_chunk": False,  # use RecursiveCharacterTextSplitter — no ML needed
        "llm_model": "gpt-4o-mini",
    }


def _make_chain(kwargs, extra=None):
    """Instantiate MovieRAGChain with ChatOpenAI patched out."""
    with patch("src.langchain.chains.movie_rag.ChatOpenAI"):
        return MovieRAGChain(**(kwargs | (extra or {})))


# --- init ---


def test_init_stores_plots_path(base_chain_kwargs):
    chain = _make_chain(base_chain_kwargs)
    assert chain.plots_path == "fake_plots.csv"


def test_init_stores_k(base_chain_kwargs):
    chain = _make_chain(base_chain_kwargs, {"k": 7})
    assert chain.k == 7


def test_init_stores_hyde_flag(base_chain_kwargs):
    chain = _make_chain(base_chain_kwargs, {"use_hyde": True})
    assert chain.use_hyde is True


def test_init_stores_reranking_flag(base_chain_kwargs):
    chain = _make_chain(base_chain_kwargs, {"use_reranking": True})
    assert chain.use_reranking is True


# --- query before build ---


def test_query_before_build_raises(base_chain_kwargs):
    chain = _make_chain(base_chain_kwargs)
    with pytest.raises(ValueError, match="Pipeline not built"):
        chain.query("Best movie?")


# --- build ---


def test_build_sets_qa_chain(base_chain_kwargs, fake_lc_docs):
    bm25 = BM25SparseRetriever()

    with (
        patch("src.langchain.chains.movie_rag.ChatOpenAI"),
        patch(
            "src.langchain.chains.movie_rag.MovieTextDocumentLoader"
        ) as mock_loader_cls,
        patch("src.langchain.chains.movie_rag.RetrievalQA") as mock_rqa,
    ):
        mock_loader_cls.return_value.load.return_value = fake_lc_docs
        mock_rqa.from_chain_type.return_value = MagicMock()

        chain = MovieRAGChain(**base_chain_kwargs, custom_retriever=bm25)
        chain.build()

    assert chain.qa_chain is not None


def test_build_sets_retriever(base_chain_kwargs, fake_lc_docs):
    bm25 = BM25SparseRetriever()

    with (
        patch("src.langchain.chains.movie_rag.ChatOpenAI"),
        patch(
            "src.langchain.chains.movie_rag.MovieTextDocumentLoader"
        ) as mock_loader_cls,
        patch("src.langchain.chains.movie_rag.RetrievalQA") as mock_rqa,
    ):
        mock_loader_cls.return_value.load.return_value = fake_lc_docs
        mock_rqa.from_chain_type.return_value = MagicMock()

        chain = MovieRAGChain(**base_chain_kwargs, custom_retriever=bm25)
        chain.build()

    assert chain.retriever is not None
    assert isinstance(chain.retriever, TextRetrieverWrapper)


# --- query ---


def _built_chain(base_chain_kwargs, fake_lc_docs, mock_answer, k=3):
    bm25 = BM25SparseRetriever()
    mock_qa = MagicMock()
    mock_qa.invoke.return_value = {
        "result": mock_answer,
        "source_documents": [Document(page_content="ctx", metadata={})] * k,
    }

    with (
        patch("src.langchain.chains.movie_rag.ChatOpenAI"),
        patch(
            "src.langchain.chains.movie_rag.MovieTextDocumentLoader"
        ) as mock_loader_cls,
        patch("src.langchain.chains.movie_rag.RetrievalQA") as mock_rqa,
    ):
        mock_loader_cls.return_value.load.return_value = fake_lc_docs
        mock_rqa.from_chain_type.return_value = mock_qa

        chain = MovieRAGChain(**base_chain_kwargs, custom_retriever=bm25, k=k)
        chain.build()

    return chain


def test_query_returns_answer(base_chain_kwargs, fake_lc_docs):
    chain = _built_chain(base_chain_kwargs, fake_lc_docs, "Inception is great.")
    result = chain.query("Best Nolan film?")
    assert result["answer"] == "Inception is great."


def test_query_returns_question(base_chain_kwargs, fake_lc_docs):
    chain = _built_chain(base_chain_kwargs, fake_lc_docs, "Some answer.")
    result = chain.query("What is noir?")
    assert result["question"] == "What is noir?"


def test_query_returns_sources(base_chain_kwargs, fake_lc_docs):
    chain = _built_chain(base_chain_kwargs, fake_lc_docs, "Answer.", k=3)
    result = chain.query("Query?")
    assert "sources" in result
    assert len(result["sources"]) == 3


def test_query_sources_have_content_and_metadata(base_chain_kwargs, fake_lc_docs):
    chain = _built_chain(base_chain_kwargs, fake_lc_docs, "Answer.", k=2)
    result = chain.query("Query?")
    for src in result["sources"]:
        assert "content" in src
        assert "metadata" in src


def test_query_without_sources(base_chain_kwargs, fake_lc_docs):
    chain = _built_chain(base_chain_kwargs, fake_lc_docs, "Answer.")
    result = chain.query("Query?", return_sources=False)
    assert "sources" not in result
