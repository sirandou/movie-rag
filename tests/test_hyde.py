import os
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_openai import ChatOpenAI

from src.langchain.prompts import HYDE_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeBaseRetriever(BaseRetriever):
    """Minimal BaseRetriever returning canned documents."""

    docs: list = []

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ):
        return list(self.docs)


def _make_hyde(base_docs, k=5):
    """
    Build a HyDERetriever with a real ChatOpenAI (fake API key).
    Patching llm.invoke is done at the ChatOpenAI class level (Pydantic v2
    forbids direct instance attribute assignment for non-field methods).
    """
    from src.langchain.retrieval.hyde import HyDERetriever

    base_retriever = FakeBaseRetriever(docs=base_docs)
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key-for-testing"}):
        return HyDERetriever(base_retriever=base_retriever, k=k)


@pytest.fixture
def base_docs():
    return [
        Document(page_content=f"Movie document {i}.", metadata={"id": i})
        for i in range(6)
    ]


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init_stores_k(base_docs):
    retriever = _make_hyde(base_docs, k=3)
    assert retriever.k == 3


def test_init_uses_default_hyde_prompt(base_docs):
    retriever = _make_hyde(base_docs)
    assert retriever.hyde_prompt is HYDE_PROMPT


def test_init_accepts_custom_prompt(base_docs):
    from langchain.prompts import PromptTemplate
    from src.langchain.retrieval.hyde import HyDERetriever

    custom = PromptTemplate(
        input_variables=["pre_hyde_query"], template="Q: {pre_hyde_query}"
    )
    base_retriever = FakeBaseRetriever(docs=base_docs)
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
        retriever = HyDERetriever(base_retriever=base_retriever, hyde_prompt=custom)
    assert retriever.hyde_prompt is custom


# ---------------------------------------------------------------------------
# _generate_hypothetical_doc  (patch ChatOpenAI.invoke at class level)
# ---------------------------------------------------------------------------


def test_generate_returns_llm_content(base_docs):
    retriever = _make_hyde(base_docs)
    mock_resp = MagicMock()
    mock_resp.content = "  Hypothetical answer.  "
    with patch.object(ChatOpenAI, "invoke", return_value=mock_resp):
        result = retriever._generate_hypothetical_doc("Best sci-fi?")
    assert result == "Hypothetical answer."  # strips whitespace


def test_generate_calls_llm_once(base_docs):
    retriever = _make_hyde(base_docs)
    mock_resp = MagicMock()
    mock_resp.content = "Answer."
    with patch.object(ChatOpenAI, "invoke", return_value=mock_resp) as mock_invoke:
        retriever._generate_hypothetical_doc("Q?")
    mock_invoke.assert_called_once()


# ---------------------------------------------------------------------------
# _get_relevant_documents  (patch _generate_hypothetical_doc directly)
# ---------------------------------------------------------------------------


def test_get_relevant_documents_returns_documents(base_docs):
    retriever = _make_hyde(base_docs)
    with patch.object(retriever, "_generate_hypothetical_doc", return_value="Hypo."):
        results = retriever._get_relevant_documents("action films")
    assert all(isinstance(d, Document) for d in results)


def test_get_relevant_documents_respects_k(base_docs):
    retriever = _make_hyde(base_docs, k=3)
    with patch.object(retriever, "_generate_hypothetical_doc", return_value="Hypo."):
        results = retriever._get_relevant_documents("horror films")
    assert len(results) <= 3


def test_get_relevant_documents_adds_hyde_metadata(base_docs):
    retriever = _make_hyde(base_docs)
    with patch.object(
        retriever, "_generate_hypothetical_doc", return_value="The answer is X."
    ):
        results = retriever._get_relevant_documents("Q?")
    for doc in results:
        assert "hyde_query" in doc.metadata
        assert doc.metadata["hyde_query"] == "The answer is X."


def test_base_retriever_called_with_hypothetical_not_original(base_docs):
    """The base retriever must receive the hypothetical doc, not the raw query."""
    calls = []

    class TrackingRetriever(BaseRetriever):
        docs: list = []

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(self, query, *, run_manager=None):
            calls.append(query)
            return list(self.docs)

    from src.langchain.retrieval.hyde import HyDERetriever

    base = TrackingRetriever(docs=base_docs)
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
        retriever = HyDERetriever(base_retriever=base)

    with patch.object(
        retriever, "_generate_hypothetical_doc", return_value="Hypothetical answer."
    ):
        retriever._get_relevant_documents("Original user query")

    assert calls == ["Hypothetical answer."]


# ---------------------------------------------------------------------------
# k=0 edge case
# ---------------------------------------------------------------------------


def test_k_zero_returns_empty(base_docs):
    retriever = _make_hyde(base_docs, k=0)
    with patch.object(retriever, "_generate_hypothetical_doc", return_value="Hypo."):
        results = retriever._get_relevant_documents("anything")
    assert results == []
