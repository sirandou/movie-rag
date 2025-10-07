from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.langchain.chunk import chunk_documents
from src.retrievers.factory import create_text_retriever
from src.langchain.loaders import MovieTextDocumentLoader
from src.langchain.retrievers import TextRetrieverWrapper
from langchain.prompts.base import BasePromptTemplate


class MovieRAGChain:
    """
    Flexible RAG chain that can use:
    1. LangChain retrievers (default) - FAISS with OpenAI embeddings
    2. Custom retrievers - wrapped with TextRetrieverWrapper
    - Supports custom chunking or LangChain chunking.
    - Uses RetrievalQA with "stuff" chain type.
    - Can use custom prompt templates.
    """

    def __init__(
        self,
        plots_path: str,
        reviews_path: str = None,
        max_movies: int = None,
        use_custom_retriever: bool = True,
        custom_retriever=None,
        retriever_config: dict = {"type": "dense"},
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        custom_chunk: bool = True,
        chunk_config: dict | None = None,
        embed_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.0,
        k: int = 5,
        custom_prompt: BasePromptTemplate | None = None,
    ):
        """
        Initialize simple RAG chain.

        Args:
            plots_path (str): Path to the plots CSV file.
            reviews_path (str, optional): Path to the reviews CSV file.
            max_movies (int, optional): Maximum number of movies to load (for testing).
            use_custom_retriever (bool): If True, use a custom retriever instead of LangChain.
            custom_retriever (Any, optional): Pre-built custom retriever instance.
            retriever_config (dict): Configuration for the custom retriever (if custom_retriever is None).
            chunk_size (int): Chunk size for the text splitter.
            chunk_overlap (int): Overlap size between text chunks.
            custom_chunk (bool): If True, use custom chunking logic.
            chunk_config (dict | None): Additional configuration for chunking.
            embed_model (str): Embedding model name.
            llm_model (str): LLM model name (OpenAI).
            llm_temperature (float): Temperature for the LLM.
            k (int): Number of results to retrieve.
            custom_prompt (BasePromptTemplate | None): Custom prompt template for the QA chain.
        """
        self.plots_path = plots_path
        self.reviews_path = reviews_path
        self.max_movies = max_movies
        self.use_custom_retriever = use_custom_retriever
        self.custom_retriever = custom_retriever
        self.retriever_config = retriever_config
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.custom_chunk = custom_chunk
        self.chunk_config = chunk_config

        self.loader = None
        self.text_splitter = None
        self.retriever = None
        self.vectorstore = None  # Store for score access if needed
        self.qa_chain = None
        self.embed_model = embed_model
        self.llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)
        self.k = k
        self.custom_prompt = custom_prompt

        retriever_type = "custom" if use_custom_retriever else "langchain"
        print("✓ MovieRAGChain initialized")
        print(f"  Retriever type: {retriever_type}")
        print(f"  LLM: {llm_model}")

    def build(self):
        """Build the RAG pipeline."""
        print("\n" + "=" * 60)
        print("Building RAG Pipeline")
        print("=" * 60)

        # 1. Load documents using document creators
        print("\n1. Loading documents...")
        self.loader = MovieTextDocumentLoader(
            plots_path=self.plots_path,
            reviews_path=self.reviews_path,
            max_movies=self.max_movies,
        )
        documents = self.loader.load()

        if self.custom_chunk:
            # 2. Custom chunk
            print("\n2. Chunking with custom func...")
            chunks = chunk_documents(
                documents,
                chunking_strategy="sentence",
            )
        else:
            # 2. Chunk with LangChain
            print("\n2. Chunking with LangChain...")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = self.text_splitter.split_documents(documents)
            print(f"  Created {len(chunks)} chunks from {len(documents)} documents")

        # 3. Build retriever (LangChain or Custom)
        if self.use_custom_retriever:
            self._build_custom_retriever(chunks)
        else:
            self._build_langchain_retriever(chunks)

        # 4. Create QA chain
        print("\n4. Creating QA chain...")
        if self.custom_prompt is None:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                return_source_documents=True,
                chain_type="stuff",
            )
        else:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                return_source_documents=True,
                chain_type="stuff",
                chain_type_kwargs={"prompt": self.custom_prompt},
            )

        print("\n" + "=" * 60)
        print("✓ RAG Pipeline Built!")
        print("=" * 60)

        return self

    def _build_langchain_retriever(self, chunks: List[Document]):
        """Build LangChain's built-in FAISS retriever."""
        print("\n3. Building LangChain FAISS retriever...")

        self.vectorstore = FAISS.from_documents(
            chunks, OpenAIEmbeddings(model=self.embed_model, chunk_size=1000)
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

    def _build_custom_retriever(self, chunks: List[Document]):
        """Build custom retriever wrapped for LangChain."""
        print("\n3. Building custom retriever...")

        if self.custom_retriever is None:
            self.custom_retriever = create_text_retriever(self.retriever_config)

        # Wrap retriever for LangChain (adds scores to metadata automatically)
        self.retriever = TextRetrieverWrapper(self.custom_retriever, k=self.k)

        # Add LangChain chunks (wrapper handles conversion!)
        self.retriever.add_documents(chunks)

    def query(self, question: str, return_sources: bool = True):
        """
        Query the RAG system.

        Args:
            question: User question
            return_sources: Return source documents

        Returns:
            Dict with answer and sources
        """
        if self.qa_chain is None:
            raise ValueError("Pipeline not built. Call build() first.")

        result = self.qa_chain.invoke({"query": question})

        response = {"answer": result["result"], "question": question}

        if return_sources and "source_documents" in result:
            response["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }  # Note: 'score' in metadata if custom retriever, not present if LangChain retriever (can add later)
                for doc in result["source_documents"][: self.k]
            ]

        return response

    def save(self, path: str):
        """Save retriever state."""
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.use_custom_retriever and hasattr(self.retriever, "retriever"):
            self.retriever.retriever.save(path / "custom_retriever")
            print(f"✓ Saved custom retriever to {path}")
        else:
            print("  LangChain retriever (no save needed - rebuild from data)")
