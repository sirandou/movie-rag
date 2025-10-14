from typing import List

from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.langchain.chunk import chunk_documents
from src.langchain.retrieval.hyde import HyDERetriever
from src.retrievers.factory import create_text_retriever
from src.langchain.loaders import MovieTextDocumentLoader
from src.langchain.retrieval.retrievers import TextRetrieverWrapper
from langchain.prompts.base import BasePromptTemplate


class MovieRAGChain:
    """
    Flexible RAG chain that can use:
    1. LangChain retrievers (default) - FAISS with OpenAI embeddings
    2. Custom retrievers - wrapped with TextRetrieverWrapper
    - Supports custom chunking or LangChain chunking.
    - Uses RetrievalQA with "stuff" chain type.
    - Can use custom prompt templates.
    - Supports reranking retrievers via custom retrievers or langchain retrievers.
    """

    def __init__(
        self,
        # Data
        plots_path: str,
        reviews_path: str = None,
        max_movies: int = None,
        # Base Retriever
        use_custom_retriever: bool = True,
        custom_retriever=None,
        retriever_config: dict = {"type": "dense"},
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        use_custom_chunk: bool = True,
        chunk_config: dict | None = None,
        embed_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.0,
        k: int = 5,
        custom_prompt: BasePromptTemplate = None,
        # Reranking
        use_reranking: bool = False,
        reranker_cfg: dict = {"type": "cross-encoder"},
        initial_k: int = 20,
        # HyDE
        use_hyde: bool = False,
        hyde_model: str = "gpt-4o-mini",
        hyde_prompt: BasePromptTemplate = None,
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
            use_custom_chunk (bool): If True, use custom chunking logic.
            chunk_config (dict | None): Additional configuration for chunking.
            embed_model (str): Embedding model name.
            llm_model (str): LLM model name (OpenAI).
            llm_temperature (float): Temperature for the LLM.
            k (int): Number of results to retrieve.
            custom_prompt (BasePromptTemplate): Custom prompt template for the QA chain, default None.
            use_reranking (bool): If True, use a reranking retriever.
            reranker_cfg (dict): Configuration for the reranking retriever.
            initial_k (int): Initial number of documents to retrieve before reranking.
            use_hyde (bool): If True, use HyDE.
            hyde_model (str): Model to use for HyDE.
            hyde_prompt (BasePromptTemplate): Custom prompt template for HyDE, default None.
        """
        self.plots_path = plots_path
        self.reviews_path = reviews_path
        self.max_movies = max_movies
        self.use_custom_retriever = use_custom_retriever
        self.custom_retriever = custom_retriever
        self.retriever_config = retriever_config
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.custom_chunk = use_custom_chunk
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

        self.use_reranking = use_reranking
        self.reranker_cfg = reranker_cfg
        self.initial_k = initial_k
        self.use_hyde = use_hyde
        self.hyde_model = hyde_model
        self.hyde_prompt = hyde_prompt

        print("✓ MovieRAGChain initialized")
        print(
            f"  Retriever type: {'custom' if use_custom_retriever else 'langchain'}{' + reranking' if use_reranking else ''}{' + HyDE' if use_hyde else ''}"
        )
        print(f"  LLM: {llm_model}")

    def build(self):
        """Build the RAG pipeline."""
        print("\n" + "=" * 60)
        print("Building RAG Pipeline")
        print("=" * 60)

        # 1. Load documents using document creators
        documents = self._load_documents()

        # 2. Chunk documents (custom or LangChain)
        chunks = self._chunk_documents(documents)

        # 3. Build retriever (LangChain or Custom) and reranker if needed
        self.base_retriever = self._build_base_retriever(chunks)

        # Step 4: Apply features in order
        self.retriever = self._apply_retrieval_features(self.base_retriever)

        # 5. Create QA chain
        self._create_qa_chain()

        print("\n" + "=" * 60)
        print("✓ RAG Pipeline Built!")
        print("=" * 60)

        return self

    # ==================== INTERNAL METHODS ====================

    def _load_documents(self) -> list[Document]:
        """
        Load documents using the MovieTextDocumentLoader.

        Returns:
            list[Document]: Loaded documents.
        """
        print("\n1. Loading documents...")
        self.loader = MovieTextDocumentLoader(
            plots_path=self.plots_path,
            reviews_path=self.reviews_path,
            max_movies=self.max_movies,
        )
        return self.loader.load()

    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Chunk documents using either custom or LangChain chunking.

        Args:
            documents (list[Document]): Documents to chunk.

        Returns:
            list[Document]: Chunked documents.
        """
        if self.custom_chunk:
            print("\n2. Chunking with custom func...")
            chunks = chunk_documents(
                documents,
                chunking_strategy="sentence",
            )
        else:
            print("\n2. Chunking with LangChain...")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = self.text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def _build_base_retriever(self, chunks: List[Document]) -> BaseRetriever:
        """
        Build the base retriever for the RAG pipeline, either custom or LangChain.

        Args:
            chunks (List[Document]): The list of document chunks to index.

        Returns:
            None
        """
        print("\nBuilding base retriever...")
        retriever_k = self.initial_k if self.use_reranking else self.k
        if self.use_custom_retriever:
            return self._build_custom_retriever(chunks, k=retriever_k)
        else:
            return self._build_langchain_retriever(chunks, k=retriever_k)

    def _build_langchain_retriever(
        self, chunks: List[Document], k: int
    ) -> VectorStoreRetriever:
        """Build LangChain's built-in FAISS retriever."""
        print("\n3. Building LangChain FAISS retriever...")

        self.vectorstore = FAISS.from_documents(
            chunks, OpenAIEmbeddings(model=self.embed_model, chunk_size=1000)
        )

        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def _build_custom_retriever(
        self, chunks: List[Document], k: int
    ) -> TextRetrieverWrapper:
        """Build custom retriever wrapped for LangChain."""
        print("\n3. Building custom retriever...")

        if self.custom_retriever is None:
            self.custom_retriever = create_text_retriever(self.retriever_config)

        # Wrap retriever for LangChain (adds scores to metadata automatically)
        retriever = TextRetrieverWrapper(self.custom_retriever, k=k)

        # Add LangChain chunks (wrapper handles conversion!)
        retriever.add_documents(chunks)
        return retriever

    def _create_qa_chain(self) -> None:
        """
        Create the RetrievalQA chain with the configured LLM, retriever, and prompt.

        Raises:
            ValueError: If LLM or retriever is not set.
        """
        if self.llm is None or self.retriever is None:
            raise ValueError("LLM and retriever must be set before creating QA chain.")

        print("\n5. Creating QA chain...")
        chain_type_kwargs: dict = {}
        if self.custom_prompt is not None:
            chain_type_kwargs["prompt"] = self.custom_prompt

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs=chain_type_kwargs,
        )

    # ==================== FEATURE IMPLEMENTATIONS ====================

    def _apply_retrieval_features(self, base_retriever: BaseRetriever) -> BaseRetriever:
        """
        Apply retrieval features in order:
        1. Reranking (if enabled)
        """
        retriever = base_retriever
        step = 4

        # Feature 1: HyDE (query transformation - applied first)
        if self.use_hyde:
            print(f"\n{step}. Adding HyDE query transformation...")
            retriever = self._add_hyde(retriever)
            step += 1

        # Feature 2: Reranking (applied last - post-processes results)
        if self.use_reranking:
            print(f"\n{step}. Adding reranking: {self.initial_k} → {self.k} docs...")
            retriever = self._add_reranking(retriever, self.k)
            step += 1

        return retriever

    def _add_hyde(self, base_retriever: BaseRetriever) -> BaseRetriever:
        """Add HyDE query transformation."""
        return HyDERetriever(
            base_retriever=base_retriever,
            llm_model=self.hyde_model,
            hyde_prompt=self.hyde_prompt,
        )

    def _add_reranking(
        self, base_retriever: BaseRetriever, k: int
    ) -> ContextualCompressionRetriever:
        from src.langchain.retrieval.reranker import create_reranking_retriever

        """Build and set up the reranking retriever for the RAG pipeline."""
        return create_reranking_retriever(
            base_retriever, top_k=k, cfg=self.reranker_cfg
        )

    # ==================== PUBLIC API ====================

    def query(self, question: str, return_sources: bool = True) -> dict:
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

    def save(self, path: str) -> None:
        """Save retriever state."""
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.use_custom_retriever and hasattr(self.retriever, "retriever"):
            self.retriever.retriever.save(path / "custom_retriever")
            print(f"✓ Saved custom retriever to {path}")
        else:
            print("  LangChain retriever (no save needed - rebuild from data)")
