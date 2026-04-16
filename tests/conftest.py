import numpy as np
import pytest

EMBED_DIM = 8


@pytest.fixture
def sample_docs():
    return [
        {
            "text": "Action thriller with great visual effects and stunts.",
            "metadata": {"doc_id": 0, "chunk_id": 0, "movie_title": "Action Movie"},
        },
        {
            "text": "Romantic comedy with wonderful chemistry between the lead actors.",
            "metadata": {"doc_id": 1, "chunk_id": 0, "movie_title": "Romance Film"},
        },
        {
            "text": "Terrifying horror film full of suspense and jump scares.",
            "metadata": {"doc_id": 2, "chunk_id": 0, "movie_title": "Horror Movie"},
        },
        {
            "text": "Science fiction epic exploring deep space and alien civilizations.",
            "metadata": {"doc_id": 3, "chunk_id": 0, "movie_title": "Sci-Fi Film"},
        },
        {
            "text": "Documentary following wildlife and nature conservation efforts.",
            "metadata": {"doc_id": 4, "chunk_id": 0, "movie_title": "Nature Doc"},
        },
    ]


class FakeEmbeddingModel:
    """Deterministic fake embedding model for testing (no model downloads)."""

    def __init__(self, model_name="all-MiniLM-L6-v2", provider="sentence-transformers"):
        self.model_name = model_name
        self.provider = provider
        self.dimension = EMBED_DIM

    def encode(self, texts, batch_size=32, show_progress=True):
        if isinstance(texts, str):
            texts = [texts]
        result = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            vec = rng.random(EMBED_DIM).astype(np.float32)
            result.append(vec)
        return np.array(result, dtype=np.float32)

    def get_dimension(self):
        return EMBED_DIM


@pytest.fixture
def fake_embedding_cls():
    return FakeEmbeddingModel
