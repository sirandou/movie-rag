import pickle
from typing import Literal

import faiss
import numpy as np
from path import Path

from src.retrievers.base import BaseRetriever
from src.utils.clip_embeddings import CLIPEmbedding


class VisualRetriever(BaseRetriever):
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        use_text_fusion: bool = True,
        fusion_method: Literal["concat", "weight_average"] = "weight_average",
        alpha: float = 0.5,
    ):
        """
        Args:
            use_text_fusion: Whether to fuse poster + movie metadata
            fusion_method: "concat" or "weight_average"
        """
        self.model_name = model_name
        self.clip = CLIPEmbedding(model_name=model_name)
        self.use_text_fusion = use_text_fusion
        self.fusion_method = fusion_method
        self.alpha = 0.5  # For weighted average fusion
        self.index = None
        self.documents = []

        print(
            f"✓ VisualRetriever (text_fusion={use_text_fusion}, method={fusion_method})"
        )
        if fusion_method == "weight_average":
            print(f"  Alpha: {alpha} (image={alpha:.1f}, text={1 - alpha:.1f})")

    def add_documents(self, documents: list[dict]) -> None:
        """
        Add poster documents with optional text fusion.

        Args:
            documents: List with 'poster_path' and 'metadata_text'
        """
        poster_paths = [doc["poster_path"] for doc in documents]

        # Encode posters
        print(f"Encoding {len(poster_paths)} posters with CLIP...")
        image_embeddings = self.clip.encode_images(poster_paths)

        if self.use_text_fusion:
            text_descriptions = [doc["text_content"] for doc in documents]

            # Encode text with CLIP
            print(f"Encoding {len(text_descriptions)} text descriptions with CLIP...")
            text_embeddings = self.clip.encode_text(text_descriptions)

            # Fuse embeddings
            if self.fusion_method == "concat":
                # Concatenate [512 + 512 = 1024 dim]
                final_embeddings = np.concatenate(
                    [image_embeddings, text_embeddings], axis=1
                )
                dimension = self.clip.get_dimension() * 2
            elif self.fusion_method == "weight_average":
                # Weighted average [512 dim]
                final_embeddings = (
                    self.alpha * image_embeddings + (1 - self.alpha) * text_embeddings
                )
                dimension = self.clip.get_dimension()
            else:
                raise ValueError(f"Unknown fusion_method: {self.fusion_method}")

            print(
                f"✓ Fused embeddings using {self.fusion_method} method (dim={dimension})"
            )
        else:
            # Just use image embeddings
            final_embeddings = image_embeddings
            dimension = self.clip.get_dimension()

        # Create FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(final_embeddings)
        self.documents = documents

        print(f"✓ Added {len(documents)} posters to index")

    def search(self, query: str, k: int = 5, **kwargs) -> list[tuple[dict, float]]:
        """Search posters using text description."""
        if self.index is None:
            raise ValueError("No index created. Call add_documents() first.")

        # Encode query with CLIP text encoder
        query_text_emb = self.clip.encode_text([query])[0]

        if self.use_text_fusion:
            if self.fusion_method == "concat":
                # For concat: duplicate query embedding to match dimension
                # [512] → [1024] by concatenating with itself
                query_emb = np.concatenate([query_text_emb, query_text_emb]).reshape(
                    1, -1
                )
            elif self.fusion_method == "weight_average":
                # For weighted average: just use text embedding as-is
                query_emb = query_text_emb.reshape(1, -1)
        else:
            query_emb = query_text_emb.reshape(1, -1)

        # Search
        distances, indices = self.index.search(query_emb, k)

        # Results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append((self.documents[idx], float(dist)))

        return results

    def save(self, path: str) -> None:
        """
        Save visual retriever to disk.

        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path / "clip.index"
        faiss.write_index(self.index, str(index_path))

        # Save documents
        docs_path = path / "documents.pkl"
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

        # Save configuration
        config_path = path / "config.pkl"
        config = {
            "use_text_fusion": self.use_text_fusion,
            "fusion_method": self.fusion_method,
            "alpha": self.alpha,
            "clip_model": self.clip.model_name
            if hasattr(self.clip, "model_name")
            else "ViT-B/32",
            "num_documents": len(self.documents),
        }
        with open(config_path, "wb") as f:
            pickle.dump(config, f)

        print(f"✓ Saved VisualRetriever to {path}")
        print(f"  Index: {index_path}")
        print(f"  Documents: {len(self.documents)}")

    def load(self, path: str) -> None:
        """
        Load visual retriever from disk.

        Args:
            path: Directory path to load from
        """
        path = Path(path)

        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")

        # Load configuration
        config_path = path / "config.pkl"
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        # Restore configuration
        self.use_text_fusion = config["use_text_fusion"]
        self.fusion_method = config["fusion_method"]
        self.alpha = config["alpha"]

        # Reinitialize CLIP model (need to reload the model)
        clip_model = config.get("clip_model", "ViT-B/32")
        self.clip = CLIPEmbedding(model_name=clip_model)

        # Load FAISS index
        index_path = path / "clip.index"
        self.index = faiss.read_index(str(index_path))

        # Load documents
        docs_path = path / "documents.pkl"
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)

        print(f"✓ Loaded VisualRetriever from {path}")
        print(f"  Documents: {len(self.documents)}")
        print(f"  Text fusion: {self.use_text_fusion}")
        print(f"  Method: {self.fusion_method}")
        if self.fusion_method == "average":
            print(f"  Alpha: {self.alpha}")

    def __repr__(self) -> str:
        num_docs = len(self.documents) if self.documents else 0
        return (
            f"VisualRetriever(docs={num_docs}, "
            f"fusion={self.use_text_fusion}, "
            f"method={self.fusion_method})"
        )
