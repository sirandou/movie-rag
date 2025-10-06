import torch
import clip
from PIL import Image
import numpy as np
from typing import List


class CLIPEmbedding:
    """Simple CLIP model for poster embeddings, and text embedding for movie descriptions or text query."""

    def __init__(self, model_name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        dimension_map = {
            "ViT-B/32": 512,
            "ViT-B/16": 512,
        }
        self.dimension = dimension_map.get(self.model_name)
        if self.dimension is None:
            raise ValueError(f"Unknown CLIP model: {self.model_name}")

        print(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print(f"✓ CLIP loaded on {self.device}")

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """Encode images to embeddings."""
        embeddings = []
        for path in image_paths:
            image = Image.open(path)
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            embeddings.append(embedding.cpu().numpy())

        return np.vstack(embeddings).astype("float32")

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text descriptions to embeddings."""
        text_tokens = clip.tokenize(texts).to(self.device)

        with torch.no_grad():
            embeddings = self.model.encode_text(text_tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.cpu().numpy().astype("float32")

    def get_dimension(self) -> int:
        return self.dimension
