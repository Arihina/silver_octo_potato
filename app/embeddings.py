from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_passages(self, texts: List[str]) -> np.ndarray:
        prefixed = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [f"query: {text}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vec


embedder = Embedder()
