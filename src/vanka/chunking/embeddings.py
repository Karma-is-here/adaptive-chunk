from __future__ import annotations

from typing import Sequence


class SentenceTransformerEmbedder:
    """Local sentence-transformer implementation of the Embedder interface."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence-transformers or one of its "
                "dependencies. Check the original traceback."
            ) from exc

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
