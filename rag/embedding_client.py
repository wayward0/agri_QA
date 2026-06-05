"""Embedding client — API-based replacement for SentenceTransformer.

Supports OpenAI-compatible embedding endpoints (e.g., bge-m3 via moark.com).
"""

from typing import List
from openai import OpenAI


class EmbeddingClient:
    """API-based embedding client compatible with SentenceTransformer interface."""

    def __init__(self, base_url: str, api_key: str, model: str = "bge-m3"):
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers={"X-Failover-Enabled": "true"},
        )
        self._model = model
        self._dimension = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            test = self.encode(["test"])
            self._dimension = len(test[0])
        return self._dimension

    def encode(
        self,
        texts: List[str],
        normalize_embeddings: bool = True,
        batch_size: int = 16,
    ) -> List[List[float]]:
        """Encode texts to embeddings. Compatible with SentenceTransformer.encode().

        Args:
            texts: List of strings to embed.
            normalize_embeddings: Ignored (API handles normalization).
            batch_size: Batch size for API calls.

        Returns:
            List of embedding vectors.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._client.embeddings.create(
                input=batch,
                model=self._model,
            )
            for item in response.data:
                all_embeddings.append(item.embedding)
        return all_embeddings
