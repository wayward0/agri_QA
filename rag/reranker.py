"""Reranker client — API-based cross-encoder reranking.

Uses moark.com rerank endpoint (bge-reranker-v2-m3).
"""

from typing import List
import requests


class RerankerClient:
    """API-based reranker for second-stage retrieval refinement."""

    def __init__(
        self,
        base_url: str = "https://api.moark.com/v1/rerank",
        api_key: str = "",
        model: str = "bge-reranker-v2-m3",
        instruction: str = "",
    ):
        self._url = base_url
        self._headers = {
            "X-Failover-Enabled": "true",
            "Authorization": f"Bearer {api_key}",
        }
        self._model = model
        self._instruction = instruction

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None,
    ) -> List[dict]:
        """Rerank documents against a query.

        Args:
            query: The search query.
            documents: List of document strings to rerank.
            top_k: Number of top results to return. None = return all.

        Returns:
            List of dicts with 'index' (int) and 'relevance_score' (float),
            sorted by relevance_score descending.
        """
        if not documents:
            return []

        payload = {
            "query": query,
            "documents": documents,
            "model": self._model,
        }
        if self._instruction:
            payload["instruction"] = self._instruction

        try:
            response = requests.post(
                self._url,
                headers=self._headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if top_k is not None:
                results = results[:top_k]
            return results
        except Exception as e:
            # Graceful fallback: return documents in original order
            fallback = [
                {"index": i, "relevance_score": 1.0 / (i + 1)}
                for i in range(len(documents))
            ]
            if top_k is not None:
                fallback = fallback[:top_k]
            return fallback
