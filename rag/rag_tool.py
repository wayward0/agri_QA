"""Unified RAGTool — the interface all agents call.

Receives HybridRetriever via constructor.
"""

from typing import List, Optional

from models import Evidence


class RAGTool:
    """Shared retrieval tool callable by any agent in the pipeline."""

    def __init__(self, retriever):
        """Args:
            retriever: HybridRetriever instance.
        """
        self._retriever = retriever

    def retrieve(
        self,
        query: str,
        intent: str = "background",
        top_k: int = 5,
        crop_filter: Optional[str] = None,
        region_filter: Optional[str] = None,
    ) -> List[Evidence]:
        """Retrieve relevant evidence.

        Args:
            query: Search query.
            intent: One of "background", "fact_check", "gap_fill".
                - background: lower threshold (0.0), more results
                - fact_check: higher threshold (0.3), precise matching
                - gap_fill: medium threshold (0.15)
            top_k: Number of results to return.
            crop_filter: Optional crop name to filter results.
            region_filter: Optional region name to filter results.

        Returns:
            List of Evidence objects.
        """
        threshold = {
            "background": 0.0,
            "fact_check": 0.0,
            "gap_fill": 0.0,
        }.get(intent, 0.0)

        results = self._retriever.retrieve(query, top_k=top_k, similarity_threshold=threshold)

        if crop_filter:
            crop_lower = crop_filter.lower()
            results = [r for r in results if crop_lower in r.content.lower()]

        if region_filter:
            region_lower = region_filter.lower()
            results = [r for r in results if region_lower in r.content.lower()]

        return results
