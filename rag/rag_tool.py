"""Unified RAGTool — the interface all agents call.

Receives HybridRetriever and optional KGIndex via constructor.
Graph expansion is transparent to callers — same retrieve() interface.
"""

from typing import List, Optional

from models import Evidence


class RAGTool:
    """Shared retrieval tool callable by any agent in the pipeline."""

    def __init__(self, retriever, kg_index=None):
        """Args:
            retriever: HybridRetriever instance.
            kg_index: Optional KGIndex for graph-expanded retrieval.
        """
        self._retriever = retriever
        self._kg = kg_index

    def retrieve(
        self,
        query: str,
        intent: str = "background",
        top_k: int = 5,
        crop_filter: Optional[str] = None,
        region_filter: Optional[str] = None,
    ) -> List[Evidence]:
        """Retrieve relevant evidence with optional graph expansion.

        If a KG index is loaded, the query is expanded via entity matching
        and 1-hop graph traversal before merging with base retrieval results.

        Args:
            query: Search query.
            intent: One of "background", "fact_check", "gap_fill".
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

        # Base hybrid retrieval (FAISS + BM25 + RRF)
        base_results = self._retriever.retrieve(query, top_k=top_k, similarity_threshold=threshold)

        # Graph expansion (if KG available)
        if self._kg:
            graph_results = self._kg.expand_query(query, top_k=top_k)
            if graph_results:
                base_results = self._merge_results(base_results, graph_results, top_k)

        # Apply metadata filters
        if crop_filter:
            crop_lower = crop_filter.lower()
            base_results = [r for r in base_results if crop_lower in r.content.lower()]

        if region_filter:
            region_lower = region_filter.lower()
            base_results = [r for r in base_results if region_lower in r.content.lower()]

        return base_results

    def _merge_results(
        self,
        base: List[Evidence],
        graph: List[Evidence],
        top_k: int,
    ) -> List[Evidence]:
        """Merge base and graph results, deduplicating by content similarity.

        Graph results get a boost to their relevance score to prioritize
        relationship-connected passages.
        """
        seen_content = set()
        merged = []

        # Graph results first (with score boost)
        for ev in graph:
            content_key = ev.content[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                # Boost graph results by 20% to prioritize relationship-connected evidence
                boosted = Evidence(
                    content=ev.content,
                    source=ev.source,
                    relevance_score=ev.relevance_score * 1.2,
                    metadata=ev.metadata,
                )
                merged.append(boosted)

        # Base results
        for ev in base:
            content_key = ev.content[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                merged.append(ev)

        # Sort by score, take top_k
        merged.sort(key=lambda e: -e.relevance_score)
        return merged[:top_k]
