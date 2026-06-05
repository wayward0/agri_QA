"""Hybrid retriever: FAISS (dense) + BM25 (sparse) + RRF fusion.

Receives indices via constructor — no file I/O.
"""

from typing import List, Optional

import faiss
import numpy as np

from models import Evidence


class HybridRetriever:
    """Hybrid dense + sparse retriever with Reciprocal Rank Fusion."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        bm25_model,
        metadata: List[dict],
        embedding_model,
        rrf_k: int = 60,
        reranker=None,
    ):
        """All dependencies injected — no file I/O in constructor.

        Args:
            faiss_index: Pre-built FAISS index.
            bm25_model: Pre-built BM25Okapi model.
            metadata: List of passage metadata dicts (must include 'text').
            embedding_model: Embedding model for query encoding (must have .encode()).
            rrf_k: RRF constant (default 60).
            reranker: Optional RerankerClient for second-stage reranking.
        """
        self._faiss = faiss_index
        self._bm25 = bm25_model
        self._metadata = metadata
        self._embedder = embedding_model
        self._rrf_k = rrf_k
        self._reranker = reranker

    def _dense_search(self, query: str, top_k: int) -> List[tuple]:
        """FAISS search. Returns [(index, score), ...]."""
        query_vec = self._embedder.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")
        scores, indices = self._faiss.search(query_vec, min(top_k, len(self._metadata)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                results.append((int(idx), float(score)))
        return results

    def _sparse_search(self, query: str, top_k: int) -> List[tuple]:
        """BM25 search. Returns [(index, score), ...]."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx])))
        return results

    def _rrf_fusion(
        self,
        dense_results: List[tuple],
        sparse_results: List[tuple],
    ) -> List[Evidence]:
        """Reciprocal Rank Fusion of dense and sparse results.

        RRF score = sum(1 / (k + rank)) across result lists.
        """
        rrf_scores = {}

        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self._rrf_k + rank + 1)

        for rank, (idx, _) in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self._rrf_k + rank + 1)

        # Sort by RRF score descending
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        results = []
        for idx in sorted_indices:
            meta = self._metadata[idx]
            results.append(Evidence(
                content=meta["text"],
                source=f"Wikipedia: {meta['article_title']}, {meta['section_title']}",
                relevance_score=round(rrf_scores[idx], 4),
                metadata={"passage_id": meta.get("id", ""), "url": meta.get("url", "")},
            ))
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> List[Evidence]:
        """Dense + sparse retrieval with RRF fusion + optional reranking.

        Pipeline: FAISS + BM25 → RRF fusion → [reranker] → top-k.

        Args:
            query: Search query.
            top_k: Number of results to return.
            similarity_threshold: Minimum RRF score to include.

        Returns:
            List of Evidence objects, sorted by relevance.
        """
        # First stage: retrieve more candidates for reranking
        # Reranker API limit: max 25 documents
        candidate_k = min(top_k * 4, 25) if self._reranker else top_k * 2
        dense = self._dense_search(query, candidate_k)
        sparse = self._sparse_search(query, candidate_k)
        fused = self._rrf_fusion(dense, sparse)
        filtered = [r for r in fused if r.relevance_score >= similarity_threshold]

        # Second stage: rerank if available
        if self._reranker and len(filtered) > top_k:
            documents = [r.content for r in filtered]
            reranked = self._reranker.rerank(query, documents, top_k=top_k)
            results = []
            for item in reranked:
                idx = item["index"]
                if idx < len(filtered):
                    evidence = filtered[idx]
                    evidence.relevance_score = round(item["relevance_score"], 4)
                    results.append(evidence)
            return results

        return filtered[:top_k]

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> List[List[Evidence]]:
        """Batch retrieval: one embedding call, per-query BM25 + FAISS + rerank.

        Args:
            queries: List of search queries.
            top_k: Number of results per query.
            similarity_threshold: Minimum RRF score to include.

        Returns:
            List of result lists, one per query.
        """
        if not queries:
            return []

        # Batch embed all queries in a single API call
        query_vecs = self._embedder.encode(queries, normalize_embeddings=True)
        query_vecs = np.array(query_vecs, dtype="float32")

        candidate_k = min(top_k * 4, 25) if self._reranker else top_k * 2

        all_results = []
        for i, query in enumerate(queries):
            # Dense search with pre-computed vector
            scores, indices = self._faiss.search(
                query_vecs[i:i+1], min(candidate_k, len(self._metadata))
            )
            dense = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    dense.append((int(idx), float(score)))

            # Sparse search
            sparse = self._sparse_search(query, candidate_k)

            # RRF fusion
            fused = self._rrf_fusion(dense, sparse)
            filtered = [r for r in fused if r.relevance_score >= similarity_threshold]

            # Rerank if available
            if self._reranker and len(filtered) > top_k:
                documents = [r.content for r in filtered]
                reranked = self._reranker.rerank(query, documents, top_k=top_k)
                results = []
                for item in reranked:
                    idx = item["index"]
                    if idx < len(filtered):
                        evidence = filtered[idx]
                        evidence.relevance_score = round(item["relevance_score"], 4)
                        results.append(evidence)
                all_results.append(results)
            else:
                all_results.append(filtered[:top_k])

        return all_results
