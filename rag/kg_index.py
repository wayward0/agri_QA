"""Knowledge Graph index — query-time entity matching and graph expansion.

Loads pre-built KG from JSON files, provides:
- Entity matching via vector search (entity_faiss)
- 1-hop graph traversal
- Passage collection with relevance scoring
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict

import faiss
import numpy as np

from models import Evidence


class KGIndex:
    """Knowledge graph index for query-time expansion."""

    def __init__(
        self,
        entities: List[Dict],
        relations: List[Dict],
        entity_faiss: faiss.Index,
        entity_map: List[int],
        passage_metadata: List[Dict],
        embedding_model,
        top_k_entities: int = 5,
        max_graph_passages: int = 10,
    ):
        self._entities = entities
        self._relations = relations
        self._entity_faiss = entity_faiss
        self._entity_map = entity_map
        self._embedding_model = embedding_model
        self._top_k_entities = top_k_entities
        self._max_graph_passages = max_graph_passages

        # Build lookup structures
        self._name_to_entity = {}
        for ent in entities:
            self._name_to_entity[ent["name"]] = ent
            for alias in ent.get("aliases", []):
                self._name_to_entity[alias] = ent

        # Build adjacency: entity_name -> [(relation, neighbor_name, confidence)]
        self._adjacency = defaultdict(list)
        for rel in relations:
            self._adjacency[rel["source"]].append(
                (rel["relation"], rel["target"], rel["confidence"], rel.get("evidence_passage_ids", []))
            )
            # Reverse for undirected traversal
            self._adjacency[rel["target"]].append(
                (rel["relation"] + "_rev", rel["source"], rel["confidence"], rel.get("evidence_passage_ids", []))
            )

        # Build passage_id -> passage metadata lookup
        self._passage_lookup = {}
        for p in passage_metadata:
            self._passage_lookup[p["id"]] = p

    @classmethod
    def load(
        cls,
        kg_dir: str,
        passage_metadata: List[Dict],
        embedding_model,
        top_k_entities: int = 5,
        max_graph_passages: int = 10,
    ) -> Optional["KGIndex"]:
        """Load KG from directory containing kg_entities.json, kg_relations.json, entity_faiss.index.

        Returns None if files don't exist.
        """
        kg_path = Path(kg_dir)
        entities_file = kg_path / "kg_entities.json"
        relations_file = kg_path / "kg_relations.json"
        faiss_file = kg_path / "entity_faiss.index"
        map_file = kg_path / "entity_faiss_map.json"

        if not all(f.exists() for f in [entities_file, relations_file, faiss_file, map_file]):
            return None

        with open(entities_file, "r", encoding="utf-8") as f:
            entities = json.load(f)
        with open(relations_file, "r", encoding="utf-8") as f:
            relations = json.load(f)
        with open(map_file, "r", encoding="utf-8") as f:
            entity_map = json.load(f)

        entity_faiss = faiss.read_index(str(faiss_file))

        return cls(
            entities=entities,
            relations=relations,
            entity_faiss=entity_faiss,
            entity_map=entity_map,
            passage_metadata=passage_metadata,
            embedding_model=embedding_model,
            top_k_entities=top_k_entities,
            max_graph_passages=max_graph_passages,
        )

    def _match_entities(self, query: str) -> List[Dict]:
        """Match query to entities via vector search."""
        if self._entity_faiss.ntotal == 0:
            return []

        query_vec = self._embedding_model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")

        k = min(self._top_k_entities, self._entity_faiss.ntotal)
        scores, indices = self._entity_faiss.search(query_vec, k)

        matched = []
        seen = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._entity_map):
                continue
            entity_idx = self._entity_map[idx]
            entity = self._entities[entity_idx]
            if entity["name"] not in seen:
                seen.add(entity["name"])
                matched.append({
                    "entity": entity,
                    "match_score": float(score),
                })

        return matched

    def _get_neighbors(self, entity_name: str) -> List[Dict]:
        """Get 1-hop neighbors of an entity."""
        neighbors = []
        for relation, neighbor_name, confidence, passage_ids in self._adjacency.get(entity_name, []):
            neighbor_ent = self._name_to_entity.get(neighbor_name)
            neighbors.append({
                "name": neighbor_name,
                "relation": relation,
                "confidence": confidence,
                "passage_ids": passage_ids,
                "entity": neighbor_ent,
            })
        return neighbors

    def expand_query(self, query: str, top_k: int = 5) -> List[Evidence]:
        """Expand query using KG: match entities, traverse graph, collect passages.

        Args:
            query: The search query.
            top_k: Max number of Evidence objects to return.

        Returns:
            List of Evidence from graph-expanded passages.
        """
        # Step 1: Match query to entities
        matched = self._match_entities(query)
        if not matched:
            return []

        # Step 2: 1-hop traversal + collect passage_ids with scores
        passage_scores = defaultdict(float)
        passage_reasons = defaultdict(list)

        for match in matched:
            entity = match["entity"]
            match_score = match["match_score"]

            # Passages directly mentioning this entity
            for pid in entity.get("passage_ids", []):
                passage_scores[pid] += match_score * 1.0
                passage_reasons[pid].append(f"entity:{entity['name']}")

            # Passages from 1-hop neighbors
            for neighbor in self._get_neighbors(entity["name"]):
                # Score: match_score * relation_confidence * distance_decay
                hop_score = match_score * neighbor["confidence"] * 0.5
                for pid in neighbor.get("passage_ids", []):
                    passage_scores[pid] += hop_score
                    passage_reasons[pid].append(
                        f"{entity['name']}--{neighbor['relation']}-->{neighbor['name']}"
                    )

        if not passage_scores:
            return []

        # Step 3: Rank passages by score, take top-k
        ranked = sorted(passage_scores.items(), key=lambda x: -x[1])[:self._max_graph_passages]

        # Step 4: Convert to Evidence objects
        evidence_list = []
        for pid, score in ranked:
            passage = self._passage_lookup.get(pid)
            if not passage:
                continue
            source = f"Wikipedia: {passage.get('article_title', '')}, {passage.get('section_title', '')}"
            reason = "; ".join(set(passage_reasons[pid]))
            evidence_list.append(Evidence(
                content=passage["text"],
                source=source,
                relevance_score=score,
                metadata={
                    "passage_id": pid,
                    "url": passage.get("url", ""),
                    "kg_reason": reason,
                },
            ))

        return evidence_list[:top_k]

    def get_entity_info(self, name: str) -> Optional[Dict]:
        """Get entity details by name (exact match)."""
        return self._name_to_entity.get(name.lower())

    def get_relations(self, entity_name: str) -> List[Dict]:
        """Get all relations for an entity."""
        return [
            {"relation": rel, "target": tgt, "confidence": conf}
            for rel, tgt, conf, _ in self._adjacency.get(entity_name, [])
        ]
