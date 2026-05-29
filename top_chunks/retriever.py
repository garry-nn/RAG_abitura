from typing import List, Dict
from qdrant_client import QdrantClient

from embeddings.embedder import Embedder


class TopChunksRetriever:

    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.embedder = Embedder()

    # SCORE NORMALIZATION
    def _normalize_score(self, chunk: Dict) -> float:
        score = chunk.get("score", 0.0)

        # CQ чуть усиливаем (как anchor retrieval)
        if chunk.get("retrieval_type") == "CQ":
            return score * 1.05

        return score


    # MAIN METHOD
    def get_top_chunks(self, query: str) -> List[Dict]:

        query_vector = self.embedder.embedding(query)

        # 1. CQ SEARCH
        cq_results = self.client.query_points(
            collection_name="query_index",
            query=query_vector,
            limit=10,
            with_payload=True
        ).points

        cq_chunks = []

        for cq in cq_results:
            payload = cq.payload or {}
            matched = payload.get("matched_chunk_payload")

            if not matched:
                continue

            cq_chunks.append({
                "retrieval_type": "CQ",
                "score": cq.score,
                "source": matched.get("source"),
                "section": matched.get("section"),
                "content": matched.get("content"),
                "kb_chunk_id": matched.get("chunk_id") or f"CQ_{cq.id}"
            })

        # 2. KB SEARCH

        kb_results = self.client.query_points(
            collection_name="knowledge_base",
            query=query_vector,
            limit=10,
            with_payload=True
        ).points

        kb_chunks = []

        for kb in kb_results:
            payload = kb.payload or {}

            kb_chunks.append({
                "retrieval_type": "KB_DIRECT",
                "score": kb.score,
                "source": payload.get("source"),
                "section": payload.get("section"),
                "content": payload.get("content"),
                "kb_chunk_id": payload.get("chunk_id") or f"KB_{kb.id}"
            })

        # 3. OPTIONAL FILTER (light, not aggressive)
        kb_chunks = [
            ch for ch in kb_chunks
            if ch.get("score", 0) >= 0.78
        ]

        # 4. MERGE + DEDUP
        seen = set()
        final = []

        for ch in cq_chunks + kb_chunks:

            key = ch.get("kb_chunk_id")

            if key in seen:
                continue

            seen.add(key)
            final.append(ch)

        # 
        # 5. SORT (NORMALIZED SCORE)
        # 
        final = sorted(
            final,
            key=self._normalize_score,
            reverse=True
        )

        return final[:10]