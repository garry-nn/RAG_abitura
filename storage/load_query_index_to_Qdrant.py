import os
import json
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from embeddings.embedder import Embedder


# ---------------- Qdrant ----------------
client = QdrantClient(host="localhost", port=6333)

QUERY_COLLECTION = "query_index"
KNOWLEDGE_COLLECTION = "knowledge_base"

# ---------------- Embedder ----------------
embedder = Embedder()


# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CQS_DIR = os.path.join(BASE_DIR, "..", "cqs")

FILES = [f for f in os.listdir(CQS_DIR) if f.endswith(".json")]


# ---------------- Search ----------------
def search_best_chunk(vector, top_k=3):
    results = client.query_points(
        collection_name=KNOWLEDGE_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )
    return results.points


# ---------------- Build Query Index ----------------
points = []
total = 0

for file in FILES:
    path = os.path.join(CQS_DIR, file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        total += 1

        cqs_id = item["cqs_id"]
        source = item["source"]
        query = item.get("query", "")
        variations = item.get("variations", [])

        # ---------------- embedding ----------------
        text_for_embedding = query + " " + " ".join(variations)
        vector = embedder.embedding(text_for_embedding)

        # ---------------- search knowledge base ----------------
        results = search_best_chunk(vector, top_k=3)

        candidates = []

        if results:
            best = results[0]
            score = float(best.score)

            chunk_id = best.id  # ✅ UUID из Qdrant

            matched_chunk_payload = {
                "chunk_id": best.payload.get("chunk_id"),
                "source": best.payload.get("source"),
                "section": best.payload.get("section"),
                "content": best.payload.get("content"),
            }

            # топ-кандидаты (для анализа качества)
            for r in results:
                candidates.append({
                    "chunk_id": r.id,
                    "score": float(r.score)
                })

        else:
            score = 0.0
            chunk_id = None
            matched_chunk_payload = None

        # ---------------- payload ----------------
        payload = {
            "cqs_id": cqs_id,
            "source": source,
            "query": query,
            "variations": variations,

            "chunk_id": chunk_id,  # ✅ UUID из knowledge_base

            "matched_chunk_payload": matched_chunk_payload,

            "match_score": score,

            "candidates": candidates  # 🔥 диагностика качества retrieval
        }

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload=payload
            )
        )


# ---------------- Upload ----------------
BATCH = 64

for i in range(0, len(points), BATCH):
    client.upsert(
        collection_name=QUERY_COLLECTION,
        points=points[i:i + BATCH]
    )
    print(f"Uploaded {i + len(points[i:i+BATCH])}/{len(points)}")

print("\nDONE")
print("Total CQs:", total)