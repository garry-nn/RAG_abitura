import os
import json
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from embeddings.embedder import Embedder


# ---------------- Qdrant ----------------
client = QdrantClient(host="localhost", port=6333)
collection_name = "knowledge_base"


# ---------------- Embedder ----------------
embedder = Embedder()


# ---------------- Data ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_DIR = os.path.join(BASE_DIR, "..", "chunks")  # папка с chunks JSON

FILES = [f for f in os.listdir(CHUNKS_DIR) if f.endswith(".json")]


points = []
total_chunks = 0


# ---------------- Load chunks ----------------
for file in FILES:
    path = os.path.join(CHUNKS_DIR, file)

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)   # <-- LIST, не dict

    for chunk in chunks:
        total_chunks += 1

        chunk_id = chunk["id"]
        section = chunk.get("section", "")
        content = chunk.get("content", "")
        source = chunk.get("source", file)

        # ---------------- embedding ----------------
        text_for_embedding = f"{section}\n{content}"
        vector = embedder.embedding(text_for_embedding)

        # ---------------- payload ----------------
        payload = {
            "chunk_id": chunk_id,
            "source": source,
            "section": section,
            "content": content
        }

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload=payload
            )
        )


# ---------------- Upload ----------------
BATCH_SIZE = 64

for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i + BATCH_SIZE]

    client.upsert(
        collection_name=collection_name,
        points=batch
    )

    print(f"Uploaded {i + len(batch)} / {len(points)}")


print("\nDONE")
print("Total chunks:", total_chunks)