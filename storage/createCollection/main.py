from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


# ---------------- Qdrant ----------------
client = QdrantClient(
    host="localhost",
    port=6333
)


# ---------------- Collections Config ----------------
collections = {
    "knowledge_base": {
        "size": 1024,
        "distance": Distance.COSINE
    },

    "query_index": {
        "size": 1024,
        "distance": Distance.COSINE
    }
}


# ---------------- Create Collections ----------------
for collection_name, config in collections.items():

    if client.collection_exists(collection_name):
        print(f"[EXISTS] {collection_name}")

    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=config["size"],
                distance=config["distance"]
            )
        )

        print(f"[CREATED] {collection_name}")


# ---------------- Result ----------------
print("\n ALL COLLECTIONS")
print(client.get_collections())