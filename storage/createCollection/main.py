from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)

collection_name = "knowledge_base"

if client.collection_exists(collection_name):
    print("Collection already exists")
else:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE
        )
    )
    print("Collection created")

print(client.get_collections())