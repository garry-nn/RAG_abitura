from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

client.delete_collection("knowledge_base")
client.delete_collection("query_index")
print("Collection deleted")