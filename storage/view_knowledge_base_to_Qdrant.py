from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
collection_name = "knowledge_base"

def shorten_vector(vector, n=10):
    return [round(x, 5) for x in vector[:n]]

def pretty_payload(payload: dict) -> str:
    lines = []
    for k, v in payload.items():
        if isinstance(v, str) and len(v) > 120:
            v = v[:120] + "..."
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)

# collection info
info = client.get_collection(collection_name)
print("\n=== COLLECTION INFO ===")
print(info)
print("\n")

# scroll 
offset = None
total = 0

print("=== POINTS PREVIEW ===\n")

while True:
    points, offset = client.scroll(
        collection_name=collection_name,
        limit=30,
        offset=offset,
        with_payload=True,
        with_vectors=True
    )

    if not points:
        break

    for p in points:
        total += 1

        print("=" * 80)
        print(f"ID: {p.id}")
        print(f"\nVECTOR (first 10 dims):")
        print(shorten_vector(p.vector, 10))

        print("\nPAYLOAD:")
        print(pretty_payload(p.payload))

    if offset is None:
        break

# summary 
print("\n" + "=" * 80)
print("TOTAL POINTS IN COLLECTION:", total)