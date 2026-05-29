import numpy as np
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
collection_name = "query_index"

MAX_POINTS = 200


def print_vector(v, n=10):
    if v is None:
        return "None"
    arr = np.array(v)
    return arr[:n].round(5).tolist()


offset = None
total = 0

print("\n" + "=" * 80)
print(f"COLLECTION: {collection_name}")
print("=" * 80 + "\n")

while True:
    points, offset = client.scroll(
        collection_name=collection_name,
        limit=50,
        offset=offset,
        with_payload=True,
        with_vectors=True
    )

    if not points:
        break

    for p in points:
        total += 1

        print("-" * 80)
        print(f"ID: {p.id}")

        if p.vector:
            print("\nVECTOR (first 10 dims):")
            print(print_vector(p.vector))

        payload = p.payload

        print("\nPAYLOAD:")
        print(f"  cqs_id     : {payload.get('cqs_id')}")
        print(f"  source     : {payload.get('source')}")
        print(f"  query      : {payload.get('query')}")
        print(f"  variations : {payload.get('variations')}")
        print(f"  chunk_id   : {payload.get('chunk_id')}")
        print(f"  match_score: {payload.get('match_score')}")

        matched = payload.get("matched_chunk_payload")
        if matched:
            print("\n  MATCHED CHUNK:")
            print(f"    chunk_id : {matched.get('chunk_id')}")
            print(f"    source   : {matched.get('source')}")
            print(f"    section  : {matched.get('section')}")
            print(f"    content  : {matched.get('content')[:120]}...")

        print("\n")

        if total >= MAX_POINTS:
            break

    if total >= MAX_POINTS or offset is None:
        break

print("=" * 80)
print(f"DONE | TOTAL: {total}")
