import json

from qdrant_client import QdrantClient

from embeddings.embedder import Embedder


#  CONFIG 
client = QdrantClient(host="localhost", port=6333)

CQ_COLLECTION = "query_index"
KB_COLLECTION = "knowledge_base"

embedder = Embedder()


#  INPUT QUESTION 
text_question = "хочу на ФИвВТ какие специальнсти есть?"


#  SEARCH 
def search_cq(vector, top_k=3):
    return client.query_points(
        collection_name=CQ_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True
    ).points


def search_kb(vector, top_k=3):
    return client.query_points(
        collection_name=KB_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True
    ).points


#  PIPELINE 

# 1. embedding question
query_vector = embedder.embedding(text_question)

# 2. search CQ
cq_results = search_cq(query_vector, top_k=3)

# 3. search KB
kb_results = search_kb(query_vector, top_k=3)


#  FINAL CHUNKS 
top_chunks = []

used_chunks = set()


#  CQ RESULTS 
for cq in cq_results:
    payload = cq.payload

    matched = payload.get("matched_chunk_payload")

    if not matched:
        continue

    chunk_key = (
        matched.get("source"),
        matched.get("section"),
        matched.get("content")
    )

    # дедупликация
    if chunk_key in used_chunks:
        continue

    used_chunks.add(chunk_key)

    top_chunks.append({
        "retrieval_type": "CQ",
        "score": round(float(cq.score), 5),

        "source": matched.get("source"),
        "section": matched.get("section"),
        "content": matched.get("content")
    })


#  KB DIRECT RESULTS 
for kb in kb_results:
    payload = kb.payload

    chunk_key = (
        payload.get("source"),
        payload.get("section"),
        payload.get("content")
    )

    # дедупликация
    if chunk_key in used_chunks:
        continue

    used_chunks.add(chunk_key)

    top_chunks.append({
        "retrieval_type": "KB_DIRECT",
        "score": round(float(kb.score), 5),

        "source": payload.get("source"),
        "section": payload.get("section"),
        "content": payload.get("content")
    })


# ---------------- SORT ----------------
top_chunks = sorted(
    top_chunks,
    key=lambda x: x["score"],
    reverse=True
)


# ---------------- OUTPUT ----------------
print("\n" + "=" * 90)
print(f"QUESTION: {text_question}")
print("=" * 90)

print("\nTOP CHUNKS:\n")

print(
    json.dumps(
        top_chunks,
        ensure_ascii=False,
        indent=2
    )
)

print("\nDONE")