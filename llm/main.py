import json

from top_chunks.retriever import TopChunksRetriever
from llm.giga_generator import generate_rag_answer

question = "Хочу на ФИвВТ какие специальнсти есть?"

# RETRIEVAL
retriever = TopChunksRetriever()
chunks = retriever.get_top_chunks(question)

chunks_for_llm = sorted(
    chunks,
    key=lambda x: x.get("score", 0),
    reverse=True
)[:5]

print("\nTOP CHUNKS (FOR LLM):\n")
print(json.dumps(chunks_for_llm, ensure_ascii=False, indent=2))

# LLM
answer = generate_rag_answer(
    query=question,
    chunks=chunks_for_llm
)

print("\nANSWER\n")
print(answer)