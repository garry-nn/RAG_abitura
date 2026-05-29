from typing import List, Dict
from llm.llm_client import call_llm   # 👈 ВАЖНО (не giga_generator)

def build_rerank_prompt(query: str, chunks: List[Dict]) -> str:
    context = ""

    for i, ch in enumerate(chunks):
        context += f"""
[{i}]
SOURCE: {ch.get("source")}
SECTION: {ch.get("section")}
CONTENT: {ch.get("content")[:300]}
"""

    return f"""
Ты rerank система.

Выбери ТОП-4 наиболее релевантных чанка.

Вопрос:
{query}

Чанки:
{context}

Верни индексы через запятую.
"""


def rerank_chunks(query: str, chunks: List[Dict], top_k: int = 4) -> List[Dict]:
    if len(chunks) <= top_k:
        return chunks

    prompt = build_rerank_prompt(query, chunks)
    response = call_llm(prompt, temperature=0.0)

    try:
        idxs = [int(x.strip()) for x in response.split(",") if x.strip().isdigit()]

        result = []
        for i in idxs:
            if 0 <= i < len(chunks):
                result.append(chunks[i])

        return result[:top_k] if result else chunks[:top_k]

    except:
        return chunks[:top_k]