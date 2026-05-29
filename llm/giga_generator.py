from typing import List, Dict

from llm.llm_client import call_llm
from llm.reranker import rerank_chunks


# ---------------- PROMPT ----------------

def build_rag_prompt(query: str, chunks: List[Dict]) -> str:
    context = ""

    for i, ch in enumerate(chunks, 1):
        context += f"""
[ЧАНК {i}]
SOURCE: {ch.get("source")}
SECTION: {ch.get("section")}
CONTENT: {ch.get("content")}
SCORE: {ch.get("score", 0)}
RETRIEVAL: {ch.get("retrieval_type")}
"""

    return f"""
Ты — полезный AI-ассистент, который отвечает на вопросы по предоставленным документам.

Отвечай естественно, понятно и дружелюбно.

Используй только информацию из чанков.
Если точного ответа в документах нет — напиши:
"Информация отсутствует в документах."

Старайся:
- не повторять текст чанков дословно без необходимости
- кратко и понятно формулировать ответ
- не придумывать факты вне документов

Формат ответа:

1. ОТВЕТ:
<ответ пользователю>

2. ИСТОЧНИКИ:
- source | section
- source | section

Вопрос:
{query}

Чанки:
{context}

Ответ:
"""


# ---------------- VERIFICATION ----------------

def build_verification_prompt(
    query: str,
    chunks: List[Dict],
    answer: str
) -> str:

    context = ""

    for ch in chunks:
        context += f"""
SOURCE: {ch.get("source")}
SECTION: {ch.get("section")}
CONTENT: {ch.get("content")}
"""

    return f"""
Проверь ответ на галлюцинации.

Оставь только информацию,
которая действительно есть в чанках.

Если какого-то факта нет —
удали его.

ВОПРОС:
{query}

ЧАНКИ:
{context}

ОТВЕТ:
{answer}

ИСПРАВЛЕННЫЙ ОТВЕТ:
"""


def verify_answer(
    query: str,
    chunks: List[Dict],
    answer: str
) -> str:

    prompt = build_verification_prompt(
        query,
        chunks,
        answer
    )

    return call_llm(
        prompt,
        temperature=0.1
    )


# ---------------- MAIN API ----------------

def generate_rag_answer(
    query: str,
    chunks: List[Dict]
) -> str:

    if not chunks:
        return "Информация отсутствует в документах."

    # sort by score
    chunks = sorted(
        chunks,
        key=lambda x: x.get("score", 0),
        reverse=True
    )

    # top chunks
    chunks = chunks[:10]

    # rerank
    chunks = rerank_chunks(
        query,
        chunks,
        top_k=4
    )

    if not chunks:
        return "Информация отсутствует в документах."

    # generation
    prompt = build_rag_prompt(
        query,
        chunks
    )

    raw_answer = call_llm(
        prompt,
        temperature=0.0
    )

    # verification
    verified_answer = verify_answer(
        query,
        chunks,
        raw_answer
    )

    return verified_answer