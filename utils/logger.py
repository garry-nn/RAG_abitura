# utils/logger.py

import json
from datetime import datetime
from pathlib import Path


# logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# files
QUESTIONS_LOG = LOGS_DIR / "questions.jsonl"
CHUNKS_LOG = LOGS_DIR / "chunks.jsonl"


def log_question(question: str) -> None:
    """
    Save user question.
    """

    data = {
        "timestamp": datetime.now().isoformat(),
        "question": question
    }

    with open(QUESTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write("\n")


def log_chunks(question: str, chunks: list[dict]) -> None:
    """
    Save retrieved chunks.
    """

    data = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "chunks": chunks
    }

    with open(CHUNKS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write("\n")

