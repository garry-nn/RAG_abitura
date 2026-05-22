import redis
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cdist

#  Redis 
r = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

print("Redis connected:", r.ping())

#  Модель трансформера 
MODEL_NAME = "intfloat/multilingual-e5-large"

print("Loading embedding model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)
model.eval()

print(f"Model loaded on {device}")


#  Эмбэдинг функция
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(),
        0.0
    )

    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed_query(text: str):
    """
    Для E5 query обязательно с prefix query:
    """

    formatted_text = f"query: {text}"

    batch = tokenizer(
        [formatted_text],
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)

    embeddings = average_pool(
        outputs.last_hidden_state,
        batch["attention_mask"]
    )

    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings[0].cpu().numpy()


# Загрузка чанков
pattern = "*:chunk:*"

chunks = []
chunk_embeddings = []
total_chunks = 0

for key in r.scan_iter(match=pattern):

    raw = r.get(key)

    if not raw:
        continue

    try:
        chunk = json.loads(raw)

        if "embedding" not in chunk:
            continue

        # сохраняем redis key внутрь chunk
        chunk["_redis_key"] = key

        chunks.append(chunk)
        chunk_embeddings.append(chunk["embedding"])
        total_chunks += 1

    except Exception as e:
        print(f"Ошибка чтения {key}: {e}")

print(f"Всего чанков загружено: {total_chunks}")

#  Тестовый запрос

test_question = "Какие документы нужны для поступления на програмную ижинерию?"

question_embedding = embed_query(test_question)
question_embedding = np.array([question_embedding])


# Поиск сходства 
distances = cdist(
    question_embedding,
    chunk_embeddings,
    metric="cosine"
)

similarities = 1 - distances.flatten()

top_k = 3

top_indices = np.argsort(distances.flatten())[:top_k]


#  Результат 
print("\n==============================")
print("Ваш запрос :")
print(test_question)

print("\nTOP RESULTS:\n")

for idx in top_indices:

    chunk = chunks[idx]

    print("--------------------------------------------------")
    print(f"SCORE: {round(similarities[idx], 4)}")
    print(f"KEY: {chunk.get('_redis_key')}")
    print(f"SECTION: {chunk.get('section')}")
    print(f"SOURCE: {chunk.get('source')}")
    print()
    print(chunk.get("content"))
    print("--------------------------------------------------")