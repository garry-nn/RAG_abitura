import redis
import json
import os
import time

r = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True
)

# ожидание Redis
for i in range(10):
    try:
        if r.ping():
            print("Redis connected")
            break
    except Exception:
        print("Waiting for Redis...")
        time.sleep(1)
else:
    raise Exception("Redis not available")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_DIR = os.path.join(BASE_DIR, "..", "chunks")

FILES = [f for f in os.listdir(CHUNKS_DIR) if f.endswith(".json")]

total_loaded = 0

for filename in FILES:
    file_path = os.path.join(CHUNKS_DIR, filename)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        print(f"{filename}: ошибка чтения → {e}")
        continue

    if not chunks:
        print(f"{filename}: пустой файл")
        continue

    pipe = r.pipeline()
    loaded = 0

    for chunk in chunks:
        key = f"{filename}:chunk:{chunk['id']}"  # namespace по файлу

        if not r.exists(key):
            pipe.set(key, json.dumps(chunk))
            loaded += 1

    pipe.execute()

    print(f"{filename}: добавлено {loaded} новых чанков")
    total_loaded += loaded

print(f"Всего добавлено: {total_loaded}")