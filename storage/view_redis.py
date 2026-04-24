import redis
import json

r = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True
)

print("Redis connected:", r.ping())

# используем SCAN вместо KEYS
pattern = "*:chunk:*"
count = 0

for key in r.scan_iter(match=pattern):
    raw = r.get(key)
    if not raw:
        continue

    try:
        data = json.loads(raw)
    except Exception as e:
        print(f"Ошибка JSON в ключе {key}: {e}")
        continue

    print(f"KEY: {key}")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("-" * 40)

    count += 1

print(f"\nВсего найдено: {count}")