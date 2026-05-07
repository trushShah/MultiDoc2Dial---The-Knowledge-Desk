from collections import Counter
import json
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
with open(os.path.join(DATA_DIR, "kb.json"), 'r', encoding = 'utf-8') as f:
    kb = json.load(f)

domains = [chunk["domain"] for chunk in kb]
counts = Counter(domains)

print("Chunks per domain:")
for domain, count in counts.items():
    print(f"{domain}: {count}")