import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
import json

# -----------------------------
# Load dataset
# -----------------------------
with open(os.path.join(DATA_DIR, "multidoc2dial_doc.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

kb = []
seen_sections = set()  # to avoid duplicate sections

# -----------------------------
# Build KB using section-level text
# -----------------------------
for domain in data["doc_data"]:
    docs = data["doc_data"][domain]

    for doc_id, doc in docs.items():
        spans = doc["spans"]

        for sp_id, sp in spans.items():
            sec_id = sp["id_sec"]

            # Unique section key
            unique_key = f"{doc_id}::{sec_id}"

            # Skip duplicates
            if unique_key in seen_sections:
                continue
            seen_sections.add(unique_key)

            # Use full section text (NOT tiny span)
            text = sp["text_sec"].strip()

            # Skip very small / useless chunks
            if not text or len(text) < 50:
                continue

            # Clean text (basic cleanup)
            text = " ".join(text.split())

            chunk = {
                "chunk_id": unique_key,
                "doc_id": doc_id,
                "text": text,
                "domain": domain
            }

            kb.append(chunk)

# -----------------------------
# Stats
# -----------------------------
print(f"Total chunks after fix: {len(kb)}")

# -----------------------------
# Save KB
# -----------------------------
with open(os.path.join(DATA_DIR, "kb.json"), "w", encoding="utf-8") as f:
    json.dump(kb, f, indent=2, ensure_ascii=False)

print("Fixed KB saved to kb.json")