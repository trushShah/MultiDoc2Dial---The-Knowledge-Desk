import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load FAISS index
# -----------------------------
index = faiss.read_index(os.path.join(DATA_DIR, "kb.index"))

# -----------------------------
# Load metadata
# -----------------------------
with open(os.path.join(DATA_DIR, "kb_meta.json"), "r", encoding="utf-8") as f:
    kb = json.load(f)

# -----------------------------
# Load model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Query function
# -----------------------------
def search(query, top_k=5):
    print(f"\nQuery: {query}")

    # Encode query
    q_emb = model.encode([query]).astype("float32")

    # Search
    distances, indices = index.search(q_emb, top_k)

    # Show results
    print("\nTop Results:\n" + "-"*50)
    for i, idx in enumerate(indices[0]):
        chunk = kb[idx]
        print(f"[{i+1}] Score: {distances[0][i]:.4f}")
        print(f"Doc: {chunk['doc_id']}")
        print(f"Text: {chunk['text']}")
        print("-"*50)


# -----------------------------
# Test queries
# -----------------------------
if __name__ == "__main__":
    while True:
        query = input("\nEnter query (or 'exit'): ")
        if query.lower() == "exit":
            break
        search(query)