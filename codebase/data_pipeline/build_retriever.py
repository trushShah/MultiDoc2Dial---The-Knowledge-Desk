import json
import numpy as np
import faiss

import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

from sentence_transformers import SentenceTransformer

# -----------------------------
# Load KB
# -----------------------------
with open(os.path.join(DATA_DIR, "kb.json"), "r", encoding="utf-8") as f:
    kb = json.load(f)

texts = [item["text"] for item in kb]

print(f"Loaded {len(texts)} chunks")

# -----------------------------
# Load embedding model
# -----------------------------
print("Loading embedding model...")
retriever_path = os.path.join(MODEL_DIR, "tuned_retriever")

if os.path.exists(retriever_path):
    print(f"[*] Found Tuned Retriever at {retriever_path}. Using locally tuned vectors.")
    model = SentenceTransformer(retriever_path)
else:
    print("[*] Tuned Retriever not found. Falling back to all-MiniLM-L6-v2 base.")
    model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Encode KB (Compute embeddings ONCE during preprocessing)
# -----------------------------
print("Encoding KB (this may take time)...")
embeddings = model.encode(texts, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

# -----------------------------
# Build FAISS index for fast retrieval
# -----------------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} vectors")

# -----------------------------
# Save Precomputed Vectors & Index
# -----------------------------
# 1. Save FAISS index
faiss.write_index(index, os.path.join(DATA_DIR, "kb.index"))

# 2. Persistently save Raw Numpy Embeddings to `.npy` (Requirement)
np.save("kb_embeddings.npy", embeddings)
print("[*] Embeddings successfully persistently cached to 'kb_embeddings.npy'")

# 3. Save chunk metadata
with open(os.path.join(DATA_DIR, "kb_meta.json"), "w", encoding="utf-8") as f:
    json.dump(kb, f, indent=2, ensure_ascii=False)

print(" Setup Complete: Retriever embeddings extracted and saved (kb.index + kb_meta.json + kb_embeddings.npy)")