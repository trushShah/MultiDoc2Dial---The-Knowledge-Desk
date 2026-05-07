import json
import torch
import random

import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def prepare_retriever_data(json_path, kb_meta_path, num_samples=500):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["dial_data"]
    with open(kb_meta_path, "r", encoding="utf-8") as f:
        kb = json.load(f)
        
    # We use a partial match since chunk_ids often extend doc_ids (e.g. #1_0::1)
    # We'll map the base doc_id to a list of its chunk texts
    doc_id_to_texts = {}
    for item in kb:
        did = item["doc_id"]
        if did not in doc_id_to_texts:
            doc_id_to_texts[did] = []
        doc_id_to_texts[did].append(item["text"])
    
    examples = []
    
    for domain, dialogues in data.items():
        for dial in dialogues:
            for turn in dial["turns"]:
                if turn["role"] == "user" and turn["references"]:
                    query = turn["utterance"]
                    ref_doc_id = turn["references"][0]["doc_id"]
                    
                    if ref_doc_id in doc_id_to_texts:
                        # Pair the query with the first chunk of the correct document
                        # In a pure setup, we'd pair with the exact span, but this is a lightweight demo
                        pos_text = doc_id_to_texts[ref_doc_id][0]
                        examples.append(InputExample(texts=[query, pos_text]))
                        
    random.seed(42)
    random.shuffle(examples)
    return examples[:num_samples]

if __name__ == "__main__":
    print("-" * 50)
    print("   FINE-TUNING RETRIEVER (Component A)")
    print("-" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Detected device: {device}")
    
    # Check if models dir exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("[*] Loading base SentenceTransformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    print("[*] Parsing MultiDoc2Dial dataset for Query->Context pairs...")
    train_examples = prepare_retriever_data(os.path.join(DATA_DIR, "multidoc2dial_dial_train.json"), os.path.join(DATA_DIR, "kb_meta.json"), num_samples=500)
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4) # small batch for laptop GPU
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    print(f"[*] Starting training on {len(train_examples)} examples (Multiple Negatives Ranking Loss)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=10,
        show_progress_bar=True
    )
    
    output_dir = os.path.join(MODEL_DIR, "tuned_retriever")
    model.save(output_dir)
    print(f" Success! Tuned Retriever weights saved to '{output_dir}'.")
