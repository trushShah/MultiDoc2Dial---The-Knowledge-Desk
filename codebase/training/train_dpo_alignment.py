import json
import torch
import random
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW

class PreferenceDataset(Dataset):
    def __init__(self, json_path, kb_meta_path, tokenizer, num_samples=200):
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        with open(os.path.join(DATA_DIR, json_path), 'r', encoding='utf-8') as f:
            data = json.load(f)["dial_data"]
        with open(os.path.join(DATA_DIR, kb_meta_path), 'r', encoding='utf-8') as f:
            kb = json.load(f)
            
        doc_id_to_texts = {}
        for item in kb:
            did = item["doc_id"]
            if did not in doc_id_to_texts:
                doc_id_to_texts[did] = []
            doc_id_to_texts[did].append(item["text"])
            
        pairs = []
        for domain, dialogues in data.items():
            for dial in dialogues:
                for i, turn in enumerate(dial["turns"]):
                    if turn["role"] == "agent" and turn.get("references"):
                        if i > 0 and dial["turns"][i-1]["role"] == "user":
                            query = dial["turns"][i-1]["utterance"]
                            answer = turn["utterance"]
                            ref_doc_id = turn["references"][0]["doc_id"]
                            
                            if ref_doc_id in doc_id_to_texts:
                                context = doc_id_to_texts[ref_doc_id][0][:500] # Use top 500 chars
                                prompt = f"Sources:\n[1] {context}\nCompare and give the exact grounded answer.\nQuestion: {query}\nAnswer:"
                                
                                # Chosen: Concise, grounded answer with citation
                                chosen_sents = answer.split(". ")
                                chosen = ". ".join(chosen_sents[:3]) + "." if len(chosen_sents) > 3 else answer
                                chosen = f"{chosen} [1]"
                                
                                # Rejected: Hallucinated / repetitive / missing citation
                                rejected = f"{answer} I also think that perhaps this requires more analysis of other irrelevant domains without checking the documents."
                                
                                pairs.append({
                                    "prompt": prompt,
                                    "chosen": chosen,
                                    "rejected": rejected
                                })
        
        random.seed(42)
        random.shuffle(pairs)
        pairs = pairs[:num_samples]
        
        self.encodings = []
        for p in pairs:
            # Tokenize prompt
            p_enc = tokenizer(p["prompt"], max_length=512, truncation=True, padding="max_length")
            # Tokenize chosen and rejected
            c_enc = tokenizer(p["chosen"], max_length=128, truncation=True, padding="max_length")
            r_enc = tokenizer(p["rejected"], max_length=128, truncation=True, padding="max_length")
            
            self.encodings.append({
                "input_ids": p_enc["input_ids"],
                "attention_mask": p_enc["attention_mask"],
                "chosen_labels": c_enc["input_ids"],
                "rejected_labels": r_enc["input_ids"]
            })

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.encodings[idx].items()}

    def __len__(self):
        return len(self.encodings)

if __name__ == "__main__":
    print("-" * 50)
    print("   TRAINING ALIGNMENT VIA PAIRWISE RANKING LOSS")
    print("-" * 50)
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Detected device: {device}")
    
    # Load base model + LoRA
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    print("[*] Applying PEFT/LoRA adapters for Alignment...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8, lora_alpha=32, lora_dropout=0.1
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    
    print("[*] Generating Preference Dataset...")
    train_dataset = PreferenceDataset("multidoc2dial_dial_train.json", "kb_meta.json", tokenizer, num_samples=150)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    # Pairwise Ranking Loss Margin
    margin = 1.0
    epochs = 2
    
    print("[*] Starting Custom Pairwise Alignment Loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)
            
            # 1. Forward chosen
            # Replace pad token id with -100 to ignore in loss
            chosen_labels[chosen_labels == tokenizer.pad_token_id] = -100
            out_chosen = model(input_ids, attention_mask=attention_mask, labels=chosen_labels)
            loss_chosen = out_chosen.loss # scalar mean loss
            
            # 2. Forward rejected
            rejected_labels[rejected_labels == tokenizer.pad_token_id] = -100
            out_rejected = model(input_ids, attention_mask=attention_mask, labels=rejected_labels)
            loss_rejected = out_rejected.loss # scalar mean loss
            
            # 3. Preference Loss
            # We want loss_chosen < loss_rejected. So difference should be large.
            # max(0, loss_chosen - loss_rejected + margin)
            pairwise_loss = F.relu(loss_chosen - loss_rejected + margin)
            
            # Combine standard negative log likelihood with pairwise margin
            combined_loss = loss_chosen + 0.5 * pairwise_loss
            
            combined_loss.backward()
            optimizer.step()
            
            total_loss += combined_loss.item()
            
            if step % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs} | Step {step} | C_Loss: {loss_chosen.item():.4f} | R_Loss: {loss_rejected.item():.4f} | Total: {combined_loss.item():.4f}")
                
        print(f"  [Epoch {epoch+1} Summary] Avg Loss: {total_loss/len(train_loader):.4f}")
    
    output_dir = os.path.join(MODEL_DIR, "tuned_alignment_lora")
    model.save_pretrained(output_dir)
    print(f" Success! Aligned LoRA weights saved to '{output_dir}'.")
