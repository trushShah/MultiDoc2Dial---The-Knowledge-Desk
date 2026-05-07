import json
import torch
import random
import numpy as np
from tqdm import tqdm

import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer

# 3 Classes: 0: ANSWER, 1: TICKET, 2: REJECT
class ToolPolicyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['text'] = self.texts[idx]
        return item

    def __len__(self):
        return len(self.labels)

def build_dataset(json_path, split_type="train"):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)["dial_data"]
        
    in_domain = []
    for domain, dialogues in data.items():
        for dial in dialogues:
            for turn in dial["turns"]:
                if turn["role"] == "user":
                    in_domain.append(turn["utterance"])
    
    random.seed(42)
    random.shuffle(in_domain)
    
    num_samples = len(in_domain)
    
    if split_type == "train":
        ticket_domain = [
            "What are the requirements to join the military?",
            "How do I renew my driver's license online?",
            "Can I get a student loan for an international university?",
            "I have a problem with my VA health benefits.",
            "Where is the nearest DMV office?",
            "How do I file a claim for disability?",
            "I need help with my FAFSA application.",
            "What is the phone number for the SSA?",
            "My social security card was stolen, what do I do?",
            "I need to report a change of address to the VA.",
            "Can you help me renew my passport?",
            "What is the process to renew a passport?"
        ]
        reject_domain = [
            "What is the weather like today?",
            "How do I bake a chocolate cake?",
            "Tell me a joke.",
            "Who won the world series in 2020?",
            "What is the capital of France?",
            "How do I fix a flat tire?",
            "Best restaurants near me?",
            "Can you write a poem about artificial intelligence?",
            "Which car is best for off-roading?",
            "What is the meaning of life?",
            "Translate this to Spanish.",
            "How tall is the Eiffel Tower?",
            "What are the rules of basketball?",
            "Can you summarize the movie Inception?",
            "How to invest in stocks?",
            "how do I apply for VI sim card?",
            "ice cream",
            "telecom issues"
        ]
    elif split_type == "val":
        ticket_domain = [
            "What are the vision requirements for a commercial driver's license?",
            "How long does it take for VA disability backpay to arrive?",
            "Can undocumented students apply for federal student aid?",
            "Do I need an appointment for a real ID at the DMV?",
            "How do I replace a lost Medicare card through the SSA?",
            "Can I transfer my GI Bill to my spouse?"
        ]
        reject_domain = [
            "How do I boil an egg?",
            "What is the square root of 144?",
            "Recommend a good sci-fi movie.",
            "Who is the current Prime Minister of the UK?",
            "What is the best workout routine for building muscle?",
            "How do I learn to play the guitar?",
            "What are the symptoms of the flu?",
            "When does the new iPhone come out?"
        ]
    else: # test
        ticket_domain = [
            "Are there any special DMV services for disabled veterans?",
            "What is the income limit for pell grants this year?",
            "How do I appeal a denied social security disability claim?",
            "Can I register my boat at the DMV?",
            "What happens to my student loans if I drop out?",
            "How do I get a copy of my DD214 separation papers?"
        ]
        reject_domain = [
            "Can dogs eat grapes?",
            "Write a python script to scrape a website.",
            "What is the origin of the universe?",
            "How much does a flight to Japan cost?",
            "Is tomato a fruit or a vegetable?",
            "Who wrote the Harry Potter books?",
            "What is the current price of Bitcoin?",
            "How do I fix a leaky faucet?"
        ]

    ticket_domain = (ticket_domain * (num_samples // len(ticket_domain) + 1))[:num_samples]
    reject_domain = (reject_domain * (num_samples // len(reject_domain) + 1))[:num_samples]
    random.shuffle(reject_domain)
    
    texts = in_domain + ticket_domain + reject_domain
    labels = [0] * len(in_domain) + [1] * len(ticket_domain) + [2] * len(reject_domain)
    
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    return texts, labels

def evaluate_model(model, dataloader, device, mu=0.5, desc="Evaluating"):
    model.eval()
    total_eval_loss = 0
    correct = 0
    tbp_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            p_correct = probs[torch.arange(len(target_labels)), target_labels]
            mask = torch.ones_like(probs, dtype=torch.bool)
            mask[torch.arange(len(target_labels)), target_labels] = False
            p_wrong_max = probs[mask].view(len(target_labels), -1).max(dim=1)[0]
            
            loss_boundary = torch.nn.functional.softplus(p_wrong_max - p_correct + mu).mean()
            total_eval_loss += loss_boundary.item()
            
            predictions = torch.argmax(probs, dim=1)
            correct += (predictions == target_labels).sum().item()
            
            # TBP metric computation
            tbp_mask = (predictions == target_labels) & ((p_correct - p_wrong_max) >= mu)
            tbp_correct += tbp_mask.sum().item()
            total_samples += len(target_labels)
            
    avg_eval_loss = total_eval_loss / len(dataloader)
    accuracy = correct / total_samples
    tbp = tbp_correct / total_samples
    
    return avg_eval_loss, accuracy, tbp

if __name__ == "__main__":
    print("-" * 50)
    print("   TRAINING TOOL POLICY CLASSIFIER (3-Class & Custom Losses)")
    print("-" * 50)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Check for DirectML (Radeon GPU), then CUDA, then fallback to CPU
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    print(f"[*] Detected device: {device}")
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)
    
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    # Keep embed_model on CPU due to DirectML inference tensor version bug
    DOMAIN_NAMES = ["Social Security Administration", "Veterans Affairs", "Federal Student Aid", "Department of Motor Vehicles"]
    kb_embeddings = embed_model.encode(DOMAIN_NAMES, convert_to_tensor=True)
    
    print("[*] Generating classification datasets... (This may take a minute for the full corpus)")
    train_texts, train_labels = build_dataset(os.path.join(DATA_DIR, "multidoc2dial_dial_train.json"), split_type="train")
    val_texts, val_labels = build_dataset(os.path.join(DATA_DIR, "multidoc2dial_dial_validation.json"), split_type="val")
    test_texts, test_labels = build_dataset(os.path.join(DATA_DIR, "multidoc2dial_dial_test.json"), split_type="test")
    
    train_dataset = ToolPolicyDataset(train_texts, train_labels, tokenizer)
    eval_dataset = ToolPolicyDataset(val_texts, val_labels, tokenizer)
    test_dataset = ToolPolicyDataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    epochs = 1  # Reduced to 1 epoch for faster iteration on the full corpus
    mu = 0.5
    tau = 0.8
    tau_0 = 0.8
    
    print("[*] Starting explicit training loop...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_labels = batch['labels'].to(device)
            raw_texts = batch['text']
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # 1. Triage Boundary Loss
            p_correct = probs[torch.arange(len(target_labels)), target_labels]
            
            mask = torch.ones_like(probs, dtype=torch.bool)
            mask[torch.arange(len(target_labels)), target_labels] = False
            p_wrong_max = probs[mask].view(len(target_labels), -1).max(dim=1)[0]
            
            loss_boundary = torch.nn.functional.softplus(p_wrong_max - p_correct + mu).mean()
            
            # 2. KB Proximity Signal (only for REJECT class -> y == 2)
            reject_mask = (target_labels == 2)
            loss_kb = torch.tensor(0.0, device=device)
            
            if reject_mask.any():
                reject_texts = [raw_texts[i] for i in range(len(raw_texts)) if reject_mask[i]]
                q_emb = embed_model.encode(reject_texts, convert_to_tensor=True)
                
                cos_sim = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(1), kb_embeddings.unsqueeze(0), dim=-1)
                max_sim = cos_sim.max(dim=1)[0]
                d_kb = 1.0 - max_sim
                
                kb_penalty_mask = d_kb < tau_0
                if kb_penalty_mask.any():
                    penalties = torch.nn.functional.softplus(tau - d_kb[kb_penalty_mask])
                    loss_kb = penalties.mean().to(device)
            
            loss = loss_boundary + 0.1 * loss_kb
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
        avg_train_loss = total_train_loss / len(train_loader)
        
        avg_eval_loss, accuracy, tbp = evaluate_model(model, eval_loader, device, mu, desc=f"Epoch {epoch+1} [Validation]")
        print(f"  [Epoch {epoch+1} Summary] Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_eval_loss:.4f} | Val Acc: {accuracy:.4f} | Val TBP: {tbp:.4f}\n")
    
    print("-" * 50)
    print("  FINAL EVALUATION ON TEST SET (multidoc2dial_dial_test.json)")
    avg_test_loss, test_accuracy, test_tbp = evaluate_model(model, test_loader, device, mu, desc="Testing")
    print(f"\n  Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.4f} | Test TBP: {test_tbp:.4f}")
    print("-" * 50)

    output_dir = os.path.join(MODEL_DIR, "tuned_tool_policy")
    model.to('cpu')  # Crucial fix for DirectML opaque tensor storage error
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f" Success! 3-Class Tool Policy Model saved to '{output_dir}'.")
