import json
import torch

import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

def build_generator_dataset(json_path, kb_meta_path, num_samples=200):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)["dial_data"]
    with open(kb_meta_path, 'r', encoding='utf-8') as f:
        kb = json.load(f)
        
    doc_id_to_texts = {}
    for item in kb:
        did = item["doc_id"]
        if did not in doc_id_to_texts:
            doc_id_to_texts[did] = []
        doc_id_to_texts[did].append(item["text"])
        
    prompts = []
    targets = []
    
    for domain, dialogues in data.items():
        for dial in dialogues:
            for i, turn in enumerate(dial["turns"]):
                # We want agent responses correctly grounded to docs
                if turn["role"] == "agent" and turn.get("references"):
                    if i > 0 and dial["turns"][i-1]["role"] == "user":
                        query = dial["turns"][i-1]["utterance"]
                        answer = turn["utterance"]
                        ref_doc_id = turn["references"][0]["doc_id"]
                        
                        if ref_doc_id in doc_id_to_texts:
                            # To train concise preference rubric, we forcefully shorten long targets limit to 3 sentences
                            sents = answer.split(". ")
                            if len(sents) > 3:
                                answer = ". ".join(sents[:3]) + "."
                                
                            context = doc_id_to_texts[ref_doc_id][0][:600]
                            
                            prompt = f"Using the following knowledge, answer concisely:\n\n{context}\n\nQuestion: {query}"
                            # Add synthetic citation token to train the citation-first rubric
                            target = f"{answer} [1]"
                            
                            prompts.append(prompt)
                            targets.append(target)
                            
    random.seed(42)
    combined = list(zip(prompts, targets))
    random.shuffle(combined)
    combined = combined[:num_samples]
    prompts, targets = zip(*combined)
    
    return Dataset.from_dict({"prompt": list(prompts), "target": list(targets)})

if __name__ == "__main__":
    print("-" * 50)
    print("   TRAINING GENERATOR VIA LoRA (Component C)")
    print("-" * 50)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Detected device: {device}")
    
    model_name = "google/flan-t5-base"
    print(f"[*] Loading base model {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    print("[*] Applying PEFT/LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    print("[*] Generating prompt-target dataset...")
    dataset = build_generator_dataset(os.path.join(DATA_DIR, "multidoc2dial_dial_train.json"), os.path.join(DATA_DIR, "kb_meta.json"), num_samples=150)
    
    def tokenize_function(examples):
        model_inputs = tokenizer(examples["prompt"], max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(examples["target"], max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir="./models/checkpoints_lora",
        per_device_train_batch_size=2,
        learning_rate=3e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="no"
    )
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset
    )
    
    print("[*] Starting LoRA fine-tuning...")
    trainer.train()
    
    output_dir = os.path.join(MODEL_DIR, "tuned_generator_lora")
    peft_model.save_pretrained(output_dir)
    print(f" Success! LoRA weights saved to '{output_dir}'.")
