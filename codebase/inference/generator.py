import json
import faiss
import numpy as np
import textwrap
import sys

# Windows PowerShell encoding fix for emojis
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel

import os

# Base directory for the codebase (one level up from inference)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

from inference.tools import CreateTicket, SearchKB

# =============================================================
#  Load retriever components
# =============================================================
print(" Loading retriever system...")
# Load Index
index = faiss.read_index(os.path.join(DATA_DIR, "kb.index"))

# Example of loading the persistent `.npy` embeddings if needed
if os.path.exists(os.path.join(DATA_DIR, "kb_embeddings.npy")):
    print("   ↳ Validated persistent numpy embeddings cache.")

with open(os.path.join(DATA_DIR, "kb_meta.json"), "r", encoding="utf-8") as f:
    kb = json.load(f)

# -------------------------------------------------------------
# 1. LOAD MODELS (Prioritizing locally Fine-Tuned variants)
# -------------------------------------------------------------

# A. Retriever
retriever_path = os.path.join(MODEL_DIR, "tuned_retriever")
if os.path.exists(retriever_path):
    print("   ↳ Loaded Tuned Retriever Module.")
    embed_model = SentenceTransformer(retriever_path)
else:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# B. Reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# C. Generator (with PEFT/LoRA)
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
try:
    import torch
    base_llm = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map='cpu',
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    print(" Base Generator LLM loaded successfully!")
except Exception as e:
    print(f" LLM load failed: {e}")
    raise

# Check for Alignment LoRA (DPO or custom Pairwise)
lora_path = os.path.join(MODEL_DIR, "tuned_alignment_lora")
if not os.path.exists(lora_path):
    lora_path = os.path.join(MODEL_DIR, "tuned_generator_lora")

if os.path.exists(lora_path):
    print(f" Active LLM Fine-Tuning weights loaded: {lora_path}")
    llm = PeftModel.from_pretrained(base_llm, lora_path)
else:
    llm = base_llm

# E. Tool-Policy Model
policy_path = os.path.join(MODEL_DIR, "tuned_tool_policy")
if os.path.exists(policy_path):
    print(" Tool Policy Classifier loaded.")
    tool_router = pipeline("text-classification", model=policy_path, tokenizer=policy_path)
else:
    tool_router = None

# =============================================================
#  Domain metadata (for user-friendly display)
# =============================================================
DOMAIN_LABELS = {
    "ssa": "Social Security Administration",
    "va": "Veterans Affairs",
    "studentaid": "Federal Student Aid",
    "dmv": "Department of Motor Vehicles",
}
DOMAIN_NAMES_LIST = list(DOMAIN_LABELS.values())
DOMAIN_EMBEDDINGS = embed_model.encode(DOMAIN_NAMES_LIST)
TICKET_CACHE_FILE = os.path.join(DATA_DIR, "ticket_cache.json")

# =============================================================
#  PHASE 1: ZERO-SHOT KEYWORD FILTER
# =============================================================
PHASE_1_KEYWORDS = [
    "dmv", "driver", "license", "car", "vehicle", "registration", "plate",
    "social security", "ssa", "medicare", "retirement", "ssn", "card",
    "veteran", "va", "military", "gi bill", "disability", "health", "claim",
    "student", "aid", "fafsa", "loan", "grant", "pell", "college", "school",
    "apply", "renew", "report", "appeal", "help", "account", "password"
]

def track_and_check_tickets(query, q_emb):
    if os.path.exists(TICKET_CACHE_FILE):
        try:
            with open(TICKET_CACHE_FILE, "r") as f:
                cache = json.load(f)
        except:
            cache = []
    else:
        cache = []
        
    cache.append({"query": query, "embedding": q_emb.tolist()})
    
    with open(TICKET_CACHE_FILE, "w") as f:
        json.dump(cache, f)
        
    count = 0
    for item in cache:
        sim = float(np.dot(q_emb, np.array(item["embedding"])))
        if sim > 0.85:
            count += 1
            
    if count >= 3:
        return True
    return False

def get_doc_title(doc_id):
    title = doc_id.split("#")[0].strip()
    parts = [p.strip() for p in title.split("|")]
    if len(parts) >= 2:
        return f"{parts[0]} — {parts[1]}"
    return title

# =============================================================
#  1. RETRIEVE  (FAISS nearest-neighbour search)
# =============================================================
def retrieve(query, top_k=15, threshold=1.0):
    """
    Retrieval using only query compute (document embeddings are precomputed in index).
    """
    q_emb = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, top_k)

    results = []
    seen_texts = set()

    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(kb): continue
        score = distances[0][i]
        if score > threshold: continue
        chunk = kb[idx]
        
        text_key = chunk["text"][:100].lower().strip()
        if text_key in seen_texts: continue
        seen_texts.add(text_key)

        results.append({
            "text": chunk["text"],
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            "domain": chunk.get("domain", "unknown"),
            "score": score,
        })
    return sorted(results, key=lambda x: x["score"])

# =============================================================
#  2. RERANK  (cross-encoder rescoring)
# =============================================================
def rerank(query, contexts, top_k=3):
    if not contexts: return []
    pairs = [(query, c["text"]) for c in contexts]
    scores = reranker.predict(pairs)
    for i, score in enumerate(scores):
        contexts[i]["rerank_score"] = float(score)
    contexts = sorted(contexts, key=lambda x: x["rerank_score"], reverse=True)
    contexts = [c for c in contexts if c["rerank_score"] > -1.0]
    return contexts[:top_k]

# =============================================================
#  3. STRUCTURED JSON TOOL CALLING & DECISION LOGIC
# =============================================================
def make_initial_decision(query):
    """
    Uses the trained policy to output a strictly structured JSON Tool Call request.
    0: SearchKB, 1: CreateTicket, 2: Reject
    """
    # ---------------------------------------------------------
    # PHASE 1: Zero-Shot Keyword Triage (Computation Reduction)
    # ---------------------------------------------------------
    query_lower = query.lower()
    if not any(kw in query_lower for kw in PHASE_1_KEYWORDS):
        return {
            "tool": "Reject",
            "probs": {"LABEL_0": 0.0, "LABEL_1": 0.0, "LABEL_2": 1.0},
            "parameters": {"query": query}
        }
        
    # ---------------------------------------------------------
    # PHASE 2: Deep Semantic Triage (DistilBERT Tool Policy)
    # ---------------------------------------------------------
    if tool_router is not None:
        predictions = tool_router(query[:512], top_k=None)
        if isinstance(predictions, list) and isinstance(predictions[0], list):
            predictions = predictions[0]
            
        probs = {p["label"]: p["score"] for p in predictions}
        label_str = max(probs, key=probs.get)
        
        if label_str == "LABEL_1":
            return {
                "tool": "CreateTicket",
                "probs": probs,
                "parameters": {
                    "summary": query,
                    "category": "out_of_domain",
                    "severity": "high"
                }
            }
        elif label_str == "LABEL_2":
            return {
                "tool": "Reject",
                "probs": probs,
                "parameters": {
                    "query": query
                }
            }
            
    # Default to SearchKB
    return {
        "tool": "SearchKB",
        "probs": {"LABEL_0": 1.0, "LABEL_1": 0.0, "LABEL_2": 0.0},
        "parameters": {
            "query": query
        }
    }

# =============================================================
#  4. SYSTEM GROUNDING ENFORCEMENT & PROMPT GENERATION
# =============================================================
def build_prompt(query, contexts):
    """
    Constructs prompt enforcing JSON Tool-style formatting constraints and
    exact Grounding citations.
    """
    source_lines = []
    char_budget = 1400
    current_chars = 0

    for i, c in enumerate(contexts):
        text = c["text"].strip()
        if len(text) > 400:
            idx = text.rfind('.', 0, 400)
            text = text[:idx+1] if idx != -1 else text[:400].rsplit(' ', 1)[0] + "."
            
        # Grounding Enforcement: inject index into prompt context
        line = f"[{i + 1}] {text}"
        
        if current_chars + len(line) > char_budget: break
        current_chars += len(line)
        source_lines.append(line)
    
    sources_block = "\n".join(source_lines)

    prompt = f"""Sources:
{sources_block}
Instruction: From each source provided, extract the two most important sentences that answer the question. Combine them into a single coherent paragraph with a logical flow. Ensure sentences depend on each other smoothly. You MUST cite your sources using [1], [2], etc.
Question: {query}
Answer:"""
    return prompt

def clean_answer(raw_answer, num_sources):
    answer = raw_answer.strip()
    for marker in ["Instruction:", "Question:", "Answer:"]:
        if marker in answer:
            answer = answer.split(marker)[-1].strip()

    if not answer:
        return "I do not have enough specific documents to answer definitively in this domain."

    has_any_cite = any(f"[{i + 1}]" in answer for i in range(num_sources))
    if not has_any_cite and num_sources > 0:
        refs = ", ".join(f"[{i + 1}]" for i in range(min(num_sources, 3)))
        answer += f"  ({refs})"

    if answer.startswith("Answer:"):
        answer = answer[7:].strip()
    return answer

# =============================================================
#  6. MAIN AGENT LOOP  (Query -> Decide -> Tool -> Feedback -> Generate)
# =============================================================
def generate_answer(query):
    search_query = query
    if len(query.split()) <= 2: search_query = f"Explain in detail about {query}"
    elif len(query.split()) <= 4: search_query = f"Explain {query}"

    # 1. Decide (Emit JSON Tool Call)
    tool_command = make_initial_decision(query)
    print(f"  [Agent Decision JSON]: {json.dumps(tool_command)}")

    # Track all queries
    q_emb = embed_model.encode([query])[0]
    advise_maker = track_and_check_tickets(query, q_emb)
    
    if advise_maker:
        print("\n  I am getting queries related to this domain ofern which is out of our domain so kindly provide me knowledge on this domain.\n")


    if tool_command["tool"] == "Reject":
        return {
            "action": "irrelevant",
            "answer": "This query is completely outside the scope of my knowledge base. My available domains are: " + ", ".join(DOMAIN_NAMES_LIST) + ".",
            "citations": [],
            "ticket_id": None,
            "advise_maker": advise_maker,
            "probs": tool_command.get("probs", {})
        }

    # 2. Execute Action Based on JSON tool output
    elif tool_command["tool"] == "CreateTicket":
        tool_output = CreateTicket(**tool_command["parameters"])
        return {
            "action": "out_of_domain",
            "answer": None,
            "citations": [],
            "ticket_id": tool_output["ticket_id"],
            "advise_maker": advise_maker,
            "probs": tool_command.get("probs", {}),
            "available_domains": [{"code": c, "label": l} for c, l in DOMAIN_LABELS.items()]
        }
        
    elif tool_command["tool"] == "SearchKB":
        # 3. Tool Execution -> Retrieve
        tool_output = SearchKB(tool_command["parameters"]["query"], retrieve)
        retrieved = tool_output["results"]
        
        # 4. Rerank Phase
        contexts = rerank(search_query, retrieved, top_k=3)
        
        # 5. Review Output & Validate
        top_score = contexts[0].get("rerank_score", -999) if contexts else -999
        action = "answer"
        
        if top_score < -2.0:
            # Hallucination filter tripped, escalate via tool
            escalate_cmd = {"tool": "CreateTicket", "parameters": {"summary": query, "category": "out_of_domain", "severity": "high"}}
            print(f"  [Fallback Tool Trigger]: {json.dumps(escalate_cmd)}")
            esc_output = CreateTicket(**escalate_cmd["parameters"])
            return {
                "action": "out_of_domain",
                "answer": None,
                "citations": [],
                "ticket_id": esc_output["ticket_id"],
                "advise_maker": advise_maker,
                "probs": tool_command.get("probs", {}),
                "available_domains": [{"code": c, "label": l} for c, l in DOMAIN_LABELS.items()]
            }
        elif top_score < 0.0:
            action = "low_confidence"

        # 6. Final Tool Policy - Generator with strictly grounded parameters
        prompt_contexts = contexts[:3]
        prompt = build_prompt(query, prompt_contexts)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = llm.generate(
            **inputs,
            max_new_tokens=300,
            min_new_tokens=40,
            repetition_penalty=1.15,
            num_beams=4,
            length_penalty=1.5,
            early_stopping=True
        )

        raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = clean_answer(raw_answer, len(prompt_contexts))

        # Grounding Enforcement Mapping
        citations = []
        for i, c in enumerate(prompt_contexts):
            citations.append({
                "index": i + 1,
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
                "title": get_doc_title(c["doc_id"]),
                "domain": DOMAIN_LABELS.get(c.get("domain", "unknown"), c.get("domain", "unknown")),
                "domain_code": c.get("domain", "unknown"),
                "score": c.get("rerank_score", 0.0),
                "snippet": c["text"][:137] + "..." if len(c["text"]) > 140 else c["text"]
            })
            
        return {
            "action": action,
            "answer": answer,
            "citations": citations,
            "ticket_id": None,
            "advise_maker": advise_maker,
            "probs": tool_command.get("probs", {})
        }

# =============================================================
#  DISPLAY COMMANDS
# =============================================================
def print_divider(char="═", width=70): print(char * width)

def print_answer(answer):
    wrapped = textwrap.fill(answer, width=68, initial_indent="  ", subsequent_indent="  ")
    print(wrapped)

if __name__ == "__main__":
    print()
    print_divider()
    print("    MultiDoc2Dial — Agentic Knowledge Support System")
    print("  Domains: SSA  |  VA  |  Student Aid  |  DMV")
    print_divider()

    while True:
        query = input("\n    Your question: ").strip()
        if query.lower() in ("exit", "quit", "q"): break
        if not query: continue

        result = generate_answer(query)
        action = result["action"]
        
        if action == "irrelevant":
            print("\n    Agent Response:")
            print_answer(result["answer"])
            continue

        if action == "out_of_domain":
            print()
            print_divider()
            print("     ESCALATION TRIGGERED")
            print_divider()
            print(f"  I have escalated this issue to a human agent. (Ticket ID: {result['ticket_id']})\n")
            continue

        if action == "low_confidence":
            print("\n    Low-Confidence Response — Verifying with official sources advised.")
        
        print("\n    Agent Response:")
        print_answer(result["answer"])

        print("\n    Grounding Citations:")
        print_divider("-")
        for c in result["citations"]:
            print(f"  [{c['index']}] {c['title']} | Domain: {c['domain']}")
            print(f"       doc_id: {c['doc_id']} | chunk_id: {c['chunk_id']}")
            print(f"       Score: {c['score']:+.2f} | \"{c['snippet']}\"")
        print_divider("-")