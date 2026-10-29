# MultiDoc2Dial: The Knowledge Desk — Master Technical Deep-Dive

> **Author's Private Reference & Conceptual Compass**  
> *Keep this document safe. Even if you don't look at this codebase for years, reading this single file will restore 100% of your intuition, architectural understanding, and mathematical grasp of this project.*

---

## Table of Contents
1. [The "30-Second Mental Model" (What This Really Is)](#1-the-30-second-mental-model)
2. [The Core Problem: Why Monolithic RAG Fails in Production](#2-the-core-problem-why-monolithic-rag-fails)
3. [The Dataset: MultiDoc2Dial Deconstructed](#3-the-dataset-multidoc2dial-deconstructed)
4. [Complete Step-by-Step Inference Walkthrough](#4-complete-step-by-step-inference-walkthrough)
5. [The Machine Learning Models & Their Roles](#5-the-machine-learning-models--their-roles)
6. [The Mathematical Loss Formulations & Why Standard Losses Failed](#6-the-mathematical-loss-formulations)
7. [File-by-File Codebase Blueprint](#7-file-by-file-codebase-blueprint)
8. [Ingenious Engineering Details & Subtle Edge Cases Handled](#8-ingenious-engineering-details)
9. [Empirical Results, Ablations & Benchmarks](#9-empirical-results-ablations--benchmarks)
10. [Master Viva & Technical Interview Cheat Sheet](#10-master-viva--technical-interview-cheat-sheet)
11. [How to Rebuild, Retrain, and Run Everything From Scratch](#11-how-to-rebuild-retrain-and-run-everything)

---

## 1. The "30-Second Mental Model"

Imagine an emergency hospital desk:
1. **The Gatekeeper (Phase 1 Keyword Triage)**: Someone asks *"Where can I buy lawnmower parts?"* The gatekeeper immediately tells them they are in the wrong building in 0.15 milliseconds without wasting doctors' time.
2. **The Triage Nurse (Phase 2 DistilBERT Tool Policy)**: A visitor asks a question. The nurse decides: Can this be answered from our official medical books (`SearchKB`), does it require opening a formal patient case ticket for a human supervisor (`CreateTicket`), or is it irrelevant (`Reject`)?
3. **The Medical Archivist (Phase 3 Bi-Encoder + Cross-Encoder Reranker)**: If routing to `SearchKB`, the archivist pulls 15 relevant pages from the library using dense semantic search (FAISS). Then a senior specialist (Cross-Encoder) reads the query side-by-side with each page. If even the best page has a terrible relevance score ($< -2.0$), the specialist stops the process and escalates to a human ticket.
4. **The Citing Physician (Phase 4 Aligned FLAN-T5 Generator)**: Given only the verified top-3 snippets, the model writes a 2-3 sentence answer. It is trained via preference alignment so it **must** cite source IDs like `[1]` and `[2]`. It never babbles or guesses.

**Why this matters**: We don't just dump text into an LLM and pray it doesn't hallucinate. We built a **deterministic, multi-tier safety pipeline** around the model.

---

## 2. The Core Problem: Why Monolithic RAG Fails

In standard industry RAG:
```
User Query ──▶ Vector Search (top-k) ──▶ Dump into LLM Prompt ──▶ Generate Answer
```

### The 3 Fatal Flaws of Standard RAG:
1. **The "No-Refusal" Trap**: If a user asks *"How do I bake a chocolate cake?"* or *"What is the best off-roading vehicle?"*, vector search will still return the closest 3 chunks (maybe DMV regulations or SSA rules about food stamps). The LLM sees the prompt: `"Context: ... Question: How do I bake cake? Answer:"` and either fabricates a recipe or hallucinates weird connections to the documents.
2. **Softmax Entropy at Decision Boundaries**: If you train a normal classifier with standard Cross-Entropy loss to classify in-domain vs. out-of-domain queries, it suffers near the boundary. A query that uses official words in an out-of-domain context (e.g., *"Which motor vehicle is best for racing?"*) causes standard classifiers to hesitate (probabilities like 51% vs 49%), leading to false positives.
3. **Citation Amnesia**: Base LLMs like FLAN-T5, LLaMA, or Mistral prefer conversational filler (*"Sure, I can help with that..."*) rather than strict, legalistic inline citations (`[1]`, `[2]`), making customer support answers impossible to verify.

---

## 3. The Dataset: MultiDoc2Dial Deconstructed

Published by IBM Research & Penn State at **EMNLP 2021**, MultiDoc2Dial is the gold standard for goal-oriented customer support dialogues grounded in multiple documents.

### The 4 High-Stakes Public Domains:
| Domain Code | Agency | Nature of Knowledge |
| :--- | :--- | :--- |
| `ssa` | **Social Security Administration** | Retirement credits, disability insurance, Medicare Part A/B, SSI eligibility |
| `va` | **Veterans Affairs** | GI Bill, disability compensation, health benefits, education Top-Up |
| `studentaid` | **Federal Student Aid** | FAFSA, Direct Subsidized/Unsubsidized Loans, Pell Grants, repayment plans |
| `dmv` | **Department of Motor Vehicles** | Driver's license renewal, REAL ID, vehicle registration, title transfer |

### Dataset Structure on Disk:
- `multidoc2dial_doc.json`: The raw knowledge base documents.
  - Contains documents organized by `domain` and `doc_id`.
  - Inside each doc: `spans` (individual sentence/phrase tags like `id_sp`, `text_sp`) and sections (`id_sec`, `text_sec`).
  - **Crucial Engineering Decision in `build_kb.py`**: Raw spans in MultiDoc2Dial are too tiny (sometimes just 3 words like *"apply online"*). If you index spans directly, vector search has no context! In `build_kb.py`, we grouped by unique section `f"{doc_id}::{sec_id}"`, extracting complete paragraphs ($\ge 50$ characters). This yielded **9,203 rich, self-contained knowledge chunks**:
    - `ssa`: 1,697 chunks
    - `va`: 2,845 chunks
    - `dmv`: 2,826 chunks
    - `studentaid`: 1,835 chunks
- `multidoc2dial_dial_train.json` & `multidoc2dial_dial_validation.json` & `multidoc2dial_dial_test.json`:
  - Contains multi-turn customer-agent dialogues.
  - Each turn includes: `role` (`"user"` or `"agent"`), `utterance` (what was said), and `references` (`id_sp`, `doc_id`).
  - These references tell us the exact ground-truth document that justifies the agent's utterance.

---

## 4. Complete Step-by-Step Inference Walkthrough

The core logic lives in [`codebase/inference/generator.py`](file:///T:/DAIICT%20Resume/Projects/RAG/MultiDoc2Dial---The-Knowledge-Desk/codebase/inference/generator.py) inside `generate_answer(query)`. Here is what happens for every possible user input:

```mermaid
flowchart TD
    Start([User Inputs Query]) --> P1{Phase 1: Keyword Check}
    
    P1 -- "No Keywords (e.g. 'Capital of France?')" --> Reject[Tool: Reject<br/>Returns Domain Scope Message]
    P1 -- "Keyword Found (e.g. 'driver', 'medicare')" --> P2{Phase 2: DistilBERT Policy}
    
    P2 -- "LABEL_1 (OOD)" --> Ticket1[Tool: CreateTicket<br/>Escalates to Human Agent]
    P2 -- "LABEL_2 (Reject)" --> Reject
    P2 -- "LABEL_0 (In-Domain)" --> P3[Phase 3: FAISS Dense Retrieve<br/>Fetches Top-15 from 9,203 Chunks]
    
    P3 --> Rerank[MS-MARCO Cross-Encoder Reranks Top-15]
    Rerank --> ScoreCheck{Top Rerank Score < -2.0?}
    
    ScoreCheck -- "Yes: Hallucination Bait<br/>(e.g. 'off-roading motor vehicle')" --> Ticket2[Fallback Escalation<br/>Tool: CreateTicket]
    ScoreCheck -- "No: High Synergy (e.g. score +8.78)" --> Prompt[Phase 4: Prompt Construction<br/>Builds Sources [1], [2], [3]]
    
    Prompt --> LLM[FLAN-T5-Base + LoRA Generator]
    LLM --> Clean[Clean & Parse Citations]
    Clean --> Output([Structured JSON Output to UI / API])
```

### Trace 1: Pure Out-Of-Domain Query
> Query: *"What is the capital of France?"*
1. **Phase 1**: Checks `PHASE_1_KEYWORDS` (`dmv`, `license`, `social security`, `medicare`, `veteran`, `loan`, etc.). None match.
2. **Decision**: Emits `{"tool": "Reject", "probs": {"LABEL_2": 1.0}}`.
3. **Execution**: No FAISS search, no cross-encoder, no LLM call. Finished in **0.15 ms**.
4. **Output**: Polite message indicating the query is outside available domains (SSA, VA, Student Aid, DMV).

### Trace 2: Legitimate In-Domain Fact Query
> Query: *"How do I apply for Medicare?"*
1. **Phase 1**: Keyword `"medicare"` matches $\to$ Passes to Phase 2.
2. **Phase 2**: DistilBERT classifies query as `LABEL_0` (Confidence: 1.0) $\to$ Emits `{"tool": "SearchKB"}`.
3. **Phase 3 Retrieval**: Encodes query into 384-d vector with `all-MiniLM-L6-v2`. Searches `kb.index` via FAISS. Finds top-15 candidate chunks.
4. **Phase 3 Reranking**: Cross-Encoder `ms-marco-MiniLM-L-6-v2` compares query against candidate texts.
   - Chunk 1: *Medicare Benefits — SSA* (Score: **+8.78**)
   - Chunk 2: *Apply for Retirement Benefits — SSA* (Score: **+6.80**)
   - Chunk 3: *CHAMPVA Benefits — VA* (Score: **+6.16**)
5. **Gating Check**: Top score ($+8.78 \ge -2.0$) passes.
6. **Phase 4 Prompt Construction**: Assembles prompt with strict `[1]`, `[2]`, `[3]` markers:
   ```text
   Sources:
   [1] You can apply online for Medicare even if you are not ready to retire...
   [2] Social Security will process your application...
   [3] CHAMPVA pays for covered services...
   Instruction: From each source provided, extract the two most important sentences that answer the question... You MUST cite your sources using [1], [2], etc.
   Question: How do I apply for Medicare?
   Answer:
   ```
7. **Generation**: FLAN-T5-Base with LoRA weights performs beam decoding (`num_beams=4`).
8. **Output**: Concise answer with inline citations `([1], [2], [3])` and full chunk metadata for the UI sources panel.

### Trace 3: Adversarial Keyword-Spoofed Query
> Query: *"which motor vehicle is best for off-roading?"*
1. **Phase 1**: Contains the token `"motor vehicle"` $\to$ Passes Phase 1!
2. **Phase 2**: If running with base fallback, tool router emits `SearchKB`.
3. **Phase 3 Retrieval**: FAISS retrieves DMV motor vehicle registration chunks (e.g. *"Registering custom or modified vehicles"*).
4. **Phase 3 Reranker**: Cross-Encoder computes semantic synergy between off-roading query and DMV registration rules.
   - Top score is **-3.41**! (Extremely poor semantic fit).
5. **Fallback Escalation**: Because `top_score < -2.0`, the system halts generation and dynamically switches tool:
   ```json
   {"tool": "CreateTicket", "parameters": {"summary": "which motor vehicle is best for off-roading?", "category": "out_of_domain", "severity": "high"}}
   ```
6. **Output**: Generates ticket `TICKET-8161`. Zero hallucination.

### Trace 4: Frequent Query Clustering ("Advise Maker")
In [`generator.py`](file:///T:/DAIICT%20Resume/Projects/RAG/MultiDoc2Dial---The-Knowledge-Desk/codebase/inference/generator.py#L117-L140), `track_and_check_tickets(query, q_emb)` caches query embeddings into `data/ticket_cache.json`.
- When 3 or more queries arrive that have cosine similarity $> 0.85$ with each other, `advise_maker` trips to `True`.
- It logs an administrative alert: *"I am getting queries related to this domain often which is out of our domain so kindly provide me knowledge on this domain."*
- This provides an autonomous signal to administrators that users are asking for information the knowledge base doesn't yet cover.

---

## 5. The Machine Learning Models & Their Roles

| Component | Model Name | Parameters | Input / Output | Why This Specific Model? |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Retriever** | `sentence-transformers/all-MiniLM-L6-v2` | 22 Million | Input: Text string<br/>Output: 384-d normalized vector | Ultra-lightweight (80MB on disk), fast cosine dot-products, trained on 1B sentence pairs. |
| **Semantic Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22 Million | Input: `(query, passage)` pair<br/>Output: Float logit scalar | Unlike bi-encoders which encode query and doc separately, Cross-Encoder performs full cross-attention across both texts, catching subtle relevance nuances. |
| **Tool Policy Classifier** | `distilbert-base-uncased` | 66 Million | Input: Query token IDs<br/>Output: 3-class logits (`ANSWER`, `TICKET`, `REJECT`) | Distilled from BERT (40% smaller, 60% faster, retains 97% of language understanding). Ideal for low-latency classification. |
| **Generator LLM** | `google/flan-t5-base` | 248 Million | Input: Prompt tokens ($\le 512$)<br/>Output: Generated text tokens ($\le 300$) | Encoder-Decoder architecture trained on instruction-tuning tasks. Much less prone to runaway babbling than decoder-only models. |
| **PEFT / LoRA Adapters** | Custom LoRA on FLAN-T5 | ~1.7 Million (trainable) | Adapts attention projection layers ($W_q, W_v$) with rank $r=8, \alpha=32$ | Allows fine-tuning the model for preference alignment without changing the 248M base parameters or blowing up VRAM. |

---

## 6. The Mathematical Loss Formulations

This project rejected the "one-size-fits-all" training approach. Each component has a dedicated statistical loss tailored to its failure modes.

### 1. Retriever: Multiple Negatives Ranking (MNR) Loss
**Located in**: [`train_retriever.py`](file:///T:/DAIICT%20Resume/Projects/RAG/MultiDoc2Dial---The-Knowledge-Desk/codebase/training/train_retriever.py)  
**Problem**: How do you train an embedding model without having millions of explicitly hand-labeled negative passages?  
**Solution**: Contrastive in-batch negatives.

For a batch of $B$ positive pairs $(q_1, d_1^+), (q_2, d_2^+), \dots, (q_B, d_B^+)$, every document $d_j^+$ for $j \neq i$ serves as an implicit negative document for query $q_i$:

$$\mathcal{L}_{\text{MNR}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{\text{sim}(q_i, d_i^+) / \tau}}{\sum_{j=1}^{B} e^{\text{sim}(q_i, d_j^+) / \tau}}$$

- When $i=j$, the similarity is maximized.
- When $i \neq j$, the similarity is pushed down.
- **Result**: Dense vectors of queries cluster tightly around the exact section chunks that answer them.

---

### 2. Tool Policy: Triage Boundary Loss ($\mathcal{L}_{\text{boundary}}$)
**Located in**: [`train_tool_policy.py`](file:///T:/DAIICT%20Resume/Projects/RAG/MultiDoc2Dial---The-Knowledge-Desk/codebase/training/train_tool_policy.py#L157)  
**Problem**: Why does standard Cross-Entropy fail?  
Cross-Entropy only asks: *"Is the true class logit higher than others?"* If $p_{\text{true}} = 0.35$ and $p_{\text{false}} = 0.33$, Cross-Entropy considers this "progress". But in production, that 2% margin causes dangerous misclassifications.

**Solution**: We enforce a **hard geometric margin** $\mu = 0.5$ between the correct class probability and the highest runner-up class probability:

$$\mathcal{L}_{\text{boundary}} = \text{softplus}\left(p_{\text{wrong\_max}} - p_{\text{correct}} + \mu\right)$$

where $\text{softplus}(x) = \ln(1 + e^x)$, a smooth approximation of $\max(0, x)$.

```
   p_correct - p_wrong_max >= 0.5   ──▶   Loss approaches 0  (Model is confident)
   p_correct - p_wrong_max < 0.5    ──▶   Loss grows rapidly (Model is penalized)
```

---

### 3. Tool Policy: KB Proximity Signal ($\mathcal{L}_{\text{KB}}$)
**Located in**: [`train_tool_policy.py`](file:///T:/DAIICT%20Resume/Projects/RAG/MultiDoc2Dial---The-Knowledge-Desk/codebase/training/train_tool_policy.py#L259-L263)  
**Problem**: An out-of-domain query ($y = \text{REJECT}$) like *"How do I register a combat tank?"* uses words like "register", which might be mathematically close to the DMV knowledge base embeddings.

**Solution**: We calculate the minimum cosine distance between the query and all 4 domain centroid embeddings:
$$d_{\text{KB}}(q) = 1 - \max_{c \in \text{KB}} \cos(\phi(q), \phi(c))$$

If an out-of-domain query lands too close to the KB ($d_{\text{KB}} < \tau_0 = 0.8$), it triggers a softplus penalty:
$$\mathcal{L}_{\text{KB}} = \text{softplus}(\tau - d_{\text{KB}}(q))$$

Combined loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{boundary}} + 0.1 \cdot \mathcal{L}_{\text{KB}}$$

**Empirical Result**: Slashed the out-of-domain False Positive Rate from **18.5% down to 0.1%**!

---

### 4. Generator: Pairwise Preference Margin Loss (DPO-Style Alignment)
**Located in**: [`train_dpo_alignment.py`](file:///T:/DAIICT%20Resume/Projects/RAG/MultiDoc2Dial---The-Knowledge-Desk/codebase/training/train_dpo_alignment.py#L141-L144)  
**Problem**: How do you force FLAN-T5 to cite sources using `[1]`, be concise, and stop hallucinating, without setting up a massive, unstable Reinforcement Learning (PPO) pipeline with actor-critic networks?

**Solution**: Direct Pairwise Preference Ranking with margin $m=1.0$.

For every training query, we formulate:
- **Chosen Response ($y_w$)**: Ground-truth answer, truncated to $\le 3$ sentences, with appended citation token `[1]`.
- **Rejected Response ($y_l$)**: Verbose response with appended hallucination statement (*"I also think this requires analysis of irrelevant domains..."*).

During forward pass:
$$\mathcal{L}_{\text{chosen}} = \text{CrossEntropy}(y_w)$$
$$\mathcal{L}_{\text{rejected}} = \text{CrossEntropy}(y_l)$$
$$\mathcal{L}_{\text{Pairwise}} = \text{ReLU}\left(\mathcal{L}_{\text{chosen}} - \mathcal{L}_{\text{rejected}} + m\right)$$
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{chosen}} + 0.5 \cdot \mathcal{L}_{\text{Pairwise}}$$

- If the model assigns lower loss to the chosen sequence by at least margin $m$, $\mathcal{L}_{\text{Pairwise}} = 0$.
- If the model prefers the hallucinated sequence, the loss propagates strongly.
- **Empirical Result**: Citation adherence jumped from **14.2% to 98.5%**, and average answer length dropped from 145 tokens to 58 tokens.

---

## 7. File-by-File Codebase Blueprint

```
MultiDoc2Dial---The-Knowledge-Desk/
├── README.md                          <-- Root presentation and research overview
├── PROJECT_DEEP_DIVE.md               <-- THIS FILE (Your private comprehensive guide)
├── .gitignore                         <-- Properly configured to ignore heavy weights and raw data
├── codebase/
│   ├── README.md                      <-- Original dataset citation notes
│   ├── requirements.txt               <-- Python dependency specifications
│   │
│   ├── data/                          <-- (Generated / Cached Assets)
│   │   ├── multidoc2dial_doc.json     <-- Raw documents (46 MB)
│   │   ├── multidoc2dial_dial_*.json  <-- Train, dev, test dialogue splits
│   │   ├── kb.json                    <-- Cleaned 9,203 section-level chunks (4 MB)
│   │   ├── kb_meta.json               <-- Metadata mapping chunk_id to doc_id and domain
│   │   ├── kb.index                   <-- Flat L2 FAISS index (14 MB)
│   │   ├── kb_embeddings.npy          <-- Raw float32 numpy vector cache (14 MB)
│   │   └── ticket_cache.json          <-- Dynamic query cache for OOD domain clustering
│   │
│   ├── data_pipeline/
│   │   ├── build_kb.py                <-- Parses multidoc2dial_doc.json into 9,203 clean chunks
│   │   └── build_retriever.py         <-- Encodes kb.json into 384-d vectors, builds FAISS index & npy
│   │
│   ├── inference/
│   │   ├── generator.py               <-- THE CORE PIPELINE: 4-Phase triage, routing, and generation
│   │   └── tools.py                   <-- Implementation of SearchKB, GetPolicy, CreateTicket
│   │
│   ├── training/
│   │   ├── train_retriever.py         <-- MNR contrastive training for sentence-transformers
│   │   ├── train_tool_policy.py       <-- 3-Class DistilBERT with Triage Boundary + KB Proximity Loss
│   │   ├── train_generator.py         <-- PEFT/LoRA fine-tuning for flan-t5-base
│   │   └── train_dpo_alignment.py     <-- Pairwise Preference Margin training (DPO-style)
│   │
│   ├── evaluation/
│   │   ├── evaluate.py                <-- 18-query comparative benchmark (Baseline vs. Improved)
│   │   ├── measure.py                 <-- Prints chunk distributions across the 4 domains
│   │   └── test_retriever.py          <-- Interactive CLI to inspect raw FAISS retrieval
│   │
│   └── demo/
│       ├── app.py                     <-- Flask backend serving web UI and /api/chat endpoint
│       └── static/
│           ├── index.html             <-- Modern split-screen dashboard UI
│           ├── style.css              <-- Dark slate theme, animations, and domain badges
│           └── app.js                 <-- Interactive cards, citation clicks, relevance bars
│
└── report/
    ├── project_summary.txt            <-- High-level project summary
    ├── report_acl2026.tex             <-- ACL 2026 conference-formatted paper draft
    ├── final_report.tex               <-- Extended LaTeX research report with full math
    ├── loss_ablation_study.csv        <-- Empirical loss ablation data
    ├── grounding_eval.csv             <-- Citation and grounding comparison data
    ├── latency_benchmarks.json        <-- Phase-by-phase compute latency measurements
    └── dpo_alignment_metrics.txt      <-- Reward training progression log
```

---

## 8. Ingenious Engineering Details

Here are the subtle, real-world implementation details that make this codebase production-hardened:

### 1. Windows PowerShell Emoji Crash Prevention
In [`generator.py`](file:///T:/DAIICT%20Resume/Projects/RAG/MultiDoc2Dial---The-Knowledge-Desk/codebase/inference/generator.py#L8-L11):
```python
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')
```
*Why*: On Windows, standard `cmd` or `powershell` defaults to `cp1252` encoding. Any terminal `print("🛡 ESCALATION")` will crash Python with a fatal `UnicodeEncodeError`. Reconfiguring `sys.stdout` to UTF-8 fixes this silently.

### 2. Dual Persistent Storage (`kb.index` + `kb_embeddings.npy`)
In [`build_retriever.py`](file:///T:/DAIICT%20Resume/Projects/RAG/MultiDoc2Dial---The-Knowledge-Desk/codebase/data_pipeline/build_retriever.py#L55-L60):
*Why*: FAISS indexes store vectors in an optimized C++ structure. But if you want to perform direct numpy math (e.g. matrix multiplying against domain centroids or clustering), unpacking FAISS vectors is slow. Saving both the FAISS binary index and raw `.npy` numpy matrix gives the best of both worlds.

### 3. Normalization for True Cosine Similarity
In [`generator.py`](file:///T:/DAIICT%20Resume/Projects/RAG/MultiDoc2Dial---The-Knowledge-Desk/codebase/inference/generator.py#L317):
```python
q_emb = embed_model.encode([query], normalize_embeddings=True)[0]
```
*Why*: A dot product $\mathbf{a} \cdot \mathbf{b}$ only equals cosine similarity if $\|\mathbf{a}\| = 1$ and $\|\mathbf{b}\| = 1$. By enforcing `normalize_embeddings=True`, `np.dot(q_emb, item["embedding"])` is mathematically guaranteed to be in $[-1, 1]$.

### 4. DirectML Opaque Tensor Storage Workaround
In [`train_tool_policy.py`](file:///T:/DAIICT%20Resume/Projects/RAG/MultiDoc2Dial---The-Knowledge-Desk/codebase/training/train_tool_policy.py#L284):
```python
model.to('cpu')
model.save_pretrained(output_dir)
```
*Why*: On AMD Radeon GPUs using `torch-directml`, saving a model directly from the DML device throws `RuntimeError: DirectML tensor storage is opaque`. Moving weights back to `'cpu'` before calling HuggingFace `save_pretrained` completely prevents this crash.

---

## 9. Empirical Results, Ablations & Benchmarks

### Summary Table: Baseline vs. Knowledge Desk Pipeline
Tested across 18 benchmark queries in [`codebase/evaluation/evaluate.py`](file:///T:/DAIICT%20Resume/Projects/RAG/MultiDoc2Dial---The-Knowledge-Desk/codebase/evaluation/evaluate.py):

| Metric | Baseline RAG | The Knowledge Desk | Why It Improved |
| :--- | :---: | :---: | :--- |
| **Tool / Escalation Accuracy** | 50.0% | **90.0% – 100%** | The baseline blindly answers everything; our pipeline stops OOD queries early. |
| **Triage Boundary Precision ($TBP_{0.5}$)** | N/A | **90.0% – 99.8%** | Triage Boundary Loss prevents classification near the 50% confusion margin. |
| **Average Grounding Score** | 0.4038 | **0.7261** | LoRA preference alignment trains the generator to stick strictly to the context. |
| **Average Relevance Score** | 0.4378 | **0.5702** | Cross-Encoder reranking ensures only truly relevant chunks are fed to the model. |
| **Hallucination Count** | 10 | **0** | Gated escalation: if rerank score $< -2.0$, it creates a ticket instead of guessing. |
| **Average Latency** | 2.27 sec | 4.83 sec | Multi-tier validation overhead (acceptable trade-off for 0 hallucinations). |

---

## 10. Master Viva & Technical Interview Cheat Sheet

If an examiner or interviewer grills you on this project, here are the exact answers:

### Q1: "Why not just use GPT-4 with a system prompt like 'Refuse to answer if not in context'?"
> **Answer**: Prompt instructions are **probabilistic and soft**. LLMs exhibit *sycophancy* and *stochastic leakage*—if an adversary crafts a clever query, the model will still hallucinate. Furthermore, running GPT-4 on every trivial query costs money and adds high API latency. Our architecture is **deterministic and hard-gated**: Phase 1 drops non-domain queries in 0.15ms on CPU; Phase 2 routes with an explicit neural policy; and Phase 3 physically denies context to the LLM if the cross-encoder score is below $-2.0$.

### Q2: "What is the difference between a Bi-Encoder and a Cross-Encoder?"
> **Answer**: 
> - **Bi-Encoder** (`all-MiniLM-L6-v2`): Encodes query and passage independently into separate vectors. You can precompute document vectors ahead of time and search in sub-milliseconds using FAISS. However, it cannot model cross-word interactions between the query and passage.
> - **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`): Takes `[CLS] Query [SEP] Passage [SEP]` as a single concatenated input and runs full multi-head self-attention across both texts simultaneously. It is much more accurate at detecting semantic synergy, but you cannot precompute vectors—you can only run it on a small set of candidates (e.g. top-15).

### Q3: "What is TBP and why did you invent it?"
> **Answer**: Standard accuracy is misleading in triage tasks. If a model predicts `SearchKB` with 34% probability and `CreateTicket` with 33% probability, it gets labeled as "correct" under standard argmax accuracy, but it was essentially guessing. **Triage Boundary Precision ($TBP_\mu$)** only counts a prediction as correct if the winning probability exceeds the runner-up by a defined confidence margin $\mu$ (we set $\mu = 0.5$).

### Q4: "Why did you choose FLAN-T5 instead of a causal LLM like LLaMA or Mistral?"
> **Answer**: FLAN-T5 is an **Encoder-Decoder (Seq2Seq)** model. Encoder-decoder architectures have bidirectional attention on the input source context and causal attention on the target tokens. This makes them inherently better at grounded summarization and information extraction than autoregressive decoder-only models, which tend to drift into free-form text generation.

### Q5: "What are the limitations of your current design?"
> **Answer**: 
> 1. Autoregressive beam search on CPU takes ~1.5s per response. In production, we would use GPU quantization (FP16/INT4) or vLLM/ONNX runtime.
> 2. The cross-encoder threshold of $-2.0$ is currently a fixed hyperparameter calibrated on MS-MARCO. An adaptive threshold based on query length would be a valuable next step.
> 3. The `CreateTicket` tool currently logs locally rather than dispatching authenticated webhooks to Jira or ServiceNow.

---

## 11. How to Rebuild, Retrain, and Run Everything

### 1. One-Command Evaluation (Verification)
To verify everything is running smoothly and reproduce all metrics:
```powershell
python codebase/evaluation/evaluate.py
```

### 2. Launch the Web UI Dashboard
Open your browser to `http://localhost:5000`:
```powershell
python codebase/demo/app.py
```

### 3. Interactive CLI Testing
```powershell
python codebase/inference/generator.py
```

### 4. How to Rebuild the Entire Knowledge Base from Scratch
If you ever delete `codebase/data/`:
```powershell
# 1. Download and extract raw multidoc2dial dataset
python -c "import urllib.request, zipfile, io, os; data = urllib.request.urlopen('http://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip').read(); z = zipfile.ZipFile(io.BytesIO(data)); z.extractall('codebase/data')"

# 2. Extract 9,203 clean section-level chunks
python codebase/data_pipeline/build_kb.py

# 3. Compute dense vectors, FAISS index, and persistent .npy cache
python codebase/data_pipeline/build_retriever.py
```

### 5. Retraining the Models (Optional)
- **Retriever**: `python codebase/training/train_retriever.py`
- **Tool Policy**: `python codebase/training/train_tool_policy.py`
- **LoRA Generator**: `python codebase/training/train_generator.py`
- **Pairwise Alignment**: `python codebase/training/train_dpo_alignment.py`

---
*Created as a comprehensive master guide for the MultiDoc2Dial Knowledge Desk Project.*
