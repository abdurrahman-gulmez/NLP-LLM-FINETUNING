# Natural Language Processing Project

This repository presents a **comprehensive portfolio of advanced Large Language Model (LLM) systems** focused on technical domains such as Python programming, data science, and competitive programming. It integrates **Retrieval-Augmented Generation (RAG)**, **LoRA-based fine-tuning**, and **autonomous ReAct agents** into a cohesive, production-oriented framework.

---

## Project Overview

The **NLP-LLM-FINETUNING** repository demonstrates how modern LLM techniques can be combined to:

* Reduce hallucinations via document-grounded generation (RAG),
* Improve code reasoning and generation through efficient instruction fine-tuning (LoRA),
* Solve complex multi-hop technical tasks using autonomous reasoning-and-action agents (ReAct).

This work is suitable for **research**, **industrial prototyping**, and **real-world technical assistants**.

---

## Core Projects

### Hybrid RAG Academic Assistant

A **production-ready Retrieval-Augmented Generation system** designed to answer technical questions using verified academic and developer documentation (e.g., NumPy, OpenCV, Matplotlib PDFs).

#### Objective

To minimize hallucinations by grounding model outputs in authoritative, document-level evidence with explicit citations.

#### Methodology

* **Document Processing**:

  * PDFs loaded via **PyMuPDF**
  * Chunking strategy: **800 tokens** with **150-token overlap** to preserve semantic context

* **Hybrid Retrieval Strategy**:

  * **Semantic Search**: ChromaDB + `all-mpnet-base-v2` embeddings
  * **Keyword Search**: BM25 Okapi

* **Advanced Re-Ranking**:

  * Cross-Encoder: `ms-marco-MiniLM-L-6-v2`
  * Top 10 candidates re-ranked → Top 5 passed to the LLM

* **Generation Model**:

  * `Qwen3-4B-Instruct-2507`
  * Streaming responses with **page-level citations**

This architecture ensures factual accuracy while maintaining fluent, contextual answers.

---

### LoRA Fine-Tuning & Optimization

This module focuses on improving **code generation and reasoning** by fine-tuning the **Qwen2.5-Coder-1.5B-Instruct** model using parameter-efficient LoRA adapters.

#### Datasets

* **DEEP Dataset (5K samples)**

  * Structured, step-by-step reasoning traces
  * Emphasizes algorithmic and logical consistency

* **DIVERSE Dataset (5K samples)**

  * Broad coverage of programming problem types
  * Encourages generalization across varied prompts

#### Training Configuration

All models were trained using **Unsloth** with the following hyperparameters:

| Parameter     | DEEP | DIVERSE |
| ------------- | ---- | ------- |
| LoRA Rank (r) | 32   | 32      |
| LoRA Alpha    | 64   | 64      |
| LoRA Dropout  | 0.1  | 0.05    |
| Learning Rate | 2e-4 | 2e-4    |
| Batch Size    | 16   | 16      |
| Epochs        | 3    | 3       |
| Warmup Ratio  | 0.03 | 0.04    |

#### Training Results

* **DEEP Model**

  * Train Loss: **0.0878**
  * Eval Loss: **0.1368**

* **DIVERSE Model**

  * Train Loss: **0.7709**
  * Eval Loss: **0.9457**

**Analysis**:
The DEEP model exhibits superior alignment with structured reasoning tasks, while the DIVERSE model reflects the increased complexity of broader task distributions.

---

### Benchmark Evaluation: LiveCodeBench

The fine-tuned models were evaluated using **LiveCodeBench** (AtCoder, Easy problems, releases 2408–2502).

#### Pass@1 Results

| Model         | Pass@1     | Solved      |
| ------------- | ---------- | ----------- |
| Base Model    | 21.9%     | 9 / 41      |
| Deep Finetuned Model    | **24.4%** | **10 / 41** |
| Diverse-Finetuned Model | **26.8%** | **11 / 41** |

#### Insights

* Both instruction-tuned models outperform the base model
* **Deep model** excels in multi-step logical reasoning
* **Diverse model** handles varied problem formulations more flexibly

These results validate the effectiveness of high-quality reasoning-oriented instruction tuning.

---

### ReAct Agent: Autonomous Technical Assistant

The **ReAct (Reason + Act)** agent serves as the orchestration layer, enabling autonomous multi-step problem solving.

#### Architecture

* **Reasoning Loop**: `Thought → Action → Observation`
* **Primary Model**: `Llama-3.3-70b-versatile`

  * Selected after smaller 8B models failed to maintain long-horizon reasoning

#### Tools

* `search_docs`: Targeted RAG document retrieval
* `web_search`: DuckDuckGo for real-time information
* `calculator`: Secure numerical computation

#### Safeguards & Reliability

* **Loop Protection**: `used_actions` tracker prevents repeated tool calls
* **Forced Reasoning**: Manual injection of `Thought:` after each observation
* **Multi-Hop Support**: Handles chained queries (retrieve → compute → respond)
* **Safety Limit**: Hard cap of **8 reasoning steps** to prevent infinite loops

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/abdurrahman-gulmez/NLP-LLM-FINETUNING.git
cd NLP-LLM-FINETUNING
```

### 2. Install Dependencies

```bash
pip install -r REACT/requirements.txt
pip install -r LORA/requirements.txt
```

### 3. Run Applications

* **ReAct Agent**:

```bash
streamlit run REACT/react.py
```

* **RAG Academic Assistant**:

```bash
streamlit run RAG/rag.py
```

---

## Conclusion

This repository demonstrates a **robust and scalable integration of modern LLM paradigms**. This project includes:

* **RAG** for factual grounding,
* **LoRA fine-tuning** for specialized reasoning and coding performance,
* **ReAct agents** for multi-step problem solving.

---


