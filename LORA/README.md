# Competitive Code Reasoning: LoRA Fine-Tuning Project

This project involves fine-tuning the **Qwen2.5-Coder-1.5B-Instruct** base model using LoRA (Low-Rank Adaptation) to enhance Python code generation capabilities. The model was trained separately on two distinct datasets: **DEEP** and **DIVERSE**.

## 🚀 Trained Models (Hugging Face)

The fine-tuned models are hosted on the Hugging Face Hub:

| Model Type | Dataset | Model Link |
|------------|---------|-------------|
| **DEEP** | `Naholav/CodeGen-Deep-5K` | [DEEP MODEL](https://huggingface.co/abdurrahman-gulmez/Qwen2.5-Coder-1.5B-Instruct-Deep-Finetuning) |
| **DIVERSE**| `Naholav/CodeGen-Diverse-5K` | [🔗 Link to your DIVERSE Model] |

> **Note:** These models were trained using the **solution-only** (code-only) fields as per the project requirements.

## 📂 Project Structure

This repository contains:

- `train_lora.ipynb`: The main training notebook utilizing the Unsloth library.
- `logs/`: Directory containing training and evaluation logs (JSONL format).
  - `training_log_deep.jsonl`
  - `training_log_diverse.jsonl`
- `evaluation_script.py` (or embedded in notebook): Scripts used to evaluate model performance on test splits.

## ⚙️ Hyperparameters

The following configuration was used for training, adhering to the project guidelines:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Base Model** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B parameter instruct model |
| **LoRA Rank (r)** | 32 | Adapter rank |
| **LoRA Alpha** | 64 | r * 2 |
| **LoRA Dropout** | 0.05 | - |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | All linear layers |
| **Learning Rate** | 2e-4 | Initial learning rate |
| **Batch Size** | 2 | Per device train batch size |
| **Gradient Accumulation** | 8 | - |
| **Effective Batch Size** | 16 | (2 * 8) |
| **Epochs** | 3 | Total training epochs |
| **Context Length** | 1024 | Max token length |
| **Optimizer** | AdamW (8-bit) | Memory optimized optimizer |
| **Scheduler** | Cosine | Cosine decay schedule |
| **Weight Decay** | 0.01 | - |
| **Warmup Ratio** | 0.03 | - |

## 📊 Training Logs

Loss values were logged every 20 steps (training) and 100 steps (validation).

- **Deep Model Final Loss:** [Insert final training loss here, e.g., 0.2837]
- **Diverse Model Final Loss:** [Insert final training loss here]

Detailed logs can be found in the `logs/` directory.

## 🛠️ Installation & Usage

To reproduce the training or run inference locally:

**1. Install Dependencies:**
```bash
pip install unsloth trl peft accelerate bitsandbytes
