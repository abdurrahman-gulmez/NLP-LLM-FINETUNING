# LoRA Fine-Tuning

This repository contains the implementation of LoRA fine-tuning on the Qwen2.5-Coder-1.5B-Instruct model using two distinct datasets: **CodeGen-Deep-5K** and **CodeGen-Diverse-5K**. The project focuses on enhancing code generation capabilities specifically using the `solution` field (code-only training).

## 📂 Datasets
* **DEEP Dataset:** [Naholav/CodeGen-Deep-5K](https://huggingface.co/datasets/Naholav/CodeGen-Deep-5K) - Contains deeper reasoning traces.
* **DIVERSE Dataset:** [Naholav/CodeGen-Diverse-5K](https://huggingface.co/datasets/Naholav/CodeGen-Diverse-5K) - Contains a wider variety of problem types.
* **Source:** NVIDIA OpenCode Reasoning dataset.

## ⚙️ Hyperparameters
The following configuration was used for Deep training runs as the project requirements:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Base Model** | Qwen2.5-Coder-1.5B-Instruct | Unsloth optimized version |
| **LoRA Rank (r)** | 32 | Matrix rank |
| **LoRA Alpha** | 64 | r * 2 |
| **LoRA Dropout** | 0.1 | Regularization |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Attention & MLP layers |
| **Learning Rate** | 2e-4 | Cosine decay scheduler |
| **Batch Size** | 16 | Effective batch size |
| **Context Length** | 1024 | Max sequence length |
| **Epochs** | 3 | Training duration |
| **Optimizer** | AdamW (8-bit) | With gradient clipping (1.0) |
| **Weight Decay** | 0.01 |  |
| **Warmup Ratio** | 0.03 |  |

The following configuration was used for Diverse training runs as the project requirements:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Base Model** | Qwen2.5-Coder-1.5B-Instruct | Unsloth optimized version |
| **LoRA Rank (r)** | 32 | Matrix rank |
| **LoRA Alpha** | 64 | r * 2 |
| **LoRA Dropout** | 0.05 | Regularization |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Attention & MLP layers |
| **Learning Rate** | 2e-4 | Cosine decay scheduler |
| **Batch Size** | 16 | Effective batch size |
| **Context Length** | 1024 | Max sequence length |
| **Epochs** | 3 | Training duration |
| **Optimizer** | AdamW (8-bit) | With gradient clipping (1.0) |
| **Weight Decay** | 0.01 |  |
| **Warmup Ratio** | 0.04 |  |

## 📊 Training Logs & Results
Training logs including loss values for every 20 steps (train) and 100 steps (eval) are located in the `experiments/` directory.

### Model 1: DEEP Training
* **Final Train Loss:** 0.0878
* **Final Eval Loss:** 0.13680578768253326
* **Hugging Face Model:** [Link Ekle]

### Model 2: DIVERSE Training
* **Final Train Loss:** 0.7709
* **Final Eval Loss:** 0.9457
* **Hugging Face Model:** [Link Ekle]

## 🚀 Usage
To run the inference using the fine-tuned adapters:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "YourUsername/Your-Model-Name", # Replace with HF model ID
    load_in_4bit = False,
)
FastLanguageModel.for_inference(model)
test_prompt = """Write a Python function that takes a list of integers and returns the sum of all even numbers."""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": test_prompt}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=1024,
    temperature=0.6,
    do_sample=True,
    top_p=0.9,
    pad_token_id=tokenizer.pad_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Output:")
print(response)
