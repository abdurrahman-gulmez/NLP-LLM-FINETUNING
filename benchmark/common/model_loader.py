"""
Model loading utilities for benchmark evaluation.

Author: naholav

Supports:
- Loading base models
- Loading LoRA fine-tuned checkpoints
- BF16 precision
- Flash Attention 2
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from typing import Tuple, Optional, List


class StopOnStrings(StoppingCriteria):
    """Stop generation when specific strings are detected in the NEW output only."""

    def __init__(self, tokenizer, stop_strings: List[str], input_length: int):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.input_length = input_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        new_tokens = input_ids[0, self.input_length:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

        for stop_string in self.stop_strings:
            if stop_string in generated_text:
                return True

        return False


def load_base_model(
    model_name: str,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_flash_attention_2: bool = True,
    trust_remote_code: bool = True,
    token: Optional[str] = None
):
    """
    Load base model (without LoRA).

    Args:
        model_name: HuggingFace model name or path
        torch_dtype: Torch dtype (default: bfloat16)
        device_map: Device mapping strategy
        use_flash_attention_2: Whether to use Flash Attention 2
        trust_remote_code: Whether to trust remote code
        token: HuggingFace token (optional)

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        token=token
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_flash_attention_2:
        tokenizer.padding_side = 'left'

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }

    if token:
        model_kwargs["token"] = token

    if use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    print(f"Model loaded ({torch_dtype})")
    if use_flash_attention_2:
        print("Flash Attention 2 enabled")

    return model, tokenizer


def load_lora_checkpoint(
    checkpoint_path: str,
    base_model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_flash_attention_2: bool = True,
    trust_remote_code: bool = True,
    token: Optional[str] = None
):
    """
    Load LoRA checkpoint on top of base model.

    Args:
        checkpoint_path: Path to LoRA checkpoint
        base_model_name: Base model name
        torch_dtype: Torch dtype (default: bfloat16)
        device_map: Device mapping strategy
        use_flash_attention_2: Whether to use Flash Attention 2
        trust_remote_code: Whether to trust remote code
        token: HuggingFace token (optional)

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\nLoading LoRA checkpoint: {checkpoint_path}")
    print(f"Base model: {base_model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=trust_remote_code
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=trust_remote_code,
            token=token
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_flash_attention_2:
        tokenizer.padding_side = 'left'

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }

    if token:
        model_kwargs["token"] = token

    if use_flash_attention_2:
        model_kwargs["attn_implementation"] = "sdpa"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    print(f"LoRA checkpoint loaded ({torch_dtype})")
    if use_flash_attention_2:
        print("Flash Attention 2 enabled")

    return model, tokenizer


def generate_code(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = None,
    num_return_sequences: int = 1,
    use_chat_template: bool = True,
    system_prompt: str = None,
) -> list[str]:
    """
    Generate code from prompt with automatic stopping on hallucinations.

    Args:
        model: Model to use for generation
        tokenizer: Tokenizer
        prompt: Input prompt (user message content)
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        top_p: Nucleus sampling parameter
        do_sample: Whether to sample (auto-determined from temperature if None)
        num_return_sequences: Number of sequences to return
        use_chat_template: Whether to use chat template for Qwen models
        system_prompt: System prompt to use (required for proper generation)

    Returns:
        List of generated code strings
    """
    if do_sample is None:
        do_sample = temperature > 0.0

    if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors='pt').to(model.device)
    input_length = inputs['input_ids'].shape[1]

    stop_strings = [
        "```\n\nHuman:",
        "\n\n\n\n\n\n",
        "Human: I'm not sure",
        "</problem>",
        "</systemoutput>",
        "</usercode>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|im_start|>",
    ]

    stopping_criteria = StoppingCriteriaList([
        StopOnStrings(tokenizer, stop_strings, input_length)
    ])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    generated_texts = []
    for output in outputs:
        generated_tokens = output[inputs['input_ids'].shape[1]:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_texts.append(text)

    return generated_texts
