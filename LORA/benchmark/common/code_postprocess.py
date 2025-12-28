"""
Code postprocessing utilities for benchmark evaluation.

Author: naholav

Handles:
- Removing <think>...</think> blocks
- Extracting code from markdown code blocks
- Cleaning up generated code for execution
"""

import re
from typing import Optional


def remove_think_tags(text: str) -> str:
    """
    Remove <think>...</think> blocks from generated text.
    Also handles UNCLOSED think blocks (when model gets cut off).

    Args:
        text: Generated text that may contain <think> tags

    Returns:
        Text with <think> blocks removed
    """
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    if '<think>' in text.lower():
        think_pos = text.lower().find('<think>')
        code_after_think = re.search(r'```(?:python)?\s*\n', text[think_pos:], re.IGNORECASE)

        if code_after_think:
            code_start = think_pos + code_after_think.start()
            text = text[:think_pos] + text[code_start:]
        else:
            text = ""

    return text.strip()


def extract_code_from_markdown(text: str) -> Optional[str]:
    """
    Extract Python code from markdown code blocks.

    Args:
        text: Text that may contain ```python ... ``` blocks

    Returns:
        Extracted code or None if no code block found
    """
    pattern = r'```python\s*(.*?)```'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()

    pattern = r'```\s*(.*?)```'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()

    return None


def extract_code(generated_text: str, fallback_to_full_text: bool = True) -> str:
    """
    Extract clean Python code from generated text.

    This function:
    1. Removes <think>...</think> blocks
    2. Extracts code from markdown blocks
    3. Falls back to full text if no code block found

    Args:
        generated_text: Raw generated text from model
        fallback_to_full_text: If True and no code block found, return cleaned full text

    Returns:
        Clean Python code ready for execution
    """
    text = remove_think_tags(generated_text)
    code = extract_code_from_markdown(text)

    if code:
        return code

    if fallback_to_full_text:
        return text.strip()

    return ""


def remove_main_wrapper(code: str) -> str:
    """
    Remove competitive programming wrappers (if __name__ == "__main__" and def main()).

    Args:
        code: Python code that may have main() wrappers

    Returns:
        Code with main wrappers removed
    """
    code = re.sub(r'\n*if\s+__name__\s*==\s*["\']__main__["\']\s*:\s*\n.*', '', code, flags=re.DOTALL)

    lines = code.split('\n')

    first_code_line = None
    first_code_idx = 0
    for i, line in enumerate(lines):
        if line.strip():
            first_code_line = line.strip()
            first_code_idx = i
            break

    if first_code_line and re.match(r'^def\s+main\s*\(\s*\)\s*:', first_code_line):
        body_lines = lines[first_code_idx + 1:]
        dedented_lines = []
        for line in body_lines:
            if line.startswith('    '):
                dedented_lines.append(line[4:])
            elif line.strip() == '':
                dedented_lines.append('')
            else:
                break
        code = '\n'.join(dedented_lines)

    return code.strip()


def clean_code_for_execution(code: str) -> str:
    """
    Final cleanup before code execution.

    Args:
        code: Python code to clean

    Returns:
        Cleaned code
    """
    code = code.strip()
    code = code.replace('```python', '').replace('```', '')

    return code.strip()


def postprocess_generated_code(generated_text: str) -> str:
    """
    Main postprocessing pipeline for generated code.

    This is the main function to use for benchmark evaluation.

    Args:
        generated_text: Raw text from model generation

    Returns:
        Clean Python code ready for execution
    """
    code = extract_code(generated_text, fallback_to_full_text=True)
    code = clean_code_for_execution(code)

    return code
