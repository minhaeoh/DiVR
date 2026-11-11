"""Prompts. Every prompt is one user turn in the generator's own chat template, followed by a fixed
response prefix that belongs to the prompt (the model continues from it). GSM8K, MATH-500 and
Countdown use d1's instructions (Zhao et al., 2025) and the prefix "<reasoning>"; the code tasks open
a Python code fence.
"""
from __future__ import annotations

#: d1's GSM8K / MATH instruction, verbatim.
MATH_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""

#: d1's Countdown instruction, verbatim (including its missing space before "Respond").
COUNTDOWN_SYSTEM_PROMPT = (
    "Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target "
    "number. You may use the operations +, -, *, and / as needed, but each number must be used exactly once. Think "
    "step-by-step. After reasoning, provide only your final expression inside \\boxed{} tags without including an "
    "equals sign or the target number. For example: \\boxed{a + b * c}"
    "Respond in the following format:\n<reasoning>\nYour reasoning here\n</reasoning>\n<answer>\n\\boxed{...}\n</answer>")

HUMANEVAL_INSTRUCTION = ("Complete the following Python function. Write the complete function, including its signature, "
                         "inside a single ```python code block.\n\n")
MBPP_INSTRUCTION = ("You are an expert Python programmer. Write a Python function for the task below. Your code must "
                    "pass the given tests. Put the complete code inside a single ```python code block.\n\n")

CODE_SPLITS = ("humaneval", "mbpp", "mbpp_train")
REASONING_PREFIX = "<reasoning>"


def chat(tok, user: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)


def build_prompt(tok, split: str, question: str) -> str:
    """The full prompt string of a problem: chat-templated user turn + response prefix."""
    if split in CODE_SPLITS:
        instr = HUMANEVAL_INSTRUCTION if split == "humaneval" else MBPP_INSTRUCTION
        return chat(tok, instr + question.strip()) + "```python\n"
    if split == "countdown":
        return chat(tok, COUNTDOWN_SYSTEM_PROMPT + "\n\n" + question.strip()) + REASONING_PREFIX
    return chat(tok, MATH_SYSTEM_PROMPT + "\n\n" + question.strip()) + REASONING_PREFIX
