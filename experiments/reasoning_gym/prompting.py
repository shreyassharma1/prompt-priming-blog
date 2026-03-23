"""Prompt helpers for the Countdown prompt-priming experiments."""

from __future__ import annotations

from typing import Literal

from tinker_cookbook import renderers

PromptStyle = Literal["zero_shot", "few_shot"]

LLAMA_REASONING_SYSTEM_PROMPT = (
    "Reason through each problem step by step before giving the final answer. "
    "State the final answer clearly at the end."
)

COUNTDOWN_FEWSHOT_EXAMPLE: list[renderers.Message] = [
    {
        "role": "user",
        "content": (
            "Find a way to make 150 using all of these numbers: 25, 3, 75.\n"
            "Each number can only be used once.\n\n"
            "Final answer format instructions:\n"
            "1. Provide your solution as an arithmetic expression (no '=' sign).\n"
            "2. Do not include the target number in the expression.\n"
            "3. Use '*' for multiplication.\n"
            "4. Use '/' for division.\n"
            "5. Do not include any other text or formatting."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "I need to make 150 from 25, 3, and 75. Let me try: 75 + 25 = 100. "
            "That leaves 3. 100 * 3 = 300, too big. What about 25 * 3 = 75, "
            "then 75 + 75 = 150. That works!\n\n"
            "25 * 3 + 75"
        ),
    },
]


def _is_llama_model(model_name: str) -> bool:
    return model_name.startswith("meta-llama/")


def build_llama_convo_prefix(
    model_name: str,
    prompt_style: PromptStyle = "zero_shot",
) -> list[renderers.Message] | None:
    if not _is_llama_model(model_name):
        return None

    messages: list[renderers.Message] = [
        {"role": "system", "content": LLAMA_REASONING_SYSTEM_PROMPT}
    ]
    if prompt_style == "few_shot":
        messages.extend(COUNTDOWN_FEWSHOT_EXAMPLE)
    return messages
