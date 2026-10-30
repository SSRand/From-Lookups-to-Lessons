from __future__ import annotations

from typing import Final


DEFAULT_PROMPT_PROTOCOL: Final[str] = "question-only-v1"

PROMPT_TEMPLATES: Final[dict[str, str]] = {
    "question-only-v1": "Answer these questions:\nQuestion: {question}\nAnswer:",
    "concise-format-v1": (
        "Answer with only the shortest possible answer span.\n"
        "Do not explain or write a complete sentence.\n"
        "If the answer is yes or no, output exactly \"yes\" or \"no\".\n\n"
        "Question: {question}\nAnswer:"
    ),
}


def resolve_prompt_protocol(prompt_protocol: str | None = None) -> str:
    protocol = str(prompt_protocol or DEFAULT_PROMPT_PROTOCOL).strip()
    if protocol not in PROMPT_TEMPLATES:
        raise ValueError(f"unknown prompt protocol: {protocol}")
    return protocol


def build_prompt(question: str, *, prompt_protocol: str = DEFAULT_PROMPT_PROTOCOL) -> str:
    clean_question = str(question).strip()
    if not clean_question:
        raise ValueError("question must be non-empty")
    return PROMPT_TEMPLATES[resolve_prompt_protocol(prompt_protocol)].format(question=clean_question)
