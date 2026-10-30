"""Evidence prompt packing for BM25-RAG (ranked top-5 passages, input budget in tokens)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from resmem.prompting import PROMPT_TEMPLATES
from resmem.scoring import normalize_text

PROMPT_PREAMBLE = "Use the retrieved evidence to answer the question."
RAG_PROMPT_PROTOCOL = "concise-format-v1"
CONCISE_PROMPT_INSTRUCTIONS = PROMPT_TEMPLATES[RAG_PROMPT_PROTOCOL].split("\n\nQuestion:", maxsplit=1)[0]
DEFAULT_PROMPT_TOP_K = 5
DEFAULT_MAX_DOCUMENT_TOKENS = 64


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    text: str


def _render_context_prompt(question: str, context_lines: Sequence[str]) -> str:
    return (
        f"{PROMPT_PREAMBLE}\n"
        f"{CONCISE_PROMPT_INSTRUCTIONS}\n\n"
        "Context:\n"
        f"{chr(10).join(context_lines)}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _token_count(tokenizer, text: str, *, add_special_tokens: bool) -> int:
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def _trim_document_to_token_budget(
    text: str,
    *,
    tokenizer,
    max_document_tokens: int,
) -> tuple[str, bool]:
    collapsed = re.sub(r"\s+", " ", str(text)).strip()
    if not collapsed:
        return "", False
    if _token_count(tokenizer, collapsed, add_special_tokens=False) <= max_document_tokens:
        return collapsed, False
    words = collapsed.split()
    fitted_text, _token_count_value = _longest_word_prefix_within_budget(
        words,
        upper_bound=len(words) - 1,
        count_tokens=lambda candidate: _token_count(tokenizer, candidate, add_special_tokens=False),
        token_budget=max_document_tokens,
    )
    return fitted_text or "", True


def _longest_word_prefix_within_budget(
    words: Sequence[str],
    *,
    upper_bound: int,
    count_tokens: Callable[[str], int],
    token_budget: int,
) -> tuple[str | None, int | None]:
    """Return the longest whitespace-delimited prefix that fits a token budget."""
    low = 0
    high = min(upper_bound, len(words))
    fitted_token_count: int | None = None
    while low < high:
        candidate_word_count = (low + high + 1) // 2
        candidate = " ".join(words[:candidate_word_count])
        candidate_token_count = count_tokens(candidate)
        if candidate_token_count <= token_budget:
            low = candidate_word_count
            fitted_token_count = candidate_token_count
        else:
            high = candidate_word_count - 1
    if low == 0:
        return None, None
    candidate = " ".join(words[:low])
    if fitted_token_count is None:
        fitted_token_count = count_tokens(candidate)
    return candidate, fitted_token_count


def _fit_document_within_prompt_budget(
    *,
    tokenizer,
    question: str,
    existing_lines: Sequence[str],
    rank: int,
    document_text: str,
    prompt_budget: int,
) -> str | None:
    fitted_text, _prompt_token_count = _fit_document_within_prompt_budget_and_count(
        tokenizer=tokenizer,
        question=question,
        existing_lines=existing_lines,
        rank=rank,
        document_text=document_text,
        prompt_budget=prompt_budget,
    )
    return fitted_text


def _fit_document_within_prompt_budget_and_count(
    *,
    tokenizer,
    question: str,
    existing_lines: Sequence[str],
    rank: int,
    document_text: str,
    prompt_budget: int,
) -> tuple[str | None, int | None]:
    candidate_line = f"[{rank}] {document_text}"
    candidate_prompt = _render_context_prompt(question, [*existing_lines, candidate_line])
    candidate_prompt_token_count = _token_count(tokenizer, candidate_prompt, add_special_tokens=True)
    if candidate_prompt_token_count <= prompt_budget:
        return document_text, candidate_prompt_token_count
    words = document_text.split()
    return _longest_word_prefix_within_budget(
        words,
        upper_bound=len(words) - 1,
        count_tokens=lambda candidate: _token_count(
            tokenizer,
            _render_context_prompt(question, [*existing_lines, f"[{rank}] {candidate}"]),
            add_special_tokens=True,
        ),
        token_budget=prompt_budget,
    )


def pack_context_prompt(
    *,
    question: str,
    retrieved_documents: Sequence[Document],
    tokenizer,
    max_input_length: int,
    max_new_tokens: int,
    prompt_top_k: int = DEFAULT_PROMPT_TOP_K,
    max_document_tokens: int = DEFAULT_MAX_DOCUMENT_TOKENS,
) -> dict[str, object]:
    if prompt_top_k <= 0:
        raise ValueError("prompt_top_k must be positive")
    if max_document_tokens <= 0:
        raise ValueError("max_document_tokens must be positive")
    prompt_budget = max_input_length - max_new_tokens
    if prompt_budget <= 0:
        raise ValueError("max_input_length must exceed max_new_tokens")
    clean_question = str(question).strip()
    if not clean_question:
        raise ValueError("question must be non-empty")
    visible_documents: list[Document] = []
    context_lines: list[str] = []
    packed_prompt_token_count: int | None = None
    prompt_truncated = len(retrieved_documents) > prompt_top_k
    for rank, document in enumerate(retrieved_documents[:prompt_top_k], start=1):
        candidate_text, did_trim = _trim_document_to_token_budget(
            document.text,
            tokenizer=tokenizer,
            max_document_tokens=max_document_tokens,
        )
        if not candidate_text:
            prompt_truncated = True
            break
        fitted_text, packed_prompt_token_count = _fit_document_within_prompt_budget_and_count(
            tokenizer=tokenizer,
            question=clean_question,
            existing_lines=context_lines,
            rank=rank,
            document_text=candidate_text,
            prompt_budget=prompt_budget,
        )
        if fitted_text is None:
            prompt_truncated = True
            break
        if fitted_text != candidate_text:
            prompt_truncated = True
        prompt_truncated = prompt_truncated or did_trim
        context_lines.append(f"[{rank}] {fitted_text}")
        visible_documents.append(Document(doc_id=document.doc_id, text=fitted_text))
        if fitted_text != candidate_text:
            break
    if not context_lines:
        context_lines = ["[1] No retrieved evidence."]
        packed_prompt_token_count = None
    prompt = _render_context_prompt(clean_question, context_lines)
    if packed_prompt_token_count is None:
        packed_prompt_token_count = _token_count(tokenizer, prompt, add_special_tokens=True)
    if packed_prompt_token_count > prompt_budget and not visible_documents:
        # Preserve the entire question and instructions when even the optional
        # no-evidence placeholder cannot fit. This remains a RAG prompt.
        prompt = _render_context_prompt(clean_question, [])
        packed_prompt_token_count = _token_count(tokenizer, prompt, add_special_tokens=True)
    if packed_prompt_token_count > prompt_budget:
        raise RuntimeError("packed prompt exceeds generation budget")
    return {
        "prompt": prompt,
        "prompt_visible_doc_ids": [document.doc_id for document in visible_documents],
        "prompt_visible_doc_count": len(visible_documents),
        "prompt_truncated": prompt_truncated,
        "prompt_budget": prompt_budget,
        "visible_documents": visible_documents,
    }


def _contains_alias_hit(documents: Sequence[Document], aliases: Sequence[str]) -> bool:
    normalized_aliases = [normalize_text(alias) for alias in aliases if normalize_text(alias)]
    if not normalized_aliases:
        return False
    combined = normalize_text("\n".join(document.text for document in documents))
    return any(alias in combined for alias in normalized_aliases)


def visible_answer_hit(aliases: Sequence[str], visible_documents: Sequence[Document]) -> float:
    """Answer recall@5 over the packed (visible) evidence."""
    return 1.0 if _contains_alias_hit(visible_documents[:5], tuple(str(alias) for alias in aliases)) else 0.0
