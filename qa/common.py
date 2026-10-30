"""File layout and helpers shared by the QA scripts (no GPU dependencies)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

DATASETS = ("nq", "webq", "triviaqa", "popqa", "hotpotqa")


def load_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(tmp, path)


def coefficient_key(value: float) -> str:
    return format(float(value), ".8g")


def cell_path(output: Path, dataset: str, coefficient: float, kind: str) -> Path:
    """``kind`` is ``predictions`` (generated answers) or ``nll`` (reference-answer token NLL)."""
    return Path(output) / dataset / f"{coefficient_key(coefficient)}.{kind}.jsonl"


def perplexity(records: list[dict]) -> float:
    """Dataset PPL: exp of the token-weighted mean NLL."""
    tokens = sum(r["token_count"] for r in records)
    return math.exp(math.fsum(v for r in records for v in r["token_nll"]) / tokens)
