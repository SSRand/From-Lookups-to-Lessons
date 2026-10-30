from __future__ import annotations

import math
import random
import re
import string
from collections import defaultdict
from copy import deepcopy
from typing import Iterable, Mapping, Sequence


ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
WHITESPACE_RE = re.compile(r"\s+")
FOLLOWUP_QUESTION_RE = re.compile(r"\s*[;,:-]?\s*Question\s*:", flags=re.IGNORECASE)
BINARY_PREFIX_RE = re.compile(r"^(yes|no)\b", flags=re.IGNORECASE)


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.translate(str.maketrans("", "", string.punctuation))
    lowered = ARTICLES_RE.sub(" ", lowered)
    return WHITESPACE_RE.sub(" ", lowered).strip()


def sanitize_prediction_text(text: str) -> str:
    stripped = str(text).strip()
    if not stripped:
        return ""
    line = stripped.splitlines()[0].strip()
    line = FOLLOWUP_QUESTION_RE.split(line, maxsplit=1)[0].strip()
    return re.sub(r"\s*;\s*$", "", line).strip()


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _aggregate_metric_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "exact_match": 0.0,
            "token_f1": 0.0,
            "answer_containment": 0.0,
        }
    return {
        "exact_match": _mean([float(row["exact_match"]) for row in rows]),
        "token_f1": _mean([float(row["token_f1"]) for row in rows]),
        "answer_containment": _mean([float(row["answer_containment"]) for row in rows]),
    }


def _aggregate_slice_metric_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    metrics = _aggregate_metric_rows(rows)  # type: ignore[arg-type]
    return {"example_count": len(rows), **metrics}


def _aggregate_curated_slice_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    literal_rows = [row for row in rows if bool(row.get("has_literal_aliases"))]
    result: dict[str, float | int] = {
        "example_count": len(rows),
        "regex_em": _mean([float(row["regex_em"]) for row in rows]),
        "literal_count": len(literal_rows),
    }
    if literal_rows:
        literal_metrics = _aggregate_metric_rows(literal_rows)  # type: ignore[arg-type]
        result.update(
            {
                "literal_exact_match": literal_metrics["exact_match"],
                "literal_token_f1": literal_metrics["token_f1"],
                "literal_answer_containment": literal_metrics["answer_containment"],
            }
        )
    return result


def _aggregate_binary_slice_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    metrics = _aggregate_metric_rows(rows)  # type: ignore[arg-type]
    return {
        "example_count": len(rows),
        "yes_count": sum(1 for row in rows if row["binary_gold"] == "yes"),
        "no_count": sum(1 for row in rows if row["binary_gold"] == "no"),
        **metrics,
        "binary_first_token_accuracy": _mean([float(row["binary_first_token_accuracy"]) for row in rows]),
        "binary_output_coverage": _mean([float(row["binary_output_coverage"]) for row in rows]),
        "binary_overgeneration_rate": _mean([float(row["binary_overgeneration_rate"]) for row in rows]),
    }


def _token_f1_single(prediction: str, answer: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    answer_tokens = normalize_text(answer).split()
    if not prediction_tokens and not answer_tokens:
        return 1.0
    if not prediction_tokens or not answer_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in answer_tokens:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in prediction_tokens:
        if counts.get(token, 0) > 0:
            overlap += 1
            counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(answer_tokens)
    return 2.0 * precision * recall / (precision + recall)


def normalized_metrics(prediction: str, aliases: Sequence[str]) -> dict[str, float]:
    clean_prediction = sanitize_prediction_text(prediction)
    normalized_prediction = normalize_text(clean_prediction)
    if not aliases:
        return {
            "exact_match": 0.0,
            "token_f1": 0.0,
            "answer_containment": 0.0,
        }
    best_exact = 0.0
    best_f1 = 0.0
    best_containment = 0.0
    for alias in aliases:
        normalized_alias = normalize_text(alias)
        if not normalized_alias:
            continue
        best_exact = max(best_exact, 1.0 if normalized_prediction == normalized_alias else 0.0)
        best_f1 = max(best_f1, _token_f1_single(clean_prediction, alias))
        best_containment = max(
            best_containment,
            1.0 if normalized_prediction and normalized_alias in normalized_prediction else 0.0,
        )
    return {
        "exact_match": best_exact,
        "token_f1": best_f1,
        "answer_containment": best_containment,
    }


def regex_match(prediction: str, pattern: str) -> bool:
    return re.search(pattern, prediction, flags=re.IGNORECASE) is not None


def _binary_gold_label(row: Mapping[str, object]) -> str | None:
    answer = normalize_text(str(row.get("answer", "")))
    if answer in {"yes", "no"}:
        return answer
    return None


def _binary_prediction_record(prediction: str, gold: str, exact_match: float) -> dict[str, float | str]:
    normalized_prediction = normalize_text(prediction)
    match = BINARY_PREFIX_RE.match(normalized_prediction)
    predicted = match.group(1).lower() if match else None
    covered = 1.0 if predicted is not None else 0.0
    correct = 1.0 if predicted == gold else 0.0
    return {
        "binary_gold": gold,
        "binary_prediction": predicted or "",
        "binary_first_token_accuracy": correct,
        "binary_output_coverage": covered,
        "binary_overgeneration_rate": 1.0 if correct and not float(exact_match) else 0.0,
    }


def bootstrap_interval(
    values: Sequence[float],
    *,
    seed: int = 20260903,
    rounds: int = 1000,
    confidence: float = 0.95,
) -> dict[str, float]:
    if not values:
        raise ValueError("values must be non-empty")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    generator = random.Random(seed)
    samples: list[float] = []
    for _ in range(rounds):
        resample = [float(generator.choice(values)) for _ in range(len(values))]
        samples.append(_mean(resample))
    ordered = sorted(samples)
    lower_index = max(0, int(((1.0 - confidence) / 2.0) * rounds))
    upper_index = min(rounds - 1, int((1.0 - (1.0 - confidence) / 2.0) * rounds) - 1)
    return {
        "mean": _mean([float(value) for value in values]),
        "low": ordered[lower_index],
        "high": ordered[upper_index],
    }


def summarize_predictions(
    rows: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    *,
    protocol: str,
) -> dict[str, object]:
    if protocol not in {"closed_book", "rag"}:
        raise ValueError("protocol must be closed_book or rag")
    if not rows:
        raise ValueError("rows must be non-empty")
    dataset = str(rows[0]["dataset"])
    reference_by_id: dict[str, Mapping[str, object]] = {}
    for row in rows:
        stable_id = str(row["stable_id"])
        row_dataset = str(row["dataset"])
        if row_dataset != dataset:
            raise ValueError("rows must all belong to one dataset")
        if stable_id in reference_by_id:
            raise ValueError(f"duplicate reference stable_id: {stable_id}")
        reference_by_id[stable_id] = row
    prediction_by_id: dict[str, Mapping[str, object]] = {}
    for row in predictions:
        stable_id = str(row["stable_id"])
        if stable_id in prediction_by_id:
            raise ValueError(f"duplicate prediction stable_id: {stable_id}")
        prediction_by_id[stable_id] = row
        if protocol != "rag" and "retrieval" in row:
            raise ValueError("retrieval metrics require rag protocol")

    reference_ids = list(reference_by_id)
    prediction_ids = list(prediction_by_id)
    missing_predictions = [stable_id for stable_id in reference_ids if stable_id not in prediction_by_id]
    extra_predictions = [stable_id for stable_id in prediction_ids if stable_id not in reference_by_id]

    scored_rows: list[dict[str, object]] = []
    literal_rows: list[dict[str, object]] = []
    regex_values: list[float] = []
    relation_groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    type_groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    curated_type_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    curated_contract_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    retrieval_groups: dict[str, list[float]] = defaultdict(list)
    binary_rows: list[dict[str, object]] = []

    for stable_id in reference_ids:
        prediction_row = prediction_by_id.get(stable_id)
        if prediction_row is None:
            continue
        prediction_text = sanitize_prediction_text(str(prediction_row.get("prediction", "")))
        aliases = tuple(str(alias) for alias in row_aliases(reference_by_id[stable_id]))
        metrics = normalized_metrics(prediction_text, aliases)
        scored_rows.append({"stable_id": stable_id, **metrics})
        binary_gold = _binary_gold_label(reference_by_id[stable_id])
        if binary_gold is not None:
            binary_rows.append(
                {
                    **metrics,
                    **_binary_prediction_record(
                        prediction_text,
                        binary_gold,
                        float(metrics["exact_match"]),
                    ),
                }
            )

        metadata = reference_by_id[stable_id].get("metadata")
        metadata_dict = metadata if isinstance(metadata, Mapping) else {}
        if dataset == "curatedtrec":
            pattern = str(metadata_dict.get("answer_regex", ""))
            regex_value = 1.0 if pattern and regex_match(prediction_text, pattern) else 0.0
            regex_values.append(regex_value)
            has_literal_aliases = bool(metadata_dict.get("has_literal_aliases")) and bool(aliases)
            if has_literal_aliases:
                literal_rows.append(metrics)
            curated_slice_row: dict[str, object] = {
                **metrics,
                "regex_em": regex_value,
                "has_literal_aliases": has_literal_aliases,
            }
            question_type = str(metadata_dict.get("question_type", "")).strip()
            if question_type:
                curated_type_groups[question_type].append(curated_slice_row)
            contract_type = "literal_alias" if has_literal_aliases else "regex_only"
            curated_contract_groups[contract_type].append(curated_slice_row)
        elif dataset == "popqa":
            relation = str(metadata_dict.get("property", ""))
            if relation:
                relation_groups[relation].append(metrics)
        elif dataset == "2wiki":
            question_type = str(metadata_dict.get("question_type", ""))
            if question_type:
                type_groups[question_type].append(metrics)

        if protocol == "rag":
            retrieval = prediction_row.get("retrieval")
            if isinstance(retrieval, Mapping):
                for name, value in retrieval.items():
                    retrieval_groups[str(name)].append(float(value))

    metrics = _aggregate_metric_rows(scored_rows)
    coverage: dict[str, object] = {
        "exact": {
            "predictions": len(predictions),
            "references": len(rows),
            "missing_predictions": missing_predictions,
            "extra_predictions": extra_predictions,
        }
    }
    confidence_intervals: dict[str, dict[str, float]] = {}
    slices: dict[str, object] = {}

    if dataset == "curatedtrec":
        metrics["regex_em"] = _mean(regex_values)
        coverage["regex_em"] = _coverage(scored=len(regex_values), total=len(rows))
        literal_total = sum(
            1
            for row in rows
            if bool((row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}).get("has_literal_aliases"))
        )
        coverage["literal_alias_metrics"] = _coverage(scored=len(literal_rows), total=len(rows))
        if literal_rows:
            literal_metrics = _aggregate_metric_rows(literal_rows)
            metrics["literal_exact_match"] = literal_metrics["exact_match"]
            metrics["literal_token_f1"] = literal_metrics["token_f1"]
            metrics["literal_answer_containment"] = literal_metrics["answer_containment"]
        else:
            metrics["literal_exact_match"] = 0.0
            metrics["literal_token_f1"] = 0.0
            metrics["literal_answer_containment"] = 0.0
        if regex_values:
            confidence_intervals["regex_em"] = bootstrap_interval(regex_values)
        if literal_total:
            confidence_intervals["literal_exact_match"] = bootstrap_interval(
                [float(row["exact_match"]) for row in literal_rows] or [0.0]
            )
        slices["question_type"] = {
            name: _aggregate_curated_slice_rows(group_rows)
            for name, group_rows in sorted(curated_type_groups.items())
        }
        slices["answer_contract"] = {
            name: _aggregate_curated_slice_rows(group_rows)
            for name, group_rows in sorted(curated_contract_groups.items())
        }
    elif dataset == "popqa":
        slices["subject_popularity_quartiles"] = _popqa_quartiles(rows, scored_rows)
        slices["relation_type"] = {
            name: _aggregate_slice_metric_rows(group_rows) for name, group_rows in sorted(relation_groups.items())
        }
    elif dataset == "2wiki":
        slices["question_type"] = {
            name: _aggregate_slice_metric_rows(group_rows) for name, group_rows in sorted(type_groups.items())
        }

    if binary_rows:
        coverage["binary_gold_subset"] = _coverage(scored=len(binary_rows), total=len(rows))
        slices["binary_gold_answer"] = {"yes_no": _aggregate_binary_slice_rows(binary_rows)}

    if protocol == "rag" and retrieval_groups:
        for name, values in sorted(retrieval_groups.items()):
            metrics[name] = _mean(values)

    return {
        "protocol": protocol,
        "dataset": dataset,
        "example_count": len(rows),
        "scored_count": len(scored_rows),
        "metrics": metrics,
        "coverage": coverage,
        "slices": slices,
        "confidence_intervals": confidence_intervals,
    }


def row_aliases(row: Mapping[str, object]) -> Sequence[str]:
    aliases = row.get("aliases")
    if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)):
        return [str(alias) for alias in aliases]
    return []


def _coverage(*, scored: int, total: int) -> dict[str, float | int]:
    return {
        "scored": scored,
        "total": total,
        "fraction": (scored / total) if total else 0.0,
    }


def _popqa_quartiles(
    rows: Sequence[Mapping[str, object]],
    scored_rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float | int]]:
    scored_by_id = {str(row["stable_id"]): row for row in scored_rows}
    ordered = sorted(
        (
            (
                int(((row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}) or {}).get("subject_popularity", 0)),
                str(row["stable_id"]),
            )
            for row in rows
        ),
        key=lambda item: (item[0], item[1]),
    )
    quartiles: dict[str, list[dict[str, float]]] = {"q1": [], "q2": [], "q3": [], "q4": []}
    total = len(ordered)
    for index, (_, stable_id) in enumerate(ordered):
        quartile_index = min(3, math.floor(index * 4 / total))
        quartile_name = f"q{quartile_index + 1}"
        scored = scored_by_id.get(stable_id)
        if scored is not None:
            quartiles[quartile_name].append(
                {
                    "exact_match": float(scored["exact_match"]),
                    "token_f1": float(scored["token_f1"]),
                    "answer_containment": float(scored["answer_containment"]),
                }
            )
    return {name: _aggregate_slice_metric_rows(values) for name, values in quartiles.items()}
