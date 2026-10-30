from __future__ import annotations

"""Freeze the five knowledge-QA test sets (38,627 questions).

NQ, TriviaQA and WebQuestions use the complete DPR test splits, HotpotQA the
complete distractor validation split, and PopQA every row of the official TSV.
"""
import argparse
import ast
import csv
import gzip
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from qa.data.datasets import _sha256_file, normalize_popqa, write_frozen_dataset


SCHEMA_VERSION = "resmem-qa-test-v1"
DPR_URLS = {
    "nq": "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv",
    "triviaqa": "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz",
    "webq": "https://dl.fbaipublicfiles.com/dpr/data/retriever/webquestions-test.qa.csv",
}
HOTPOT_HF_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet"
)
HOTPOT_OFFICIAL_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
POPQA_URL = "https://raw.githubusercontent.com/AlexTMallen/adaptive-retrieval/main/data/popQA.tsv"
EXPECTED_ROWS = {
    "nq": 3610,
    "triviaqa": 11313,
    "webq": 2032,
    "hotpotqa": 7405,
    "popqa": 14267,
}
EXPECTED_SOURCE_SHA256 = {
    "nq": "480cfe1d5b81bedd879ffe7c6a9ec294d317811e71c44daabfe242ca9d719d46",
    "triviaqa": "e8adaadb1e5b4f010b17b34a47e2ccac5dc6e4a863588642d5d292c682f1bc1f",
    "webq": "0864b9498077ba9b74ca2460c37954a2b4ffe4179e51ae2a354593fd3d21ae41",
    "hotpotqa": "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6",
    "popqa": "9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b",
}


def _verify_full_source(dataset: str, *, row_count: int, source_sha256: str) -> None:
    expected_rows = EXPECTED_ROWS[dataset]
    if row_count != expected_rows:
        raise ValueError(f"{dataset} row count mismatch: {row_count} != {expected_rows}")
    expected_sha256 = EXPECTED_SOURCE_SHA256[dataset]
    if source_sha256 != expected_sha256:
        raise ValueError(f"{dataset} source SHA-256 mismatch: {source_sha256} != {expected_sha256}")


def _aliases(values: Iterable[object], *, where: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{where} answers must be non-empty strings")
        normalized = value.strip()
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    if not result:
        raise ValueError(f"{where} requires at least one answer")
    return result


def normalize_dpr_tsv(
    lines: Iterable[str],
    *,
    dataset: str,
    split: str,
) -> list[dict[str, object]]:
    """Normalize every row of a DPR QA TSV; this function never samples or caps."""
    rows: list[dict[str, object]] = []
    for source_ordinal, parts in enumerate(csv.reader(lines, delimiter="\t")):
        if not parts or not any(part.strip() for part in parts):
            continue
        where = f"{dataset} row {source_ordinal}"
        if len(parts) != 2 or not parts[0].strip():
            raise ValueError(f"{where} must contain a question and answer list")
        try:
            answer_payload = ast.literal_eval(parts[1].strip())
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"{where} has an invalid answer list") from error
        if not isinstance(answer_payload, (list, tuple)):
            raise ValueError(f"{where} answers must be a list")
        aliases = _aliases(answer_payload, where=where)
        rows.append(
            {
                "stable_id": f"{dataset}:{split}:{source_ordinal:06d}",
                "dataset": dataset,
                "question": parts[0].strip(),
                "answer": aliases[0],
                "aliases": aliases,
                "metadata": {
                    "source_ordinal": source_ordinal,
                    "source_split": split,
                },
            }
        )
    if not rows:
        raise ValueError(f"{dataset} source is empty")
    return rows


def normalize_hotpotqa(
    source_rows: Iterable[Mapping[str, object]],
    *,
    split: str,
) -> list[dict[str, object]]:
    """Normalize the complete answer-bearing HotpotQA split without context fields."""
    rows: list[dict[str, object]] = []
    for source_ordinal, source in enumerate(source_rows):
        where = f"hotpotqa row {source_ordinal}"
        source_id_value = source.get("id", source.get("_id"))
        question_value = source.get("question")
        answer_value = source.get("answer")
        if not isinstance(source_id_value, str) or not source_id_value.strip():
            raise ValueError(f"{where} requires a source id")
        if not isinstance(question_value, str) or not question_value.strip():
            raise ValueError(f"{where} requires a question")
        aliases = _aliases([answer_value], where=where)
        source_id = source_id_value.strip()
        metadata: dict[str, object] = {
            "source_id": source_id,
            "source_split": split,
        }
        question_type = source.get("type")
        if isinstance(question_type, str) and question_type.strip():
            metadata["question_type"] = question_type.strip()
        level = source.get("level")
        if isinstance(level, str) and level.strip():
            metadata["level"] = level.strip()
        rows.append(
            {
                "stable_id": f"hotpotqa:{source_id}",
                "dataset": "hotpotqa",
                "question": question_value.strip(),
                "answer": aliases[0],
                "aliases": aliases,
                "metadata": metadata,
            }
        )
    if not rows:
        raise ValueError("hotpotqa source is empty")
    return rows


def _read_dpr_lines(path: Path) -> list[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return list(handle)
    return path.read_text(encoding="utf-8").splitlines()


def _read_hotpot_rows(path: Path) -> list[dict[str, object]]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("reading a HotpotQA parquet source requires pyarrow") from error
        return pq.read_table(path).to_pylist()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError("HotpotQA source must be a JSON array of objects")
    return payload


def build_original_full_datasets(
    *,
    nq_path: Path,
    triviaqa_path: Path,
    webq_path: Path,
    hotpotqa_path: Path,
    popqa_path: Path,
    data_root: Path,
) -> dict[str, object]:
    """Freeze the complete public evaluation splits and record source identities."""
    inputs = {
        "nq": (nq_path, "test"),
        "triviaqa": (triviaqa_path, "test"),
        "webq": (webq_path, "test"),
    }
    results: dict[str, object] = {}
    for dataset, (source_path, split) in inputs.items():
        rows = normalize_dpr_tsv(_read_dpr_lines(source_path), dataset=dataset, split=split)
        source_sha256 = _sha256_file(source_path)
        _verify_full_source(dataset, row_count=len(rows), source_sha256=source_sha256)
        source = {
            "name": f"dpr-{dataset}-test",
            "url": DPR_URLS[dataset],
            "download_sha256": source_sha256,
            "format": "dpr-qa-tsv",
        }
        results[dataset] = write_frozen_dataset(
            output_path=data_root / "frozen" / f"{dataset}.jsonl",
            manifest_path=data_root / "manifests" / f"{dataset}.manifest.json",
            rows=rows,
            schema_version=SCHEMA_VERSION,
            split=split,
            source_manifest={"sources": [source]},
        )

    hotpot_rows = normalize_hotpotqa(_read_hotpot_rows(hotpotqa_path), split="validation")
    hotpot_sha256 = _sha256_file(hotpotqa_path)
    _verify_full_source("hotpotqa", row_count=len(hotpot_rows), source_sha256=hotpot_sha256)
    hotpot_source = {
        "name": "hotpotqa-hf-official-mirror-distractor-validation",
        "url": HOTPOT_HF_URL,
        "official_url": HOTPOT_OFFICIAL_URL,
        "download_sha256": hotpot_sha256,
        "format": hotpotqa_path.suffix.lstrip("."),
        "source_revision": "1908d6afbbead072334abe2965f91bd2709910ab",
    }
    results["hotpotqa"] = write_frozen_dataset(
        output_path=data_root / "frozen" / "hotpotqa.jsonl",
        manifest_path=data_root / "manifests" / "hotpotqa.manifest.json",
        rows=hotpot_rows,
        schema_version=SCHEMA_VERSION,
        split="validation",
        source_manifest={"sources": [hotpot_source]},
    )

    popqa_rows = normalize_popqa(list(csv.DictReader(popqa_path.read_text(encoding="utf-8").splitlines(), delimiter="\t")))
    popqa_sha256 = _sha256_file(popqa_path)
    _verify_full_source("popqa", row_count=len(popqa_rows), source_sha256=popqa_sha256)
    results["popqa"] = write_frozen_dataset(
        output_path=data_root / "frozen" / "popqa.jsonl",
        manifest_path=data_root / "manifests" / "popqa.manifest.json",
        rows=popqa_rows,
        schema_version=SCHEMA_VERSION,
        split="test",
        source_manifest={"sources": [{"name": "popqa-official", "url": POPQA_URL, "download_sha256": popqa_sha256, "format": "tsv"}]},
    )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the five knowledge-QA test sets.")
    parser.add_argument("--nq", required=True, type=Path)
    parser.add_argument("--triviaqa", required=True, type=Path)
    parser.add_argument("--webq", required=True, type=Path)
    parser.add_argument("--hotpotqa", required=True, type=Path)
    parser.add_argument("--popqa", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args(argv)
    results = build_original_full_datasets(
        nq_path=args.nq,
        triviaqa_path=args.triviaqa,
        webq_path=args.webq,
        hotpotqa_path=args.hotpotqa,
        popqa_path=args.popqa,
        data_root=args.data_root,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
