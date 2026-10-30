"""Freeze one validation set per knowledge-QA dataset for coefficient selection.

Validation questions come from splits that the test sets do not use:

    nq        DPR NQ dev             (the test set is DPR NQ test)
    triviaqa  DPR TriviaQA dev       (the test set is DPR TriviaQA test)
    webq      DPR WebQuestions dev   (the test set is DPR WebQuestions test)
    hotpotqa  HotpotQA train         (the test set is the distractor validation split)
    popqa     EntityQuestions dev    (PopQA has a single split, which is all test)

EntityQuestions, like PopQA, consists of template questions about Wikidata
relations, so it serves as PopQA's validation distribution. Rows whose
normalized question appears in any test set are removed, duplicates are
removed, and at most ``--per-dataset`` rows are kept per dataset in a fixed
hash order that does not depend on any model output.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import string
import unicodedata
import zipfile
from pathlib import Path

from qa.data.build_test_sets import _read_dpr_lines, _read_hotpot_rows, normalize_dpr_tsv, normalize_hotpotqa
from qa.data.datasets import _sha256_file, write_frozen_dataset

SCHEMA_VERSION = "resmem-qa-validation-v1"
DATASETS = ("nq", "triviaqa", "webq", "hotpotqa", "popqa")
SELECTION_SALT = "resmem-qa-validation-v1"
SOURCE_URLS = {
    "nq": "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv",
    "triviaqa": "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-dev.qa.csv.gz",
    "webq": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-webquestions-dev.json.gz",
    "hotpotqa": "https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/main/distractor (train-*.parquet)",
    "popqa": "https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip",
}


def normalize_question(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def _aliases(values) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if isinstance(value, str) and value.strip()))


def load_webq_dev(path: Path) -> list[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for ordinal, item in enumerate(payload):
        aliases = _aliases(item.get("answers", []))
        if not aliases:
            continue
        rows.append({
            "stable_id": f"webq:dev:{ordinal:06d}",
            "dataset": "webq",
            "question": str(item["question"]).strip(),
            "answer": aliases[0],
            "aliases": aliases,
            "metadata": {"source_ordinal": ordinal, "source_split": "dev"},
        })
    return rows


def load_entityquestions_dev(path: Path) -> list[dict[str, object]]:
    """Read every ``dev/<relation>.dev.json`` file of the EntityQuestions archive."""
    rows = []
    with zipfile.ZipFile(path) as archive:
        members = sorted(name for name in archive.namelist() if re.search(r"(^|/)dev/[^/]+\.dev\.json$", name))
        if not members:
            raise ValueError("EntityQuestions archive has no dev/*.dev.json files")
        for member in members:
            relation = Path(member).name.split(".", 1)[0]
            for ordinal, item in enumerate(json.loads(archive.read(member))):
                aliases = _aliases(item.get("answers", []))
                if not aliases:
                    continue
                rows.append({
                    # Validation rows for PopQA's coefficient carry PopQA's dataset label.
                    "stable_id": f"entityquestions:dev:{relation}:{ordinal:06d}",
                    "dataset": "popqa",
                    "question": str(item["question"]).strip(),
                    "answer": aliases[0],
                    "aliases": aliases,
                    "metadata": {"source": "entityquestions", "source_split": "dev", "relation": relation, "source_ordinal": ordinal},
                })
    return rows


def load_source(dataset: str, paths: list[Path]) -> list[dict[str, object]]:
    if dataset in {"nq", "triviaqa"}:
        return normalize_dpr_tsv(_read_dpr_lines(paths[0]), dataset=dataset, split="dev")
    if dataset == "webq":
        return load_webq_dev(paths[0])
    if dataset == "hotpotqa":
        source_rows = [row for path in paths for row in _read_hotpot_rows(path)]
        rows = normalize_hotpotqa(source_rows, split="train")
        return [dict(row, stable_id=f"hotpotqa:train:{row['metadata']['source_id']}") for row in rows]
    if dataset == "popqa":
        return load_entityquestions_dev(paths[0])
    raise ValueError(dataset)


def test_questions(test_root: Path) -> set[str]:
    questions = set()
    for dataset in DATASETS:
        with (test_root / f"{dataset}.jsonl").open(encoding="utf-8") as handle:
            questions.update(normalize_question(json.loads(line)["question"]) for line in handle if line.strip())
    return questions


def select_rows(rows: list[dict[str, object]], *, dataset: str, excluded: set[str], count: int) -> tuple[list[dict[str, object]], dict[str, int]]:
    counters = {"source_rows": len(rows), "test_overlap": 0, "duplicate": 0, "empty": 0}
    seen: set[str] = set()
    eligible = []
    for row in rows:
        key = normalize_question(row["question"])
        if not key:
            counters["empty"] += 1
        elif key in excluded:
            counters["test_overlap"] += 1
        elif key in seen:
            counters["duplicate"] += 1
        else:
            seen.add(key)
            rank = hashlib.sha256(f"{SELECTION_SALT}|{dataset}|{key}".encode()).hexdigest()
            eligible.append((rank, str(row["stable_id"]), row))
    eligible.sort(key=lambda item: item[:2])
    counters["eligible"] = len(eligible)
    selected = [row for _, _, row in eligible[:count]]
    counters["selected"] = len(selected)
    return selected, counters


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-root", required=True, type=Path, help="directory with the frozen test <dataset>.jsonl files")
    parser.add_argument("--nq", required=True, type=Path, help="nq-dev.qa.csv")
    parser.add_argument("--triviaqa", required=True, type=Path, help="trivia-dev.qa.csv.gz")
    parser.add_argument("--webq", required=True, type=Path, help="biencoder-webquestions-dev.json.gz")
    parser.add_argument("--hotpotqa", required=True, type=Path, nargs="+", help="HotpotQA distractor train parquet shard(s)")
    parser.add_argument("--popqa", required=True, type=Path, help="EntityQuestions dataset.zip")
    parser.add_argument("--per-dataset", type=int, default=1000)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)

    excluded = test_questions(args.test_root)
    sources = {"nq": [args.nq], "triviaqa": [args.triviaqa], "webq": [args.webq], "hotpotqa": args.hotpotqa, "popqa": [args.popqa]}
    report = {}
    for dataset in DATASETS:
        rows, counters = select_rows(load_source(dataset, sources[dataset]), dataset=dataset, excluded=excluded, count=args.per_dataset)
        source_manifest = {"sources": [{"url": SOURCE_URLS[dataset], "files": [{"name": path.name, "sha256": _sha256_file(path)} for path in sources[dataset]]}],
                           "selection": {"salt": SELECTION_SALT, "rule": "lowest SHA-256(salt|dataset|normalized question)", **counters}}
        report[dataset] = write_frozen_dataset(
            output_path=args.output_root / "frozen" / f"{dataset}.jsonl",
            manifest_path=args.output_root / "manifests" / f"{dataset}.manifest.json",
            rows=rows,
            schema_version=SCHEMA_VERSION,
            split="validation",
            source_manifest=source_manifest,
        )
        report[dataset]["selection"] = counters
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
