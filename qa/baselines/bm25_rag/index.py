"""Build and validate the BM25S index over the passage corpus (English stopwords, bm25s defaults).

    python -m qa.baselines.bm25_rag.index bm25 --corpus CORPUS_DIR/corpus.jsonl --output CORPUS_DIR/bm25s-index
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, Sequence


BM25_SCHEMA_VERSION = "coherent-wiki-bm25s-v1"
BM25_MANIFEST_FILENAME = "bm25_manifest.json"
BM25_REQUIRED_FILES = frozenset(
    {
        "corpus.jsonl",
        "data.csc.index.npy",
        "indices.csc.index.npy",
        "indptr.csc.index.npy",
        "params.index.json",
        "vocab.index.json",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json_object(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _iter_corpus_rows(corpus_path: Path, *, start_row: int = 0) -> Iterator[dict[str, object]]:
    with corpus_path.open("r", encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            if not line.strip():
                raise ValueError(f"corpus row {ordinal} is empty")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"corpus row {ordinal} is invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"corpus row {ordinal} must be an object")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"corpus row {ordinal} missing non-empty text")
            if ordinal >= start_row:
                yield row


def _iter_corpus_texts(corpus_path: Path, *, start_row: int = 0) -> Iterator[str]:
    for row in _iter_corpus_rows(corpus_path, start_row=start_row):
        yield str(row["text"])


def _corpus_identity(corpus_path: Path) -> dict[str, object]:
    corpus_path = Path(corpus_path)
    if not corpus_path.is_file():
        raise FileNotFoundError(corpus_path)
    row_count = sum(1 for _ in _iter_corpus_texts(corpus_path))
    if row_count == 0:
        raise ValueError("corpus contains no rows")
    return {
        "filename": corpus_path.name,
        "row_count": row_count,
        "sha256": _sha256_file(corpus_path),
        "text_field": "text",
    }


def _file_records(root: Path, *, excluded: set[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded or any(part.startswith(".") for part in Path(relative).parts):
            continue
        records.append(
            {
                "bytes": path.stat().st_size,
                "path": relative,
                "sha256": _sha256_file(path),
            }
        )
    return records


def _load_bm25s():
    import bm25s

    return bm25s


def build_bm25_index(
    corpus_path: Path,
    output_root: Path,
    *,
    bm25_module=None,
) -> dict[str, object]:
    """Build the exact ``bm25s.BM25.save(..., corpus=...)`` directory format."""

    corpus_path = Path(corpus_path)
    output_root = Path(output_root)
    corpus = _corpus_identity(corpus_path)
    manifest_path = output_root / BM25_MANIFEST_FILENAME
    if manifest_path.is_file():
        return validate_bm25_index(corpus_path, output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"refusing to replace non-empty unverified BM25 directory: {output_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    try:
        texts = list(_iter_corpus_texts(corpus_path))
        dependency = bm25_module if bm25_module is not None else _load_bm25s()
        tokenized = dependency.tokenize(texts, stopwords="en", show_progress=False)
        retriever = dependency.BM25(corpus=texts)
        retriever.index(tokenized)
        retriever.save(str(staging), corpus=_iter_corpus_rows(corpus_path))
        files = _file_records(staging, excluded={BM25_MANIFEST_FILENAME})
        if not files:
            raise RuntimeError("bm25s did not publish any index files")
        manifest: dict[str, object] = {
            "corpus": corpus,
            "files": files,
            "index_api": "bm25s.BM25.save(corpus=corpus)",
            "schema_version": BM25_SCHEMA_VERSION,
            "tokenization": {"stopwords": "en"},
        }
        _atomic_write_json(staging / BM25_MANIFEST_FILENAME, manifest)
        if output_root.exists():
            output_root.rmdir()
        os.replace(staging, output_root)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def validate_bm25_index(corpus_path: Path, output_root: Path) -> dict[str, object]:
    corpus_path = Path(corpus_path)
    output_root = Path(output_root)
    manifest = _load_json_object(output_root / BM25_MANIFEST_FILENAME)
    if manifest.get("schema_version") != BM25_SCHEMA_VERSION:
        raise ValueError("unsupported BM25 manifest schema_version")
    if manifest.get("corpus") != _corpus_identity(corpus_path):
        raise ValueError("BM25 corpus identity mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("BM25 manifest has no index files")
    recorded_paths = {
        str(entry.get("path"))
        for entry in files
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    missing_required = sorted(BM25_REQUIRED_FILES - recorded_paths)
    if missing_required:
        raise ValueError(f"BM25 manifest missing required files: {missing_required}")
    expected_paths: list[str] = []
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("BM25 manifest has an invalid file record")
        relative = str(entry["path"])
        expected_paths.append(relative)
        path = output_root / relative
        if not path.is_file():
            raise ValueError(f"BM25 index file missing: {relative}")
        if _sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"BM25 index file hash mismatch: {relative}")
        if path.stat().st_size != entry.get("bytes"):
            raise ValueError(f"BM25 index file size mismatch: {relative}")
    actual_paths = [entry["path"] for entry in _file_records(output_root, excluded={BM25_MANIFEST_FILENAME})]
    if expected_paths != actual_paths:
        raise ValueError("BM25 index file set mismatch")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build coherent Wikipedia retrieval assets")
    commands = parser.add_subparsers(dest="command", required=True)

    bm25 = commands.add_parser("bm25", help="build a BM25S index")
    bm25.add_argument("--corpus", type=Path, required=True)
    bm25.add_argument("--output", type=Path, required=True)

    validate_bm25 = commands.add_parser("validate-bm25", help="validate a BM25S asset directory")
    validate_bm25.add_argument("--corpus", type=Path, required=True)
    validate_bm25.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "bm25":
        result = build_bm25_index(args.corpus, args.output)
    elif args.command == "validate-bm25":
        result = validate_bm25_index(args.corpus, args.output)
    else:  # pragma: no cover - argparse enforces a known subcommand
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
