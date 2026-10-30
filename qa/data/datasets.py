from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib import request
from urllib.error import URLError


QUESTION_TYPES = {
    "comparison",
    "inference",
    "compositional",
    "bridge_comparison",
    "bridge-comparison",
}
SCHEMA_VERSION = "resmem-qa-v1"
SAFE_LITERAL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .,'&()/-]*[A-Za-z0-9.)]$|^[A-Za-z0-9]$")


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _required_text(row: Mapping[str, object], field: str, *, where: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} requires non-empty {field}")
    return value.strip()


def _optional_text(row: Mapping[str, object], field: str) -> str | None:
    value = row.get(field)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_aliases(values: Iterable[object]) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, str):
            normalized = value.strip()
        else:
            normalized = ""
        if not normalized or normalized in seen:
            continue
        aliases.append(normalized)
        seen.add(normalized)
    return aliases


def _coerce_int(value: object, *, where: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{where} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        return int(value.strip())
    raise ValueError(f"{where} must be an integer")


def _load_possible_answers(value: object, *, where: str) -> list[str]:
    if isinstance(value, str):
        payload = value.strip()
        if not payload:
            raise ValueError(f"{where} requires non-empty possible_answers")
        parsed = json.loads(payload)
    elif isinstance(value, list):
        parsed = value
    else:
        raise ValueError(f"{where} requires possible_answers")
    aliases = _normalize_aliases(parsed)
    if not aliases:
        raise ValueError(f"{where} requires at least one possible answer")
    return aliases


def normalize_popqa(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for source_ordinal, row in enumerate(rows):
        where = f"PopQA row {source_ordinal}"
        source_id = _required_text(row, "id", where=where)
        question = _required_text(row, "question", where=where)
        subject = _optional_text(row, "subj") or _optional_text(row, "s_wiki_title") or source_id
        prop = _required_text(row, "prop", where=where)
        answer = _required_text(row, "obj", where=where)
        aliases = _load_possible_answers(row.get("possible_answers"), where=where)
        if answer not in aliases:
            aliases = [answer, *aliases]
        normalized.append(
            {
                "stable_id": f"popqa:{source_id}",
                "dataset": "popqa",
                "question": question,
                "answer": answer,
                "aliases": _normalize_aliases(aliases),
                "metadata": {
                    "source_id": source_id,
                    "subject": subject,
                    "property": prop,
                    "object": answer,
                    "subject_popularity": _coerce_int(row.get("s_pop"), where=f"{where} s_pop"),
                    "object_popularity": _coerce_int(row.get("o_pop"), where=f"{where} o_pop"),
                },
            }
        )
    return normalized


def _strip_outer_group(pattern: str) -> str:
    candidate = pattern.strip()
    while candidate.startswith("(?i)"):
        candidate = candidate[4:].strip()
    while candidate.startswith("^") and candidate.endswith("$"):
        candidate = candidate[1:-1].strip()
    if candidate.startswith("(") and candidate.endswith(")"):
        depth = 0
        for index, char in enumerate(candidate):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(candidate) - 1:
                    return candidate
        return candidate[1:-1].strip()
    return candidate


def _split_top_level_alternatives(pattern: str) -> list[str] | None:
    values: list[str] = []
    current: list[str] = []
    depth = 0
    escaped = False
    for char in pattern:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            current.append(char)
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return None
        if char == "|" and depth == 0:
            values.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if depth != 0:
        return None
    values.append("".join(current).strip())
    return values


def _extract_literal_aliases(pattern: str) -> list[str]:
    stripped = _strip_outer_group(pattern)
    alternatives = _split_top_level_alternatives(stripped)
    if not alternatives:
        return []
    aliases: list[str] = []
    for value in alternatives:
        candidate = value.strip()
        if candidate.startswith("?:"):
            candidate = candidate[2:].strip()
        if not candidate or not SAFE_LITERAL_RE.fullmatch(candidate):
            return []
        aliases.append(candidate)
    return _normalize_aliases(aliases)


def normalize_curated_trec(lines: Sequence[str]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for source_ordinal, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            raise ValueError(f"CuratedTREC row {source_ordinal} must have exactly 4 TSV fields")
        source_id, question_type, question, answer_regex = (part.strip() for part in parts)
        if not source_id or not question_type or not question or not answer_regex:
            raise ValueError(
                f"CuratedTREC row {source_ordinal} requires id, question_type, question, and answer regex"
            )
        aliases = _extract_literal_aliases(answer_regex)
        answer = aliases[0] if aliases else answer_regex
        normalized.append(
            {
                "stable_id": f"curatedtrec:{source_id}",
                "dataset": "curatedtrec",
                "question": question,
                "answer": answer,
                "aliases": aliases,
                "metadata": {
                    "source_id": source_id,
                    "question_type": question_type,
                    "answer_regex": answer_regex,
                    "has_literal_aliases": bool(aliases),
                },
            }
        )
    return normalized


def _alias_map_from_payload(id_aliases: object) -> dict[str, list[str]]:
    if isinstance(id_aliases, dict):
        normalized: dict[str, list[str]] = {}
        for key, value in id_aliases.items():
            if isinstance(key, str):
                if isinstance(value, list):
                    normalized[key] = _normalize_aliases(value)
                elif isinstance(value, dict):
                    candidates = []
                    for field in ("aliases", "demonyms", "values"):
                        payload = value.get(field)
                        if isinstance(payload, list):
                            candidates.extend(payload)
                    normalized[key] = _normalize_aliases(candidates)
                elif isinstance(value, str):
                    normalized[key] = _normalize_aliases([value])
        return normalized
    if isinstance(id_aliases, list):
        normalized = {}
        for item in id_aliases:
            if not isinstance(item, dict):
                continue
            qid = item.get("Q_id") or item.get("id")
            if not isinstance(qid, str) or not qid.strip():
                continue
            candidates = []
            for field in ("aliases", "demonyms", "alias", "name"):
                payload = item.get(field)
                if isinstance(payload, list):
                    candidates.extend(payload)
                elif isinstance(payload, str):
                    candidates.append(payload)
            normalized[qid.strip()] = _normalize_aliases(candidates)
        return normalized
    raise ValueError("id_aliases payload must be a dict or list")


def normalize_2wiki(rows: Sequence[Mapping[str, object]], id_aliases: object) -> list[dict[str, object]]:
    alias_map = _alias_map_from_payload(id_aliases)
    normalized: list[dict[str, object]] = []
    for source_ordinal, row in enumerate(rows):
        where = f"2Wiki row {source_ordinal}"
        source_id = _required_text(row, "_id", where=where)
        question = _required_text(row, "question", where=where)
        answer = _required_text(row, "answer", where=where)
        question_type = _required_text(row, "type", where=where).replace("-", "_")
        if question_type not in {item.replace("-", "_") for item in QUESTION_TYPES}:
            raise ValueError(f"{where} has unsupported question type {question_type!r}")
        answer_id = _optional_text(row, "answer_id")
        aliases = alias_map.get(answer_id or "", [])
        if answer not in aliases:
            aliases = [answer, *aliases]
        metadata = {
            "source_id": source_id,
            "question_type": question_type,
        }
        for field in ("answer_id", "supporting_facts", "evidences", "evidences_id", "entity_ids", "context"):
            if field in row:
                metadata[field] = row[field]
        normalized.append(
            {
                "stable_id": f"2wiki:{source_id}",
                "dataset": "2wiki",
                "question": question,
                "answer": answer,
                "aliases": _normalize_aliases(aliases),
                "metadata": metadata,
            }
        )
    return normalized


def validate_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        where = f"row {index}"
        stable_id = _required_text(row, "stable_id", where=where)
        if stable_id in seen:
            raise ValueError(f"duplicate stable_id: {stable_id}")
        seen.add(stable_id)
        dataset = _required_text(row, "dataset", where=where)
        question = _required_text(row, "question", where=where)
        answer = _required_text(row, "answer", where=where)
        aliases_value = row.get("aliases")
        if not isinstance(aliases_value, list):
            raise ValueError(f"{where} aliases must be a list")
        aliases = _normalize_aliases(aliases_value)
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{where} metadata must be an object")
        answer_regex = metadata.get("answer_regex")
        has_answer_contract = bool(aliases) or (isinstance(answer_regex, str) and bool(answer_regex.strip()))
        if not has_answer_contract:
            raise ValueError(f"{where} requires a non-empty answer contract")
        normalized.append(
            {
                "stable_id": stable_id,
                "dataset": dataset,
                "question": question,
                "answer": answer,
                "aliases": aliases,
                "metadata": metadata,
            }
        )
    if not normalized:
        raise ValueError("rows are empty")
    return normalized


def write_frozen_dataset(
    *,
    output_path: Path,
    manifest_path: Path,
    rows: Sequence[Mapping[str, object]],
    schema_version: str,
    split: str,
    source_manifest: Mapping[str, object],
) -> dict[str, object]:
    validated = sorted(validate_rows(rows), key=lambda row: str(row["stable_id"]))
    datasets = {str(row["dataset"]) for row in validated}
    if len(datasets) != 1:
        raise ValueError("write_frozen_dataset expects exactly one dataset per output")
    dataset = next(iter(datasets))
    jsonl_text = "".join(f"{_canonical_json(row)}\n" for row in validated)
    _atomic_write_text(output_path, jsonl_text)
    converted_sha256 = _sha256_bytes(jsonl_text.encode("utf-8"))
    manifest = {
        "schema_version": schema_version,
        "dataset": dataset,
        "split": split,
        "row_count": len(validated),
        "converted_sha256": converted_sha256,
        "sources": list(source_manifest.get("sources", [])),
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    _atomic_write_text(manifest_path, manifest_text)
    return {
        "dataset": dataset,
        "row_count": len(validated),
        "converted_sha256": converted_sha256,
        "manifest_sha256": _sha256_bytes(manifest_text.encode("utf-8")),
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
    }


def _download(url: str, *, timeout: int = 120) -> bytes:
    with request.urlopen(url, timeout=timeout) as response:
        return response.read()


def _download_with_fallback(sources: Sequence[Mapping[str, object]]) -> tuple[bytes, dict[str, object]]:
    failures: list[str] = []
    for source in sources:
        url = str(source["url"])
        try:
            payload = _download(url)
        except URLError as error:
            failures.append(f"{url}: {error}")
            continue
        chosen = dict(source)
        chosen["download_sha256"] = _sha256_bytes(payload)
        return payload, chosen
    raise RuntimeError("all source URLs failed: " + "; ".join(failures))


def _expected_hash_fields_for_source(source: Mapping[str, object]) -> tuple[str, ...]:
    if "archive_sha256" in source or "dev_sha256" in source or "id_aliases_sha256" in source:
        return ("archive_sha256", "dev_sha256", "id_aliases_sha256")
    return ("download_sha256",)


def _validate_expected_artifacts(
    dataset_name: str,
    dataset_config: Mapping[str, object],
    result: Mapping[str, object],
) -> None:
    expected_rows = dataset_config.get("expected_rows")
    if isinstance(expected_rows, int) and result["row_count"] != expected_rows:
        raise ValueError(f"{dataset_name} row count mismatch: {result['row_count']} != {expected_rows}")
    expected_converted_sha256 = dataset_config.get("expected_converted_sha256")
    if isinstance(expected_converted_sha256, str) and expected_converted_sha256:
        if result["converted_sha256"] != expected_converted_sha256:
            raise ValueError(
                f"{dataset_name} converted_sha256 mismatch: {result['converted_sha256']} != {expected_converted_sha256}"
            )
    expected_manifest_sha256 = dataset_config.get("expected_manifest_sha256")
    if isinstance(expected_manifest_sha256, str) and expected_manifest_sha256:
        if result["manifest_sha256"] != expected_manifest_sha256:
            raise ValueError(
                f"{dataset_name} manifest_sha256 mismatch: {result['manifest_sha256']} != {expected_manifest_sha256}"
            )
    expected_sources = dataset_config.get("sources")
    actual_sources = result.get("sources")
    if not isinstance(expected_sources, list) or not expected_sources:
        raise ValueError(f"{dataset_name} approved sources are missing")
    if not isinstance(actual_sources, list) or not actual_sources:
        raise ValueError(f"{dataset_name} selected sources are missing")
    if not actual_sources:
        raise ValueError(f"{dataset_name} sources are empty")
    actual_source = actual_sources[0]
    if not isinstance(actual_source, dict):
        raise ValueError(f"{dataset_name} source manifest entry must be an object")
    actual_name = actual_source.get("name")
    if not isinstance(actual_name, str) or not actual_name:
        raise ValueError(f"{dataset_name} selected source must have a non-empty name")
    approved_source = None
    for candidate in expected_sources:
        if isinstance(candidate, dict) and candidate.get("name") == actual_name:
            approved_source = candidate
            break
    if approved_source is None:
        raise ValueError(f"{dataset_name} selected source name is not approved: {actual_name}")
    for field in _expected_hash_fields_for_source(approved_source):
        expected_value = approved_source.get(field)
        if not isinstance(expected_value, str) or not expected_value:
            raise ValueError(f"{dataset_name} approved source is missing pinned {field}")
        actual_value = actual_source.get(field)
        if actual_value != expected_value:
            raise ValueError(f"{dataset_name} source {field} mismatch: {actual_value} != {expected_value}")


def _reuse_existing_file(path: Path, *, expected_sha256: str | None) -> bytes | None:
    if not path.is_file() or not expected_sha256:
        return None
    payload = path.read_bytes()
    if _sha256_bytes(payload) != expected_sha256:
        return None
    return payload


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_2wiki_alias_payload(path: Path) -> object:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty alias file: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def _extract_2wiki_zip(payload: bytes, *, raw_dir: Path) -> tuple[Path, Path]:
    _atomic_write_bytes(raw_dir / "2wiki_data_ids.zip", payload)
    with zipfile.ZipFile(raw_dir / "2wiki_data_ids.zip") as archive:
        dev_member = None
        alias_member = None
        for name in archive.namelist():
            normalized = name.replace("\\", "/")
            if normalized == "dev.json" or normalized.endswith("/dev.json"):
                dev_member = name
            elif normalized == "id_aliases.json" or normalized.endswith("/id_aliases.json"):
                alias_member = name
        if dev_member is None or alias_member is None:
            raise ValueError("2Wiki archive must contain dev.json and id_aliases.json")
        dev_bytes = archive.read(dev_member)
        alias_bytes = archive.read(alias_member)
    dev_path = raw_dir / "dev.json"
    alias_path = raw_dir / "id_aliases.json"
    _atomic_write_bytes(dev_path, dev_bytes)
    _atomic_write_bytes(alias_path, alias_bytes)
    return dev_path, alias_path


def _load_cached_2wiki_payload(dataset_config: Mapping[str, object], *, raw_dir: Path) -> tuple[bytes, dict[str, object]] | None:
    for source in dataset_config.get("sources", []):
        if not isinstance(source, dict):
            continue
        expected_archive = source.get("archive_sha256")
        cached = _reuse_existing_file(raw_dir / "2wiki_data_ids.zip", expected_sha256=expected_archive if isinstance(expected_archive, str) else None)
        if cached is None:
            continue
        expected_dev = source.get("dev_sha256")
        expected_aliases = source.get("id_aliases_sha256")
        if not isinstance(expected_dev, str) or not isinstance(expected_aliases, str):
            raise ValueError("2wiki approved source is missing pinned extracted hashes")
        if _sha256_file(raw_dir / "dev.json") != expected_dev:
            continue
        if _sha256_file(raw_dir / "id_aliases.json") != expected_aliases:
            continue
        chosen = dict(source)
        return cached, chosen
    return None


def build_dataset(dataset_name: str, *, config: Mapping[str, object], root: Path) -> dict[str, object]:
    raw_dir = root / "raw" / dataset_name
    frozen_dir = root / "frozen"
    manifest_dir = root / "manifests"
    dataset_config = config["datasets"][dataset_name]
    split = str(dataset_config["split"])
    if dataset_name == "popqa":
        chosen_seed = dataset_config["sources"][0]
        expected_download = chosen_seed.get("download_sha256") if isinstance(chosen_seed, dict) else None
        payload = _reuse_existing_file(
            raw_dir / "popQA.tsv",
            expected_sha256=expected_download if isinstance(expected_download, str) else None,
        )
        if payload is None:
            payload, chosen_source = _download_with_fallback(dataset_config["sources"])
        else:
            chosen_source = dict(chosen_seed)
        raw_path = raw_dir / "popQA.tsv"
        _atomic_write_bytes(raw_path, payload)
        rows = list(csv.DictReader(payload.decode("utf-8").splitlines(), delimiter="\t"))
        normalized = normalize_popqa(rows)
        result = write_frozen_dataset(
            output_path=frozen_dir / "popqa.jsonl",
            manifest_path=manifest_dir / "popqa.manifest.json",
            rows=normalized,
            schema_version=SCHEMA_VERSION,
            split=split,
            source_manifest={"sources": [chosen_source]},
        )
    elif dataset_name == "curatedtrec":
        chosen_seed = dataset_config["sources"][0]
        expected_download = chosen_seed.get("download_sha256") if isinstance(chosen_seed, dict) else None
        payload = _reuse_existing_file(
            raw_dir / "curated-test.tsv",
            expected_sha256=expected_download if isinstance(expected_download, str) else None,
        )
        if payload is None:
            payload, chosen_source = _download_with_fallback(dataset_config["sources"])
        else:
            chosen_source = dict(chosen_seed)
        raw_path = raw_dir / "curated-test.tsv"
        _atomic_write_bytes(raw_path, payload)
        normalized = normalize_curated_trec(payload.decode("utf-8").splitlines())
        result = write_frozen_dataset(
            output_path=frozen_dir / "curatedtrec.jsonl",
            manifest_path=manifest_dir / "curatedtrec.manifest.json",
            rows=normalized,
            schema_version=SCHEMA_VERSION,
            split=split,
            source_manifest={"sources": [chosen_source]},
        )
    elif dataset_name == "2wiki":
        cached = _load_cached_2wiki_payload(dataset_config, raw_dir=raw_dir)
        if cached is None:
            payload, chosen_source = _download_with_fallback(dataset_config["sources"])
            dev_path, alias_path = _extract_2wiki_zip(payload, raw_dir=raw_dir)
        else:
            payload, chosen_source = cached
            dev_path = raw_dir / "dev.json"
            alias_path = raw_dir / "id_aliases.json"
        rows = _read_json(dev_path)
        if not isinstance(rows, list):
            raise ValueError("2Wiki dev payload must be a JSON array")
        normalized = normalize_2wiki(rows, _load_2wiki_alias_payload(alias_path))
        chosen_source["archive_sha256"] = _sha256_bytes(payload)
        chosen_source["dev_sha256"] = _sha256_file(dev_path)
        chosen_source["id_aliases_sha256"] = _sha256_file(alias_path)
        result = write_frozen_dataset(
            output_path=frozen_dir / "2wiki.jsonl",
            manifest_path=manifest_dir / "2wiki.manifest.json",
            rows=normalized,
            schema_version=SCHEMA_VERSION,
            split=split,
            source_manifest={"sources": [chosen_source]},
        )
    else:
        raise ValueError(f"unsupported dataset {dataset_name!r}")
    result_with_sources = {**result, "sources": [chosen_source]}
    _validate_expected_artifacts(dataset_name, dataset_config, result_with_sources)
    return result_with_sources


def build_all(*, config_path: Path, data_root: Path) -> dict[str, object]:
    config = _read_json(config_path)
    results = {}
    for dataset_name in ("popqa", "curatedtrec", "2wiki"):
        results[dataset_name] = build_dataset(dataset_name, config=config, root=data_root)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build frozen QA datasets")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("sources.json")),
        help="Path to the dataset source configuration",
    )
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).with_name("data")),
        help="Output root for downloaded and frozen datasets",
    )
    args = parser.parse_args(argv)
    results = build_all(config_path=Path(args.config), data_root=Path(args.data_root))
    print(json.dumps(results, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
