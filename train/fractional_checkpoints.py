"""Checkpoint schedule, per-rank data cursors, cache identity, and resume-contract helpers for train.pretrain."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

DEFAULT_FRACTIONAL_POINTS = (1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)


def default_fractional_points() -> tuple[float, ...]:
    return DEFAULT_FRACTIONAL_POINTS


def normalize_fractional_points(points: tuple[float, ...]) -> tuple[float, ...]:
    if not points:
        return ()
    return tuple(sorted({float(item) for item in points}))


def parse_fractional_points(raw: str) -> tuple[float, ...]:
    text = raw.strip()
    if not text:
        return ()
    values: list[float] = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"fractional checkpoint must satisfy 0 < x <= 1, got {item}")
        values.append(value)
    return normalize_fractional_points(tuple(values))


def milestone_label(fraction: float) -> str:
    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError(f"fractional checkpoint label requires 0 < x < 1, got {fraction}")
    rendered = f"{fraction:.7f}".rstrip("0").rstrip(".")
    return "pass" + rendered.replace(".", "p")


def checkpoint_label(*, epoch: int, epoch_fraction: float, epoch_complete: bool) -> str:
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if epoch_complete:
        return f"ep{epoch + 1}"
    if math.isclose(epoch_fraction, 0.0):
        return "pass0" if epoch == 0 else f"ep{epoch}"
    if not 0.0 < epoch_fraction < 1.0:
        raise ValueError(
            f"fractional checkpoint requires 0 <= epoch_fraction < 1 unless epoch_complete, got {epoch_fraction}"
        )
    return milestone_label(epoch_fraction)


def build_fractional_milestones(
    *,
    cap_steps_per_epoch: int,
    fractions: tuple[float, ...],
) -> list[dict[str, float | int | str]]:
    if cap_steps_per_epoch <= 0:
        raise ValueError("cap_steps_per_epoch must be positive")
    dedup_by_step: dict[int, float] = {}
    for fraction in fractions:
        if math.isclose(fraction, 1.0):
            continue
        step = max(1, math.ceil(cap_steps_per_epoch * fraction))
        dedup_by_step[step] = max(fraction, dedup_by_step.get(step, 0.0))
    milestones = []
    for step in sorted(dedup_by_step):
        fraction = dedup_by_step[step]
        milestones.append(
            {
                "fraction": fraction,
                "label": milestone_label(fraction),
                "step": step,
            }
        )
    return milestones


def rank_cursor(
    *,
    file_index: int,
    span_batch_start: int,
    token_chunk_offset: int,
    cache_row_pointer: int,
    rank_seen_tokens: int,
) -> dict[str, int]:
    return {
        "file_index": int(file_index),
        "span_batch_start": int(span_batch_start),
        "token_chunk_offset": int(token_chunk_offset),
        "cache_row_pointer": int(cache_row_pointer),
        "rank_seen_tokens": int(rank_seen_tokens),
    }


def resume_epoch(*, epoch: int, epoch_complete: bool) -> int:
    return int(epoch) + (1 if epoch_complete else 0)


def normalize_resume_position(*, global_progress: dict[str, object], cursor: dict[str, int]) -> dict[str, object]:
    epoch_complete = bool(global_progress["epoch_complete"])
    normalized_cursor = dict(cursor)
    if epoch_complete:
        normalized_cursor = rank_cursor(
            file_index=0,
            span_batch_start=0,
            token_chunk_offset=0,
            cache_row_pointer=0,
            rank_seen_tokens=int(cursor["rank_seen_tokens"]),
        )
    return {
        "start_epoch": resume_epoch(epoch=int(global_progress["epoch"]), epoch_complete=epoch_complete),
        "epoch_step": 0 if epoch_complete else int(global_progress["epoch_step"]),
        "cursor": normalized_cursor,
    }


def advance_rank_cursor(
    *,
    file_index: int,
    span_batch_start: int,
    batch_base_row_pointer: int,
    next_batch_row_pointer: int,
    chunk_end: int,
    batch_token_count: int,
    next_span_batch_start: int,
    has_more_batches: bool,
    has_more_files: bool,
    rank_seen_tokens: int,
) -> dict[str, int]:
    if chunk_end < batch_token_count:
        return rank_cursor(
            file_index=file_index,
            span_batch_start=span_batch_start,
            token_chunk_offset=chunk_end,
            cache_row_pointer=batch_base_row_pointer,
            rank_seen_tokens=rank_seen_tokens,
        )
    if has_more_batches:
        return rank_cursor(
            file_index=file_index,
            span_batch_start=next_span_batch_start,
            token_chunk_offset=0,
            cache_row_pointer=next_batch_row_pointer,
            rank_seen_tokens=rank_seen_tokens,
        )
    if has_more_files:
        return rank_cursor(
            file_index=file_index + 1,
            span_batch_start=0,
            token_chunk_offset=0,
            cache_row_pointer=0,
            rank_seen_tokens=rank_seen_tokens,
        )
    return rank_cursor(
        file_index=file_index,
        span_batch_start=next_span_batch_start,
        token_chunk_offset=0,
        cache_row_pointer=next_batch_row_pointer,
        rank_seen_tokens=rank_seen_tokens,
    )


def resume_cache_rows(cursor: dict[str, int], batch_rows: list[int]) -> list[int]:
    if batch_rows and int(cursor["cache_row_pointer"]) != int(batch_rows[0]):
        raise ValueError("cache_row_pointer must point at the batch base row")
    return list(batch_rows[int(cursor["token_chunk_offset"]):])


def build_synchronized_step_plan(*, per_rank_segments: list[list[int]], accum_pos: int) -> list[list[dict[str, object]]]:
    if accum_pos <= 0:
        raise ValueError("accum_pos must be positive")
    per_rank_totals = [sum(segments) for segments in per_rank_segments]
    step_count = min(total // accum_pos for total in per_rank_totals)
    rank_indices = [0 for _ in per_rank_segments]
    rank_offsets = [0 for _ in per_rank_segments]
    plan: list[list[dict[str, object]]] = []
    for _ in range(step_count):
        step_entries: list[dict[str, object]] = []
        for rank_id, segments in enumerate(per_rank_segments):
            remaining = accum_pos
            slices = []
            while remaining > 0:
                segment_index = rank_indices[rank_id]
                start = rank_offsets[rank_id]
                available = segments[segment_index] - start
                take = min(remaining, available)
                slices.append(
                    {
                        "segment_index": segment_index,
                        "start": start,
                        "end": start + take,
                    }
                )
                remaining -= take
                rank_offsets[rank_id] += take
                if rank_offsets[rank_id] == segments[segment_index]:
                    rank_indices[rank_id] += 1
                    rank_offsets[rank_id] = 0
            step_entries.append(
                {
                    "rank": rank_id,
                    "token_count": accum_pos,
                    "slices": slices,
                }
            )
        plan.append(step_entries)
    return plan


def cumulative_wall_seconds(*, previous_wall_seconds: float, current_run_seconds: float) -> float:
    return float(previous_wall_seconds) + float(current_run_seconds)


def cumulative_gpu_hours(*, wall_seconds: float, world_size: int) -> float:
    return float(wall_seconds) * int(world_size) / 3600.0


def load_cache_manifest_identity(*, cache_dir: str | Path, cache_manifest_path: str | Path | None = None) -> dict[str, object]:
    resolved_cache_dir = Path(cache_dir).resolve(strict=False)
    if not cache_manifest_path:
        from train.cache_manifest import build_cache_manifest

        payload = build_cache_manifest(resolved_cache_dir, with_sha256=False)
        return _cache_identity_from_payload(
            payload=payload,
            resolved_cache_dir=resolved_cache_dir,
            manifest_path=None,
            manifest_sha256=None,
        )
    resolved_manifest_path = Path(cache_manifest_path).resolve(strict=False)
    payload = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    return _cache_identity_from_payload(
        payload=payload,
        resolved_cache_dir=resolved_cache_dir,
        manifest_path=str(resolved_manifest_path),
        manifest_sha256=_sha256_path(resolved_manifest_path),
    )


def _cache_identity_from_payload(
    *,
    payload: dict[str, object],
    resolved_cache_dir: Path,
    manifest_path: str | None,
    manifest_sha256: str | None,
) -> dict[str, object]:
    resolved_manifest_path = manifest_path or "<generated from cache_root>"
    if payload.get("status") != "complete":
        raise ValueError(f"cache manifest must be complete: {resolved_manifest_path}")
    manifest_cache_root = Path(payload.get("cache_root", "")).resolve(strict=False)
    if manifest_cache_root != resolved_cache_dir:
        raise ValueError(f"cache_root mismatch: expected {resolved_cache_dir}, got {manifest_cache_root}")
    builder_manifest = payload.get("builder_manifest")
    if builder_manifest is not None:
        if not isinstance(builder_manifest, dict) or not builder_manifest.get("path") or not builder_manifest.get("sha256"):
            raise ValueError("builder_manifest requires path and sha256")
        builder_manifest = {
            "path": str(builder_manifest["path"]),
            "sha256": str(builder_manifest["sha256"]),
        }
    entries = payload.get("ordered_cache_entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("cache manifest requires non-empty ordered_cache_entries")
    resolved_files: list[str] = []
    normalized_entries: list[dict[str, object]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"ordered_cache_entries[{index}] must be an object")
        required = ("relative_path", "size", "mtime_ns", "sha256", "token_count")
        missing = [key for key in required if key not in entry]
        if missing:
            raise ValueError(f"ordered_cache_entries[{index}] missing required keys: {','.join(missing)}")
        relative_path = Path(str(entry["relative_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"ordered_cache_entries[{index}] has invalid relative_path")
        file_path = (resolved_cache_dir / relative_path).resolve(strict=False)
        try:
            file_path.relative_to(resolved_cache_dir)
        except ValueError as exc:
            raise ValueError(f"ordered_cache_entries[{index}] escapes cache_root") from exc
        if not file_path.exists():
            raise ValueError(f"ordered cache file missing: {file_path}")
        stat_result = file_path.stat()
        if int(entry["size"]) != int(stat_result.st_size):
            raise ValueError(f"ordered_cache_entries[{index}] size mismatch for {relative_path}")
        if int(entry["mtime_ns"]) != int(stat_result.st_mtime_ns):
            raise ValueError(f"ordered_cache_entries[{index}] mtime mismatch for {relative_path}")
        token_count = int(entry["token_count"])
        if token_count < 0:
            raise ValueError(f"ordered_cache_entries[{index}] token_count must be non-negative")
        normalized_entries.append(
            {
                "relative_path": relative_path.as_posix(),
                "size": int(entry["size"]),
                "mtime_ns": int(entry["mtime_ns"]),
                "sha256": None if entry["sha256"] is None else str(entry["sha256"]),
                "token_count": token_count,
            }
        )
        resolved_files.append(str(file_path))
    fingerprint_payload = {
        "cache_root": str(resolved_cache_dir),
        "builder_manifest": builder_manifest,
        "ordered_cache_entries": normalized_entries,
        "span_order_sha256": payload.get("span_order_sha256"),
        "total_rows": payload.get("total_rows"),
        "total_spans": payload.get("total_spans"),
    }
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "cache_root": str(resolved_cache_dir),
        "builder_manifest": fingerprint_payload["builder_manifest"],
        "files": resolved_files,
        "ordered_cache_entries": normalized_entries,
        "cache_identity": hashlib.sha256(_json_dumps(fingerprint_payload).encode("utf-8")).hexdigest(),
    }


def validate_checkpoint_manifest_lineage(
    *,
    manifest: dict[str, object],
    manifest_path: str | Path,
    output_root: str | Path,
    expected_run_id: str,
    checkpoints_root: str | Path,
    expected_arm: str,
    expected_lineage: str,
    expected_out_dir: str | Path,
    expected_training_seed: int,
    expected_checkpoint_identity: str,
) -> Path:
    resolved_root = Path(output_root).resolve(strict=False)
    resolved_manifest_path = Path(manifest_path).resolve(strict=False)
    resolved_checkpoints_root = Path(checkpoints_root).resolve(strict=False)
    try:
        resolved_manifest_path.relative_to(resolved_root)
        resolved_checkpoints_root.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("manifest and checkpoints_root must stay under output_root") from exc
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete checkpoint manifest: {resolved_manifest_path}")
    label = str(manifest.get("label", ""))
    checkpoint_dir = Path(str(manifest.get("checkpoint_dir", ""))).resolve(strict=False)
    try:
        checkpoint_dir.relative_to(resolved_checkpoints_root)
    except ValueError as exc:
        raise ValueError(f"checkpoint_dir must stay under checkpoints_root: {checkpoint_dir}") from exc
    if manifest.get("run_id") != expected_run_id:
        raise ValueError(f"checkpoint manifest run_id mismatch: expected {expected_run_id}, got {manifest.get('run_id')}")
    if manifest.get("arm") != expected_arm:
        raise ValueError(f"checkpoint manifest arm mismatch: expected {expected_arm}, got {manifest.get('arm')}")
    if manifest.get("lineage") != expected_lineage:
        raise ValueError(f"checkpoint manifest lineage mismatch: expected {expected_lineage}, got {manifest.get('lineage')}")
    resolved_out_dir = Path(expected_out_dir).resolve(strict=False)
    if Path(str(manifest.get("out_dir", ""))).resolve(strict=False) != resolved_out_dir:
        raise ValueError(f"checkpoint manifest out_dir mismatch: expected {resolved_out_dir}, got {manifest.get('out_dir')}")
    if int(manifest.get("training_seed")) != int(expected_training_seed):
        raise ValueError(
            f"checkpoint manifest training_seed mismatch: expected {expected_training_seed}, got {manifest.get('training_seed')}"
        )
    if manifest.get("checkpoint_identity") != expected_checkpoint_identity:
        raise ValueError(
            "checkpoint manifest checkpoint_identity mismatch: "
            f"expected {expected_checkpoint_identity}, got {manifest.get('checkpoint_identity')}"
        )
    if checkpoint_dir.parent != resolved_checkpoints_root:
        raise ValueError(f"checkpoint_dir parent mismatch for checkpoints_root: {checkpoint_dir}")
    if checkpoint_dir.name != label:
        raise ValueError(f"checkpoint_dir leaf must equal label: {checkpoint_dir.name} vs {label}")
    if checkpoint_dir != resolved_manifest_path.parent:
        raise ValueError(f"checkpoint_dir must equal manifest parent: {checkpoint_dir} vs {resolved_manifest_path.parent}")
    return checkpoint_dir


def validate_resume_contract(resume_state: dict[str, object], *, expected_contract: dict[str, object]) -> None:
    contract = resume_state.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("resume state is missing contract")
    for key, want in expected_contract.items():
        got = contract.get(key)
        if got != want:
            raise ValueError(f"resume contract mismatch for {key}: expected {want}, got {got}")


def should_stop_after_publish(stop_after_label: str | None, published_label: str) -> bool:
    return bool(stop_after_label) and stop_after_label == published_label
