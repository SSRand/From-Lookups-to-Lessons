"""Atomic, immutable checkpoint publishing, resume pointers, and RNG capture for train.pretrain."""

import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch


_RESERVED_MANIFEST_KEYS = {"status", "label", "progress", "checkpoint_dir"}
_STABLE_PROGRESS_KEYS = (
    "epoch",
    "epoch_complete",
    "optimizer_step",
    "epoch_step",
    "epoch_fraction",
    "seen_tokens",
    "world_size",
    "effective_batch",
)


def _json_dumps(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _fsync_directory(path):
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _cleanup_tree(path):
    if path.exists():
        _make_tree_writable(path)
        shutil.rmtree(path, onerror=_handle_remove_readonly)


def _handle_remove_readonly(func, path, excinfo):
    if Path(path).is_symlink():
        func(path)
        return
    os.chmod(path, 0o700)
    func(path)


def _make_tree_writable(path):
    root = Path(path)
    for child in root.rglob("*"):
        if child.is_symlink():
            continue
        child.chmod(0o700)
    root.chmod(0o700)


def atomic_write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_json_dumps(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        _fsync_directory(target.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return target


def append_jsonl_durable(path, record):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(_json_dumps(record))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(target.parent)
    return target


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def validate_under_root(path, output_root):
    target = Path(path).resolve(strict=False)
    root = Path(output_root).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"durable output must stay under {root}: {target}") from exc
    return target


def _make_read_only_descendants(path):
    for child in Path(path).rglob("*"):
        if child.is_symlink():
            raise ValueError(f"checkpoint tree contains symlink: {child}")
        if child.is_dir():
            child.chmod(0o555)
        else:
            child.chmod(0o444)


def _make_read_only_tree(path):
    _make_read_only_descendants(path)
    Path(path).chmod(0o555)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_read_only_tree(path):
    root = Path(path)
    if root.stat().st_mode & 0o222:
        return False
    for child in root.rglob("*"):
        if child.stat().st_mode & 0o222:
            return False
    return True


def validate_immutable_checkpoint_tree(path, output_root):
    root = validate_under_root(path, output_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"checkpoint tree is missing or invalid: {root}")
    if root.stat().st_mode & 0o222:
        raise ValueError(f"checkpoint tree is not immutable: {root}")
    for child in root.rglob("*"):
        if child.is_symlink():
            raise ValueError(f"checkpoint tree contains symlink: {child}")
        if not child.is_dir() and not child.is_file():
            raise ValueError(f"checkpoint tree contains unsupported entry: {child}")
        if child.stat().st_mode & 0o222:
            raise ValueError(f"checkpoint tree is not immutable: {child}")
    return root


def _stable_progress_identity(progress):
    return {key: progress.get(key) for key in _STABLE_PROGRESS_KEYS}


def _validate_existing_manifest(destination, *, label, progress, metadata, output_root):
    manifest_path = destination / "manifest.json"
    if not manifest_path.exists():
        raise FileExistsError(f"checkpoint label already published without manifest: {destination}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise FileExistsError(f"checkpoint label already published with incomplete manifest: {destination}")
    if manifest.get("label") != label:
        raise FileExistsError(f"checkpoint label already published with conflicting label: {destination}")
    manifest_progress = manifest.get("progress")
    if _stable_progress_identity(manifest_progress) != _stable_progress_identity(progress):
        raise FileExistsError(f"checkpoint label already published with conflicting progress: {destination}")
    manifest_dir = validate_under_root(manifest.get("checkpoint_dir"), output_root)
    if manifest_dir != destination:
        raise FileExistsError(f"checkpoint label already published with conflicting checkpoint_dir: {destination}")
    if metadata.get("checkpoint_identity") is None:
        raise FileExistsError(f"checkpoint label already published without checkpoint identity request: {destination}")
    for key, value in metadata.items():
        if manifest.get(key) != value:
            raise FileExistsError(f"checkpoint label already published with conflicting metadata: {destination}")
    try:
        validate_immutable_checkpoint_tree(destination, output_root)
    except ValueError as exc:
        raise FileExistsError(
            f"checkpoint label already published but is not immutable: {destination}"
        ) from exc
    return manifest_path


def _publish_staging_dir(staging_dir, destination):
    os.replace(staging_dir, destination)


def publish_checkpoint(
    *,
    checkpoints_root,
    latest_resume_path,
    label,
    progress,
    output_root,
    build_checkpoint,
    metadata=None,
):
    metadata = metadata or {}
    reserved_keys = sorted(_RESERVED_MANIFEST_KEYS.intersection(metadata))
    if reserved_keys:
        raise ValueError(f"reserved manifest keys cannot be overridden: {reserved_keys}")
    checkpoints_dir = validate_under_root(checkpoints_root, output_root)
    resume_pointer = validate_under_root(latest_resume_path, output_root)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoints_dir / label
    if destination.exists():
        manifest_path = _validate_existing_manifest(
            destination,
            label=label,
            progress=progress,
            metadata=metadata,
            output_root=output_root,
        )
        atomic_write_json(
            resume_pointer,
            {
                "label": label,
                "manifest_path": str(manifest_path),
            },
        )
        return manifest_path

    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{label}.tmp-", dir=checkpoints_dir)
    )
    try:
        build_checkpoint(staging_dir)
        manifest = {
            "status": "complete",
            "label": label,
            "progress": progress,
            "checkpoint_dir": str(destination),
        }
        manifest.update(metadata)
        atomic_write_json(staging_dir / "manifest.json", manifest)
        _make_read_only_tree(staging_dir)
        validate_immutable_checkpoint_tree(staging_dir, output_root)
        _publish_staging_dir(staging_dir, destination)
        _fsync_directory(checkpoints_dir)
        atomic_write_json(
            resume_pointer,
            {
                "label": label,
                "manifest_path": str(destination / "manifest.json"),
            },
        )
        return destination / "manifest.json"
    except Exception:
        _cleanup_tree(staging_dir)
        raise


def load_resume_checkpoint(latest_resume_path, output_root):
    resume_pointer = validate_under_root(latest_resume_path, output_root)
    resume_payload = _read_json(resume_pointer)
    manifest_path = validate_under_root(
        resume_payload["manifest_path"],
        output_root,
    )
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete checkpoint manifest: {manifest_path}")
    if resume_payload.get("label") != manifest.get("label"):
        raise ValueError(
            f"pointer label {resume_payload.get('label')} does not match manifest label {manifest.get('label')}"
        )
    checkpoint_dir = validate_under_root(manifest.get("checkpoint_dir"), output_root)
    if checkpoint_dir != manifest_path.parent:
        raise ValueError(f"manifest checkpoint_dir does not match parent: {manifest_path}")
    validate_immutable_checkpoint_tree(checkpoint_dir, output_root)
    return {
        "label": resume_payload["label"],
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }
