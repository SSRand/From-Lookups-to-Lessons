"""ResMem trainer: residual self-distillation of a parametric memory on a frozen base LM.

The memory reads h_t (input to the last decoder MLP) and emits logits s; the training loss is
0.5 * KL(top-64 teacher || softmax(z_B + s)) + 0.5 * CE, with z_B the frozen base logits
(--objective-mode joint_base_anchored). --objective-mode standalone_memory trains softmax(s) alone.
Teacher targets are the cached top-64 distributions from teacher.forward_teacher. Checkpoints are
published atomically with exact post-step resume state.
"""

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from resmem.memory import MistralMLPModel
from train import fractional_checkpoints
from train.checkpointing import (
    atomic_write_json,
    capture_rng_state,
    load_resume_checkpoint,
    publish_checkpoint,
    restore_rng_state,
    validate_immutable_checkpoint_tree,
    validate_under_root,
)
from train.fractional_checkpoints import (
    advance_rank_cursor,
    build_fractional_milestones,
    checkpoint_label,
    cumulative_gpu_hours,
    cumulative_wall_seconds,
    default_fractional_points,
    load_cache_manifest_identity,
    normalize_resume_position,
    normalize_fractional_points,
    parse_fractional_points,
    rank_cursor,
    should_stop_after_publish,
    validate_checkpoint_manifest_lineage,
    validate_resume_contract,
)
from train.train_logging import append_jsonl, loss_record
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-v0.3"
DEFAULT_MEMORY_INIT = "Rubin-Wei/MLPMemory-Mistral-wikipedia"
TARGET_GLOBAL_BATCH = 1024
ALPHA = 0.5
OBJECTIVE_MODES = ("joint_base_anchored", "standalone_memory")


def broadcast_student_parameters(student):
    for parameter in student.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in student.buffers():
        dist.broadcast(buffer.data, src=0)


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def resolve_fractional_points(raw):
    points = parse_fractional_points(raw)
    if points:
        return points
    return normalize_fractional_points(default_fractional_points())


def _epoch_fraction_label(epoch, fraction):
    if epoch == 0:
        return checkpoint_label(epoch=epoch, epoch_fraction=fraction, epoch_complete=False)
    rendered = f"{fraction:.7f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"ep{epoch}p{rendered.split('p', 1)[1]}"


def build_epoch_checkpoint_milestones(
    *,
    cap_steps_per_epoch,
    epochs,
    first_epoch_fractions,
    later_epoch_fractions=(),
):
    if cap_steps_per_epoch <= 0:
        raise ValueError("cap_steps_per_epoch must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    schedule = [
        {
            "label": "pass0",
            "epoch": 0,
            "epoch_step": 0,
            "optimizer_step": 0,
            "absolute_cumulative_epoch": 0.0,
            "epoch_complete": False,
        }
    ]
    for epoch in range(epochs):
        fractions = first_epoch_fractions if epoch == 0 else later_epoch_fractions
        milestone_rows = build_fractional_milestones(
            cap_steps_per_epoch=cap_steps_per_epoch,
            fractions=normalize_fractional_points(tuple(fractions)),
        )
        for row in milestone_rows:
            epoch_step = int(row["step"])
            actual_fraction = epoch_step / cap_steps_per_epoch
            nominal_fraction = float(row["fraction"])
            schedule.append(
                {
                    "label": _epoch_fraction_label(epoch, nominal_fraction),
                    "epoch": epoch,
                    "epoch_step": epoch_step,
                    "optimizer_step": epoch * cap_steps_per_epoch + epoch_step,
                    "absolute_cumulative_epoch": epoch + actual_fraction,
                    "epoch_complete": False,
                }
            )
        schedule.append(
            {
                "label": checkpoint_label(epoch=epoch, epoch_fraction=1.0, epoch_complete=True),
                "epoch": epoch,
                "epoch_step": cap_steps_per_epoch,
                "optimizer_step": (epoch + 1) * cap_steps_per_epoch,
                "absolute_cumulative_epoch": float(epoch + 1),
                "epoch_complete": True,
            }
        )
    return schedule


def build_checkpoint_progress(
    *,
    epoch,
    epoch_step,
    epoch_fraction,
    epoch_complete,
    optimizer_step,
    seen_tokens,
    wall_seconds,
    world_size,
    effective_batch,
):
    return {
        "epoch": int(epoch),
        "epoch_complete": bool(epoch_complete),
        "epoch_step": int(epoch_step),
        "epoch_fraction": float(epoch_fraction),
        "absolute_cumulative_epoch": float(epoch + epoch_fraction),
        "optimizer_step": int(optimizer_step),
        "seen_tokens": int(seen_tokens),
        "wall_seconds": float(wall_seconds),
        "world_size": int(world_size),
        "effective_batch": int(effective_batch),
    }


def build_checkpoint_resume_metadata():
    return {
        "resume_state_path": "state.pt",
        "resume_state_manifest_path": "resume_state_manifest.json",
        "resume_state_components": ["optimizer", "scheduler", "rank_rng", "rank_cursor"],
    }


def _optimizer_steps(optimizer_state):
    steps = []
    for value in optimizer_state.get("state", {}).values():
        step = value.get("step") if isinstance(value, dict) else None
        if step is not None:
            steps.append(int(step.item() if hasattr(step, "item") else step))
    return steps


def validate_loaded_resume_state(selection, state_payload):
    manifest_progress = selection.get("manifest", {}).get("progress")
    state_progress = state_payload.get("global_progress")
    if not isinstance(manifest_progress, dict) or state_progress != manifest_progress:
        raise ValueError("resume state global progress does not match checkpoint manifest")
    optimizer = state_payload.get("optimizer")
    scheduler = state_payload.get("scheduler")
    if not isinstance(optimizer, dict) or not isinstance(scheduler, dict):
        raise ValueError("resume state is missing optimizer or scheduler state")
    expected_step = int(manifest_progress["optimizer_step"])
    if int(scheduler.get("last_epoch", -1)) != expected_step:
        raise ValueError("resume scheduler step does not match checkpoint progress")
    optimizer_steps = _optimizer_steps(optimizer)
    if expected_step > 0 and (not optimizer_steps or any(step != expected_step for step in optimizer_steps)):
        raise ValueError("resume optimizer step does not match checkpoint progress")
    return state_payload


def sha256_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_identity(path):
    model_path = Path(path).resolve(strict=False) if Path(path).exists() else Path(path)
    config_path = model_path / "config.json"
    return {
        "path": str(model_path),
        "config_sha256": sha256_path(config_path) if config_path.exists() else None,
    }


def training_logits_for_objective(objective_mode, *, zbase, student_out):
    if objective_mode == "joint_base_anchored":
        return zbase + student_out
    if objective_mode == "standalone_memory":
        return student_out
    raise ValueError(f"unknown objective_mode: {objective_mode}")


def build_resume_contract(
    *,
    run_id,
    out,
    checkpoints_root,
    cache_identity,
    world,
    batch_size,
    effective_batch,
    cap_steps_per_epoch,
    fractional_points,
    later_fractional_points=(),
    lr,
    warmup,
    mem_layers,
    epochs,
    total_steps,
    arm_name,
    lineage,
    seed,
    objective_mode,
    base_model=DEFAULT_BASE_MODEL,
    memory_init=DEFAULT_MEMORY_INIT,
):
    trainer_path = Path(__file__).resolve(strict=False)
    helper_path = Path(fractional_checkpoints.__file__).resolve(strict=False)
    return {
        "run_id": str(run_id),
        "arm": str(arm_name),
        "lineage": str(lineage),
        "out_dir": str(out),
        "checkpoints_root": str(checkpoints_root),
        "world_size": int(world),
        "batch_size": int(batch_size),
        "effective_batch": int(effective_batch),
        "cap_steps_per_epoch": int(cap_steps_per_epoch),
        "fractional_points": list(normalize_fractional_points(tuple(fractional_points))),
        "later_fractional_points": list(
            normalize_fractional_points(tuple(later_fractional_points))
        ),
        "checkpoint_schedule": build_epoch_checkpoint_milestones(
            cap_steps_per_epoch=cap_steps_per_epoch,
            epochs=epochs,
            first_epoch_fractions=fractional_points,
            later_epoch_fractions=later_fractional_points,
        ),
        "cache_manifest_path": str(cache_identity["manifest_path"]),
        "cache_manifest_sha256": str(cache_identity["manifest_sha256"]),
        "cache_identity": str(cache_identity["cache_identity"]),
        "trainer_source_sha256": sha256_path(trainer_path),
        "fractional_helper_sha256": sha256_path(helper_path),
        "base_identity": model_identity(base_model),
        "init_identity": model_identity(memory_init),
        "optimizer": {"type": "AdamW", "lr": float(lr)},
        "scheduler": {"type": "linear_warmup", "warmup": int(warmup), "total_steps": int(total_steps)},
        "seed": int(seed),
        "mem_layers": int(mem_layers),
        "epochs": int(epochs),
        "total_steps": int(total_steps),
        "objective_mode": str(objective_mode),
    }


def build_resume_contract_sidecar_payload(*, resume_contract):
    return {"contract": dict(resume_contract)}


def load_resume_contract_sidecar(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("contract"), dict):
        raise ValueError(f"resume contract sidecar must be an object with contract: {path}")
    return payload


def publish_resume_contract_sidecar(*, sidecar_path, resume_contract):
    path = Path(sidecar_path)
    payload = build_resume_contract_sidecar_payload(resume_contract=resume_contract)
    existing = None
    try:
        existing = load_resume_contract_sidecar(path)
    except FileNotFoundError:
        existing = None
    except json.JSONDecodeError as exc:
        raise ValueError(f"resume contract sidecar drift: {path}") from exc
    if existing is not None:
        if existing != payload:
            raise ValueError(f"resume contract sidecar drift: {path}")
        return existing
    atomic_write_json(path, payload)
    return payload


def build_run_status_payload(
    *,
    status,
    run_id,
    training_seed,
    attempt_id,
    resume_from_step,
    latest_label,
    optimizer_step,
    epoch,
    epoch_step,
    epoch_fraction,
    seen_tokens,
    wall_seconds,
    gpu_hours,
    world_size,
    effective_batch,
    objective_mode,
):
    return {
        "status": status,
        "run_id": run_id,
        "training_seed": int(training_seed),
        "attempt_id": attempt_id,
        "resume_from_step": int(resume_from_step),
        "latest_label": latest_label,
        "optimizer_step": int(optimizer_step),
        "epoch": int(epoch),
        "epoch_step": int(epoch_step),
        "epoch_fraction": float(epoch_fraction),
        "seen_tokens": int(seen_tokens),
        "wall_seconds": float(wall_seconds),
        "gpu_hours": float(gpu_hours),
        "world_size": int(world_size),
        "effective_batch": int(effective_batch),
        "objective_mode": str(objective_mode),
    }


def build_train_log_payload(
    *,
    arm_name,
    lineage,
    run_id,
    training_seed,
    epochs,
    cap_steps_per_epoch,
    total_steps,
    world,
    accum_pos,
    effective_batch,
    lr,
    warmup,
    mem_layers,
    alpha,
    wall_seconds,
    gpu_hours,
    cache_identity,
    latest_resume_path,
    objective_mode,
):
    return {
        "arm": arm_name,
        "lineage": lineage,
        "run_id": run_id,
        "training_seed": int(training_seed),
        "epochs": int(epochs),
        "cap_steps_per_epoch": int(cap_steps_per_epoch),
        "total_steps": int(total_steps),
        "world": int(world),
        "accum_pos": int(accum_pos),
        "global_batch": int(effective_batch),
        "lr": float(lr),
        "warmup": int(warmup),
        "mem_layers": int(mem_layers),
        "alpha": float(alpha),
        "wall_sec": float(wall_seconds),
        "gpu_hours": float(gpu_hours),
        "cache_manifest_path": str(cache_identity["manifest_path"]),
        "cache_manifest_sha256": str(cache_identity["manifest_sha256"]),
        "cache_identity": str(cache_identity["cache_identity"]),
        "latest_resume_path": str(latest_resume_path),
        "objective_mode": str(objective_mode),
    }


def count_local_tokens_from_manifest(*, ordered_cache_entries, rank, world):
    total = 0
    for entry in ordered_cache_entries[rank::world]:
        total += int(entry["token_count"])
    return total


def checkpoint_identity_for(*, run_id, arm_name, label, optimizer_step):
    return f"{run_id}:{arm_name}:{label}:{optimizer_step}"


def load_resume_selection(
    *,
    resume_manifest,
    latest_resume_path,
    output_root,
    expected_run_id,
    checkpoints_root,
    expected_arm,
    expected_lineage,
    expected_out_dir,
    expected_training_seed,
):
    if resume_manifest:
        manifest_path = validate_under_root(resume_manifest, output_root)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        expected_checkpoint_identity = checkpoint_identity_for(
            run_id=expected_run_id,
            arm_name=expected_arm,
            label=str(manifest["label"]),
            optimizer_step=int(manifest["progress"]["optimizer_step"]),
        )
        checkpoint_root = validate_checkpoint_manifest_lineage(
            manifest=manifest,
            manifest_path=manifest_path,
            output_root=output_root,
            expected_run_id=expected_run_id,
            checkpoints_root=checkpoints_root,
            expected_arm=expected_arm,
            expected_lineage=expected_lineage,
            expected_out_dir=expected_out_dir,
            expected_training_seed=expected_training_seed,
            expected_checkpoint_identity=expected_checkpoint_identity,
        )
        validate_immutable_checkpoint_tree(checkpoint_root, output_root)
        return {"label": manifest["label"], "manifest_path": str(manifest_path), "manifest": manifest}
    selection = load_resume_checkpoint(latest_resume_path, output_root)
    expected_checkpoint_identity = checkpoint_identity_for(
        run_id=expected_run_id,
        arm_name=expected_arm,
        label=str(selection["manifest"]["label"]),
        optimizer_step=int(selection["manifest"]["progress"]["optimizer_step"]),
    )
    validate_checkpoint_manifest_lineage(
        manifest=selection["manifest"],
        manifest_path=selection["manifest_path"],
        output_root=output_root,
        expected_run_id=expected_run_id,
        checkpoints_root=checkpoints_root,
        expected_arm=expected_arm,
        expected_lineage=expected_lineage,
        expected_out_dir=expected_out_dir,
        expected_training_seed=expected_training_seed,
        expected_checkpoint_identity=expected_checkpoint_identity,
    )
    return selection


def load_state_payload(selection):
    checkpoint_root = Path(selection["manifest_path"]).parent
    # The immutable manifest and full lineage are validated before this call.
    # RNG state contains Python/NumPy objects and therefore cannot be decoded
    # by PyTorch's weights-only unpickler.
    state_payload = torch.load(checkpoint_root / "state.pt", map_location="cpu", weights_only=False)
    validate_loaded_resume_state(selection, state_payload)
    return checkpoint_root, state_payload


def sync_rank_metadata(rank_payload):
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, rank_payload)
    return gathered


def sync_stop_flag(rank, should_stop, dev):
    stop_tensor = torch.tensor([1 if rank == 0 and should_stop else 0], device=dev, dtype=torch.int32)
    dist.broadcast(stop_tensor, src=0)
    return bool(stop_tensor.item())


class LocalTokenStream:
    def __init__(self, *, files, batch_size, base_kz, vocab_size, dev, start_cursor):
        self.files = files
        self.batch_size = batch_size
        self.base_kz = base_kz
        self.vocab_size = vocab_size
        self.dev = dev
        self.cursor = dict(start_cursor)
        self.current_file_index = None
        self.current_file = None
        self.current_batch = None

    def _load_file(self, file_index):
        if self.current_file_index == file_index:
            return
        if self.current_file is not None:
            self.current_file["data"].close()
            self.current_file = None
            self.current_file_index = None
        data = np.load(self.files[file_index], mmap_mode="r")
        span_lens = data["span_lens"].astype(np.int64)
        token_lens = np.clip(span_lens - 1, 0, None)
        row_offsets = np.zeros(len(token_lens) + 1, dtype=np.int64)
        row_offsets[1:] = np.cumsum(token_lens)
        self.current_file_index = file_index
        self.current_file = {
            "data": data,
            "span_ids": data["span_ids"],
            "span_lens": span_lens,
            "teacher_tok": data["teacher_tok"],
            "teacher_prob": data["teacher_prob"],
            "row_offsets": row_offsets,
        }

    def _advance_to_next_cursor(self, *, file_index, span_start, span_end, next_batch_row_pointer, rank_seen_tokens):
        self.cursor = advance_rank_cursor(
            file_index=file_index,
            span_batch_start=span_start,
            batch_base_row_pointer=int(self.current_file["row_offsets"][span_start]),
            next_batch_row_pointer=next_batch_row_pointer,
            chunk_end=int(self.current_file["row_offsets"][span_end] - self.current_file["row_offsets"][span_start]),
            batch_token_count=int(self.current_file["row_offsets"][span_end] - self.current_file["row_offsets"][span_start]),
            next_span_batch_start=span_end,
            has_more_batches=span_end < len(self.current_file["span_lens"]),
            has_more_files=file_index + 1 < len(self.files),
            rank_seen_tokens=rank_seen_tokens,
        )
        self.current_batch = None

    def _ensure_batch(self):
        """Assemble the next batch: base forward on the spans (h_t, z_B), next-token targets,
        and the renormalized top-64 teacher targets scattered over the vocabulary."""
        while self.current_batch is None:
            file_index = int(self.cursor["file_index"])
            if file_index >= len(self.files):
                raise StopIteration
            self._load_file(file_index)
            span_start = int(self.cursor["span_batch_start"])
            if span_start >= len(self.current_file["span_lens"]):
                self.cursor = rank_cursor(
                    file_index=file_index + 1,
                    span_batch_start=0,
                    token_chunk_offset=0,
                    cache_row_pointer=0,
                    rank_seen_tokens=int(self.cursor["rank_seen_tokens"]),
                )
                continue
            span_end = min(span_start + self.batch_size, len(self.current_file["span_lens"]))
            batch_base_row_pointer = int(self.current_file["row_offsets"][span_start])
            next_batch_row_pointer = int(self.current_file["row_offsets"][span_end])
            token_count = next_batch_row_pointer - batch_base_row_pointer
            if int(self.cursor["cache_row_pointer"]) != batch_base_row_pointer:
                raise ValueError(
                    f"resume cache row mismatch for file {file_index} batch {span_start}: "
                    f"expected {batch_base_row_pointer}, got {self.cursor['cache_row_pointer']}"
                )
            token_chunk_offset = int(self.cursor["token_chunk_offset"])
            if token_chunk_offset > token_count:
                raise ValueError(f"resume token offset {token_chunk_offset} exceeds batch size {token_count}")
            if token_count == 0 or token_chunk_offset == token_count:
                self._advance_to_next_cursor(
                    file_index=file_index,
                    span_start=span_start,
                    span_end=span_end,
                    next_batch_row_pointer=next_batch_row_pointer,
                    rank_seen_tokens=int(self.cursor["rank_seen_tokens"]),
                )
                continue
            batch_spans = self.current_file["span_ids"][span_start:span_end]
            batch_lens = self.current_file["span_lens"][span_start:span_end]
            key, zbase, xids = self.base_kz(batch_spans, batch_lens)
            key_parts = []
            zbase_parts = []
            target_parts = []
            for idx in range(len(batch_lens)):
                length = int(batch_lens[idx])
                if length <= 1:
                    continue
                key_parts.append(key[idx, : length - 1])
                zbase_parts.append(zbase[idx, : length - 1])
                target_parts.append(xids[idx, 1:length])
            batch_tok = self.current_file["teacher_tok"][batch_base_row_pointer : batch_base_row_pointer + token_count]
            batch_prob = self.current_file["teacher_prob"][batch_base_row_pointer : batch_base_row_pointer + token_count].astype(
                np.float32
            )
            pk = torch.zeros(token_count, self.vocab_size, device=self.dev)
            pk.scatter_(
                1,
                torch.from_numpy(batch_tok.astype(np.int64)).to(self.dev),
                torch.from_numpy(batch_prob).to(self.dev),
            )
            pk = pk / pk.sum(1, keepdim=True).clamp_min(1e-9)
            self.current_batch = {
                "file_index": file_index,
                "span_batch_start": span_start,
                "next_span_batch_start": span_end,
                "batch_base_row_pointer": batch_base_row_pointer,
                "next_batch_row_pointer": next_batch_row_pointer,
                "token_count": token_count,
                "offset": token_chunk_offset,
                "has_more_batches": span_end < len(self.current_file["span_lens"]),
                "has_more_files": file_index + 1 < len(self.files),
                "key": torch.cat(key_parts),
                "zbase": torch.cat(zbase_parts),
                "target": torch.cat(target_parts),
                "pk": pk,
            }

    def take_optimizer_step_tokens(self, accum_pos, rank_seen_tokens):
        slices = []
        taken = 0
        while taken < accum_pos:
            self._ensure_batch()
            batch = self.current_batch
            start = int(batch["offset"])
            available = int(batch["token_count"]) - start
            take = min(accum_pos - taken, available)
            end = start + take
            slices.append(
                {
                    "key": batch["key"][start:end],
                    "zbase": batch["zbase"][start:end],
                    "pk": batch["pk"][start:end],
                    "target": batch["target"][start:end],
                    "token_count": take,
                }
            )
            rank_seen_tokens += take
            self.cursor = advance_rank_cursor(
                file_index=int(batch["file_index"]),
                span_batch_start=int(batch["span_batch_start"]),
                batch_base_row_pointer=int(batch["batch_base_row_pointer"]),
                next_batch_row_pointer=int(batch["next_batch_row_pointer"]),
                chunk_end=end,
                batch_token_count=int(batch["token_count"]),
                next_span_batch_start=int(batch["next_span_batch_start"]),
                has_more_batches=bool(batch["has_more_batches"]),
                has_more_files=bool(batch["has_more_files"]),
                rank_seen_tokens=rank_seen_tokens,
            )
            batch["offset"] = end
            taken += take
            if end == int(batch["token_count"]):
                self.current_batch = None
        return {"slices": slices, "cursor": dict(self.cursor), "rank_seen_tokens": rank_seen_tokens}

    def close(self):
        if self.current_file is not None:
            self.current_file["data"].close()
            self.current_file = None
            self.current_file_index = None
        self.current_batch = None


def worker(rank, world, args):
    accum_pos = max(1, TARGET_GLOBAL_BATCH // world)
    effective_batch = accum_pos * world
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29583")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(hours=2))
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    process_seed = int(args.seed) + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)

    out = Path(args.out).resolve(strict=False)
    output_root = out
    checkpoints_root = out / "checkpoints"
    telemetry_dir = out / "telemetry"
    latest_resume_path = out / "latest_resume.json"
    resume_contract_sidecar_path = out / "resume_contract.json"
    run_status_path = out / "run_status.json"
    run_id = args.run_id or out.name
    arm_name = args.arm or out.name
    lineage = args.lineage or arm_name
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        telemetry_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    run_start = time.time()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    time.sleep(rank * 6)
    try:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).cuda(rank).eval()
    except Exception:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).cuda(rank).eval()
    for parameter in base.parameters():
        parameter.requires_grad = False
    hidden_size = base.config.hidden_size
    vocab_size = base.config.vocab_size
    keycap = {"x": None}
    normcap = {"x": None}
    base.model.layers[-1].mlp.register_forward_pre_hook(
        lambda _module, args_in, _kwargs: keycap.__setitem__("x", args_in[0].detach()),
        with_kwargs=True,
    )
    base.model.norm.register_forward_hook(lambda _module, _inputs, output: normcap.__setitem__("x", output.detach()))

    @torch.no_grad()
    def base_kz(span_ids_np, lens_np):
        max_length = int(lens_np.max())
        input_ids = torch.from_numpy(span_ids_np[:, :max_length].astype(np.int64)).to(dev)
        arange = torch.arange(max_length, device=dev)[None, :]
        attention_mask = (arange < torch.from_numpy(lens_np.astype(np.int64)).to(dev)[:, None]).long()
        base(input_ids=input_ids, attention_mask=attention_mask)
        return keycap["x"].clone(), base.lm_head(normcap["x"].clone()).float(), input_ids

    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    telemetry_steps = max(1, int(args.telemetry_steps))
    objective_mode = str(args.objective_mode)
    fractional_points = resolve_fractional_points(args.fractional_save_points)
    later_fractional_points = parse_fractional_points(args.later_fractional_save_points)
    if args.cache_manifest_sha256 and not args.cache_manifest:
        raise ValueError("--cache-manifest-sha256 requires --cache-manifest")
    cache_identity = load_cache_manifest_identity(
        cache_dir=args.cache,
        cache_manifest_path=args.cache_manifest,
    )
    if args.cache_manifest_sha256 and cache_identity["manifest_sha256"] != args.cache_manifest_sha256:
        raise ValueError(
            "cache manifest sha256 mismatch: "
            f"expected {args.cache_manifest_sha256}, got {cache_identity['manifest_sha256']}"
        )
    files = list(cache_identity["files"])
    my_files = files[rank::world]

    cap_tensor = torch.tensor(
        [count_local_tokens_from_manifest(ordered_cache_entries=cache_identity["ordered_cache_entries"], rank=rank, world=world) // accum_pos],
        device=dev,
        dtype=torch.int64,
    )
    dist.all_reduce(cap_tensor, op=dist.ReduceOp.MIN)
    cap_steps_per_epoch = int(cap_tensor.item())
    total_steps = cap_steps_per_epoch * epochs

    checkpoint_schedule = build_epoch_checkpoint_milestones(
        cap_steps_per_epoch=cap_steps_per_epoch,
        epochs=epochs,
        first_epoch_fractions=fractional_points,
        later_epoch_fractions=later_fractional_points,
    )
    milestone_by_position = {
        (int(item["epoch"]), int(item["epoch_step"])): item
        for item in checkpoint_schedule
        if not bool(item["epoch_complete"]) and str(item["label"]) != "pass0"
    }
    if rank == 0:
        atomic_write_json(
            out / "fractional_checkpoints.json",
            {
                "cap_steps_per_epoch": cap_steps_per_epoch,
                "milestones": build_fractional_milestones(
                    cap_steps_per_epoch=cap_steps_per_epoch,
                    fractions=fractional_points,
                ),
            },
        )
        atomic_write_json(
            out / "checkpoint_schedule.json",
            {
                "cap_steps_per_epoch": cap_steps_per_epoch,
                "epochs": epochs,
                "milestones": checkpoint_schedule,
            },
        )

    resume_contract = build_resume_contract(
        run_id=run_id,
        out=out,
        checkpoints_root=checkpoints_root,
        cache_identity=cache_identity,
        world=world,
        batch_size=batch_size,
        effective_batch=effective_batch,
        cap_steps_per_epoch=cap_steps_per_epoch,
        fractional_points=fractional_points,
        later_fractional_points=later_fractional_points,
        lr=args.lr,
        warmup=args.warmup,
        mem_layers=args.mem_layers,
        epochs=epochs,
        total_steps=total_steps,
        arm_name=arm_name,
        lineage=lineage,
        seed=args.seed,
        objective_mode=objective_mode,
        base_model=args.base_model,
        memory_init=args.memory_init,
    )
    if rank == 0:
        publish_resume_contract_sidecar(
            sidecar_path=resume_contract_sidecar_path,
            resume_contract=resume_contract,
        )
    dist.barrier()
    cfg = AutoConfig.from_pretrained(args.memory_init)
    attempt_id_values = [args.attempt_id if rank == 0 else None]
    if rank == 0 and attempt_id_values[0] is None:
        attempt_id_values[0] = f"{run_id}-{uuid.uuid4().hex[:12]}"
    dist.broadcast_object_list(attempt_id_values, src=0)
    attempt_id = attempt_id_values[0]
    resume_selection = None
    resume_state = None
    if args.resume_manifest or latest_resume_path.exists():
        resume_selection = load_resume_selection(
            resume_manifest=args.resume_manifest,
            latest_resume_path=latest_resume_path,
            output_root=output_root,
            expected_run_id=run_id,
            checkpoints_root=checkpoints_root,
            expected_arm=arm_name,
            expected_lineage=lineage,
            expected_out_dir=out,
            expected_training_seed=args.seed,
        )
        checkpoint_root, resume_state = load_state_payload(resume_selection)
        student = MistralMLPModel.from_pretrained(
            checkpoint_root / "checkpoint",
            config=AutoConfig.from_pretrained(checkpoint_root / "checkpoint"),
            input_dim=hidden_size,
            output_dim=hidden_size,
            torch_dtype=torch.bfloat16,
        ).cuda(rank).train()
    elif getattr(args, "mem_layers", 0) and args.mem_layers > 0:
        cfg.num_hidden_layers = int(args.mem_layers)
        student = MistralMLPModel(cfg, input_dim=hidden_size, output_dim=hidden_size).to(torch.bfloat16).cuda(rank).train()
    else:
        student = MistralMLPModel.from_pretrained(
            args.memory_init,
            config=cfg,
            input_dim=hidden_size,
            output_dim=hidden_size,
            torch_dtype=torch.bfloat16,
        ).cuda(rank).train()

    broadcast_student_parameters(student)

    optim = torch.optim.AdamW(student.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup, num_training_steps=total_steps)
    prior_wall_seconds = 0.0
    if resume_state is not None:
        validate_resume_contract(
            resume_state,
            expected_contract=resume_contract,
        )
        optim.load_state_dict(resume_state["optimizer"])
        move_optimizer_state_to_device(optim, dev)
        sched.load_state_dict(resume_state["scheduler"])
        restore_rng_state(resume_state["rank_progress"][rank]["rng_state"])
        prior_wall_seconds = float(resume_state["global_progress"]["wall_seconds"])

    global_progress = resume_state["global_progress"] if resume_state is not None else {
        "epoch": 0,
        "epoch_complete": False,
        "optimizer_step": 0,
        "epoch_step": 0,
        "epoch_fraction": 0.0,
        "seen_tokens": 0,
        "wall_seconds": 0.0,
        "world_size": world,
        "effective_batch": effective_batch,
    }
    resume_cursor = (
        resume_state["rank_progress"][rank]["cursor"]
        if resume_state is not None
        else rank_cursor(
            file_index=0,
            span_batch_start=0,
            token_chunk_offset=0,
            cache_row_pointer=0,
            rank_seen_tokens=0,
        )
    )
    resume_position = normalize_resume_position(global_progress=global_progress, cursor=resume_cursor)
    start_epoch = int(resume_position["start_epoch"])
    step = int(global_progress["optimizer_step"])
    resume_from_step = step
    seen_tokens = int(global_progress["seen_tokens"])
    rank_seen_tokens = int(resume_position["cursor"]["rank_seen_tokens"])
    last_label = resume_selection["label"] if resume_selection is not None else "pass0"
    last_epoch = int(global_progress["epoch"])
    last_epoch_step = int(resume_position["epoch_step"])
    last_epoch_fraction = float(global_progress["epoch_fraction"])
    last_metric_wall_seconds = prior_wall_seconds
    interval_loss_sum_local = 0.0
    interval_ce_sum_local = 0.0
    interval_kl_sum_local = 0.0
    interval_token_count_local = 0
    optim.zero_grad()

    def cumulative_wall_now():
        return cumulative_wall_seconds(previous_wall_seconds=prior_wall_seconds, current_run_seconds=time.time() - run_start)

    def emit_run_status(status):
        if rank != 0:
            return
        wall_seconds = cumulative_wall_now()
        atomic_write_json(
            run_status_path,
            build_run_status_payload(
                status=status,
                run_id=run_id,
                training_seed=args.seed,
                attempt_id=attempt_id,
                resume_from_step=resume_from_step,
                latest_label=last_label,
                optimizer_step=step,
                epoch=last_epoch,
                epoch_step=last_epoch_step,
                epoch_fraction=last_epoch_fraction,
                seen_tokens=seen_tokens,
                wall_seconds=wall_seconds,
                gpu_hours=cumulative_gpu_hours(wall_seconds=wall_seconds, world_size=world),
                world_size=world,
                effective_batch=effective_batch,
                objective_mode=objective_mode,
            ),
        )

    def build_rank_payload(cursor):
        return {"rank": rank, "cursor": cursor, "rng_state": capture_rng_state()}

    def write_checkpoint_payload(staging_dir, state_payload):
        checkpoint_dir = staging_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(checkpoint_dir, safe_serialization=False)
        tok.save_pretrained(checkpoint_dir)
        state_path = staging_dir / "state.pt"
        torch.save(state_payload, state_path)
        atomic_write_json(
            staging_dir / "resume_state_manifest.json",
            {
                "status": "complete",
                "state_path": "state.pt",
                "state_sha256": sha256_path(state_path),
                "progress": state_payload["global_progress"],
                "components": build_checkpoint_resume_metadata()["resume_state_components"],
            },
        )

    def publish_training_checkpoint(label, *, epoch, epoch_step, epoch_fraction, epoch_complete, cursor):
        nonlocal last_label, last_epoch, last_epoch_step, last_epoch_fraction
        flush_metrics_if_needed(epoch=epoch, epoch_step=epoch_step, epoch_fraction=epoch_fraction, force=True)
        rank_payloads = sync_rank_metadata(build_rank_payload(cursor))
        should_stop = False
        if rank == 0:
            wall_seconds = cumulative_wall_now()
            global_snapshot = build_checkpoint_progress(
                epoch=epoch,
                epoch_step=epoch_step,
                epoch_fraction=epoch_fraction,
                epoch_complete=epoch_complete,
                optimizer_step=step,
                seen_tokens=seen_tokens,
                wall_seconds=wall_seconds,
                world_size=world,
                effective_batch=effective_batch,
            )
            state_payload = {
                "global_progress": global_snapshot,
                "rank_progress": rank_payloads,
                "contract": {
                    **resume_contract,
                },
                "optimizer": optim.state_dict(),
                "scheduler": sched.state_dict(),
            }
            metadata = {
                "run_id": run_id,
                "arm": arm_name,
                "lineage": lineage,
                "training_seed": int(args.seed),
                "attempt_id": attempt_id,
                "resume_from_step": resume_from_step,
                "out_dir": str(out),
                "checkpoint_identity": checkpoint_identity_for(
                    run_id=run_id,
                    arm_name=arm_name,
                    label=label,
                    optimizer_step=step,
                ),
                "objective_mode": objective_mode,
                **build_checkpoint_resume_metadata(),
            }
            publish_checkpoint(
                checkpoints_root=checkpoints_root,
                latest_resume_path=latest_resume_path,
                label=label,
                progress=global_snapshot,
                output_root=output_root,
                build_checkpoint=lambda staging_dir: write_checkpoint_payload(staging_dir, state_payload),
                metadata=metadata,
            )
            should_stop = should_stop_after_publish(args.stop_after_label, label)
        dist.barrier()
        last_label = label
        last_epoch = int(epoch)
        last_epoch_step = int(epoch_step)
        last_epoch_fraction = float(epoch_fraction)
        emit_run_status("checkpoint_complete")
        return sync_stop_flag(rank, should_stop, dev)

    def run_optimizer_step(step_payload):
        nonlocal step, seen_tokens, interval_loss_sum_local, interval_ce_sum_local, interval_kl_sum_local, interval_token_count_local
        step_loss_sum_local = 0.0
        step_ce_sum_local = 0.0
        step_kl_sum_local = 0.0
        for item in step_payload["slices"]:
            student_out = student(inputs_embeds=item["key"].unsqueeze(1))[:, 0].float()
            objective_logits = training_logits_for_objective(
                objective_mode,
                zbase=item["zbase"],
                student_out=student_out,
            )
            loss_kl = F.kl_div(F.log_softmax(objective_logits, dim=-1), item["pk"], reduction="batchmean")
            loss_ce = F.cross_entropy(objective_logits, item["target"])
            loss = ALPHA * loss_kl + (1 - ALPHA) * loss_ce
            token_count = int(item["token_count"])
            (loss * token_count).backward()
            step_loss_sum_local += float(loss) * token_count
            step_ce_sum_local += float(loss_ce) * token_count
            step_kl_sum_local += float(loss_kl) * token_count
        for parameter in student.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad /= effective_batch
        optim.step()
        sched.step()
        optim.zero_grad()
        step += 1
        seen_tokens += effective_batch
        interval_loss_sum_local += step_loss_sum_local
        interval_ce_sum_local += step_ce_sum_local
        interval_kl_sum_local += step_kl_sum_local
        interval_token_count_local += accum_pos

    last_flushed_step = step

    def flush_metrics_if_needed(*, epoch, epoch_step, epoch_fraction, force=False):
        nonlocal interval_loss_sum_local, interval_ce_sum_local, interval_kl_sum_local
        nonlocal interval_token_count_local, last_metric_wall_seconds, last_flushed_step
        should_flush = step > last_flushed_step and (
            step % telemetry_steps == 0 or (force and interval_token_count_local > 0)
        )
        if not should_flush:
            return
        metric_tensor = torch.tensor(
            [
                interval_loss_sum_local,
                interval_ce_sum_local,
                interval_kl_sum_local,
                float(interval_token_count_local),
            ],
            device=dev,
            dtype=torch.float64,
        )
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            wall_seconds = cumulative_wall_now()
            global_token_count = max(1.0, float(metric_tensor[3].item()))
            record = loss_record(
                step,
                epoch,
                float(metric_tensor[1].item() / global_token_count),
                float(metric_tensor[2].item() / global_token_count),
                float(metric_tensor[0].item() / global_token_count),
                float(sched.get_last_lr()[0]),
                global_token_count / max(wall_seconds - last_metric_wall_seconds, 1e-9),
                epoch_fraction=epoch_fraction,
                seen_tokens=seen_tokens,
                wall_seconds=wall_seconds,
                gpu_hours=cumulative_gpu_hours(wall_seconds=wall_seconds, world_size=world),
                world_size=world,
            )
            record["attempt_id"] = attempt_id
            record["resume_from_step"] = resume_from_step
            record["objective_mode"] = objective_mode
            append_jsonl(telemetry_dir / "train_metrics.jsonl", record)
        last_metric_wall_seconds = cumulative_wall_now()
        interval_loss_sum_local = 0.0
        interval_ce_sum_local = 0.0
        interval_kl_sum_local = 0.0
        interval_token_count_local = 0
        last_flushed_step = step
        emit_run_status("attempt_running")

    stop_training = False
    if resume_state is None:
        stop_training = publish_training_checkpoint(
            checkpoint_label(epoch=0, epoch_fraction=0.0, epoch_complete=False),
            epoch=0,
            epoch_step=0,
            epoch_fraction=0.0,
            epoch_complete=False,
            cursor=resume_position["cursor"],
        )

    for epoch in range(start_epoch, epochs):
        epoch_cursor = (
            dict(resume_position["cursor"])
            if epoch == start_epoch
            else rank_cursor(
                file_index=0,
                span_batch_start=0,
                token_chunk_offset=0,
                cache_row_pointer=0,
                rank_seen_tokens=rank_seen_tokens,
            )
        )
        epoch_step = int(resume_position["epoch_step"]) if epoch == start_epoch else 0
        stream = LocalTokenStream(
            files=my_files,
            batch_size=batch_size,
            base_kz=base_kz,
            vocab_size=vocab_size,
            dev=dev,
            start_cursor=epoch_cursor,
        )
        try:
            while epoch_step < cap_steps_per_epoch and not stop_training:
                step_payload = stream.take_optimizer_step_tokens(accum_pos, rank_seen_tokens)
                rank_seen_tokens = int(step_payload["rank_seen_tokens"])
                run_optimizer_step(step_payload)
                epoch_step += 1
                last_epoch = epoch
                last_epoch_step = epoch_step
                last_epoch_fraction = epoch_step / cap_steps_per_epoch if cap_steps_per_epoch else 1.0
                resume_position["cursor"] = dict(step_payload["cursor"])
                flush_metrics_if_needed(epoch=epoch, epoch_step=epoch_step, epoch_fraction=last_epoch_fraction)
                milestone = milestone_by_position.get((epoch, epoch_step))
                if milestone is not None:
                    stop_training = publish_training_checkpoint(
                        str(milestone["label"]),
                        epoch=epoch,
                        epoch_step=epoch_step,
                        epoch_fraction=last_epoch_fraction,
                        epoch_complete=False,
                        cursor=step_payload["cursor"],
                    )
        finally:
            stream.close()
            gc.collect()
        resume_position["epoch_step"] = 0
        if stop_training:
            break
        epoch_complete_cursor = rank_cursor(
            file_index=0,
            span_batch_start=0,
            token_chunk_offset=0,
            cache_row_pointer=0,
            rank_seen_tokens=rank_seen_tokens,
        )
        stop_training = publish_training_checkpoint(
            checkpoint_label(epoch=epoch, epoch_fraction=1.0, epoch_complete=True),
            epoch=epoch,
            epoch_step=epoch_step,
            epoch_fraction=1.0,
            epoch_complete=True,
            cursor=epoch_complete_cursor,
        )
        resume_position["cursor"] = epoch_complete_cursor

    dist.barrier()
    if rank == 0:
        wall_seconds = cumulative_wall_now()
        emit_run_status("attempt_complete")
        atomic_write_json(
            out / "train_log.json",
            build_train_log_payload(
                arm_name=arm_name,
                lineage=lineage,
                run_id=run_id,
                training_seed=args.seed,
                epochs=epochs,
                cap_steps_per_epoch=cap_steps_per_epoch,
                total_steps=total_steps,
                world=world,
                accum_pos=accum_pos,
                effective_batch=effective_batch,
                lr=args.lr,
                warmup=args.warmup,
                mem_layers=args.mem_layers,
                alpha=ALPHA,
                wall_seconds=wall_seconds,
                gpu_hours=cumulative_gpu_hours(wall_seconds=wall_seconds, world_size=world),
                cache_identity=cache_identity,
                latest_resume_path=latest_resume_path,
                objective_mode=objective_mode,
            ),
        )
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, help="directory of teacher.forward_teacher outputs (*.npz)")
    parser.add_argument(
        "--cache-manifest",
        default=None,
        help="optional JSON written by train.cache_manifest; if omitted, the sorted *.npz files under --cache are scanned",
    )
    parser.add_argument("--cache-manifest-sha256", default=None, help="optional expected sha256 of --cache-manifest")
    parser.add_argument("--out", required=True, help="run directory; checkpoints are published under <out>/checkpoints")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--memory-init",
        default=DEFAULT_MEMORY_INIT,
        help="memory config source; also the initial weights when --mem_layers 0",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--arm")
    parser.add_argument("--lineage")
    parser.add_argument("--attempt-id")
    parser.add_argument("--resume-manifest")
    parser.add_argument("--stop-after-label")
    parser.add_argument("--telemetry-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mem_layers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fractional_save_points", default="")
    parser.add_argument("--later-fractional-save-points", default="")
    parser.add_argument(
        "--objective-mode",
        choices=OBJECTIVE_MODES,
        default="joint_base_anchored",
    )
    args = parser.parse_args()
    if args.world == 1:
        worker(0, 1, args)
    else:
        mp.spawn(worker, args=(args.world, args), nprocs=args.world, join=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
