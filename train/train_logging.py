"""Per-step training metric records appended durably as JSONL."""

from datetime import datetime, timezone

from train.checkpointing import append_jsonl_durable


def loss_record(
    optimizer_step,
    epoch,
    ce,
    kl,
    loss,
    lr,
    tokens_per_s,
    *,
    epoch_fraction=0.0,
    seen_tokens=0,
    wall_seconds=0.0,
    gpu_hours=0.0,
    world_size=1,
    attempt_id=None,
    resume_from_step=None,
    timestamp=None,
):
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "step": optimizer_step,
        "optimizer_step": optimizer_step,
        "epoch": epoch,
        "epoch_fraction": epoch_fraction,
        "seen_tokens": seen_tokens,
        "ce": ce,
        "kl": kl,
        "loss": loss,
        "lr": lr,
        "tokens_per_s": tokens_per_s,
        "wall_seconds": wall_seconds,
        "gpu_hours": gpu_hours,
        "world_size": world_size,
        "attempt_id": attempt_id,
        "resume_from_step": resume_from_step,
        "timestamp": timestamp,
    }


def append_jsonl(path, record):
    return append_jsonl_durable(path, record)
