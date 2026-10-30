"""CPU-only checks for the objective switch and resume-contract bookkeeping in train.pretrain."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from train.fractional_checkpoints import validate_resume_contract


def _load_pretrain():
    class _Dummy:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    if importlib.util.find_spec("transformers") is None:
        transformers = types.ModuleType("transformers")
        transformers.AutoConfig = _Dummy
        transformers.AutoModelForCausalLM = _Dummy
        transformers.AutoTokenizer = _Dummy
        transformers.get_linear_schedule_with_warmup = lambda *args, **kwargs: None
        sys.modules.setdefault("transformers", transformers)
        memory = types.ModuleType("resmem.memory")
        memory.MistralMLPModel = _Dummy
        sys.modules.setdefault("resmem.memory", memory)
    return importlib.import_module("train.pretrain")


pretrain = _load_pretrain()


def test_training_logits_follow_selected_objective() -> None:
    zbase = torch.tensor([[1.0, 2.0]])
    student_out = torch.tensor([[0.25, -0.5]])

    joint = pretrain.training_logits_for_objective(
        "joint_base_anchored", zbase=zbase, student_out=student_out
    )
    standalone = pretrain.training_logits_for_objective(
        "standalone_memory", zbase=zbase, student_out=student_out
    )

    assert torch.equal(joint, zbase + student_out)
    assert torch.equal(standalone, student_out)


def test_resume_contract_pins_objective_mode_and_rejects_resume_drift(tmp_path: Path) -> None:
    cache_manifest = tmp_path / "cache_manifest.json"
    cache_manifest.write_text("{}\n", encoding="utf-8")
    common = dict(
        run_id="run-123",
        out=tmp_path / "out",
        checkpoints_root=tmp_path / "out" / "checkpoints",
        cache_identity={
            "manifest_path": str(cache_manifest),
            "manifest_sha256": "a" * 64,
            "cache_identity": "cache-identity",
        },
        world=2,
        batch_size=32,
        effective_batch=1024,
        cap_steps_per_epoch=8,
        fractional_points=(0.125, 1.0),
        lr=5e-4,
        warmup=32,
        mem_layers=8,
        epochs=1,
        total_steps=8,
        arm_name="joint_base_anchored",
        lineage="residual",
        seed=42,
    )

    contract = pretrain.build_resume_contract(
        objective_mode="joint_base_anchored", **common
    )
    assert contract["objective_mode"] == "joint_base_anchored"

    resume_state = {"contract": dict(contract)}
    validate_resume_contract(resume_state, expected_contract=contract)

    with pytest.raises(ValueError, match="objective_mode"):
        validate_resume_contract(
            resume_state,
            expected_contract=pretrain.build_resume_contract(
                objective_mode="standalone_memory", **common
            ),
        )


def test_resume_contract_sidecar_is_created_reused_and_rejects_drift(
    tmp_path: Path,
) -> None:
    cache_manifest = tmp_path / "cache_manifest.json"
    cache_manifest.write_text("{}\n", encoding="utf-8")
    contract = pretrain.build_resume_contract(
        run_id="run-123",
        out=tmp_path / "out",
        checkpoints_root=tmp_path / "out" / "checkpoints",
        cache_identity={
            "manifest_path": str(cache_manifest),
            "manifest_sha256": "a" * 64,
            "cache_identity": "cache-identity",
        },
        world=2,
        batch_size=32,
        effective_batch=1024,
        cap_steps_per_epoch=8,
        fractional_points=(0.125, 1.0),
        lr=5e-4,
        warmup=32,
        mem_layers=8,
        epochs=1,
        total_steps=8,
        arm_name="joint_base_anchored",
        lineage="residual",
        seed=42,
        objective_mode="joint_base_anchored",
    )
    sidecar_path = tmp_path / "out" / "resume_contract.json"

    pretrain.publish_resume_contract_sidecar(
        sidecar_path=sidecar_path,
        resume_contract=contract,
    )
    published = pretrain.load_resume_contract_sidecar(sidecar_path)
    assert published == {"contract": contract}

    pretrain.publish_resume_contract_sidecar(
        sidecar_path=sidecar_path,
        resume_contract=contract,
    )
    assert pretrain.load_resume_contract_sidecar(sidecar_path) == {"contract": contract}

    drifted = {"contract": {**contract, "objective_mode": "standalone_memory"}}
    pretrain.atomic_write_json(sidecar_path, drifted)
    with pytest.raises(ValueError, match="resume contract sidecar drift"):
        pretrain.publish_resume_contract_sidecar(
            sidecar_path=sidecar_path,
            resume_contract=contract,
        )


def test_status_and_train_log_payloads_record_objective_mode(tmp_path: Path) -> None:
    status = pretrain.build_run_status_payload(
        status="attempt_running",
        run_id="run-123",
        training_seed=42,
        attempt_id="attempt-1",
        resume_from_step=0,
        latest_label="pass0p125",
        optimizer_step=8,
        epoch=0,
        epoch_step=8,
        epoch_fraction=0.125,
        seen_tokens=8192,
        wall_seconds=12.5,
        gpu_hours=0.01,
        world_size=2,
        effective_batch=1024,
        objective_mode="standalone_memory",
    )
    assert status["objective_mode"] == "standalone_memory"

    train_log = pretrain.build_train_log_payload(
        arm_name="standalone_memory",
        lineage="residual",
        run_id="run-123",
        training_seed=42,
        epochs=1,
        cap_steps_per_epoch=8,
        total_steps=8,
        world=2,
        accum_pos=512,
        effective_batch=1024,
        lr=5e-4,
        warmup=32,
        mem_layers=8,
        alpha=0.5,
        wall_seconds=12.5,
        gpu_hours=0.01,
        cache_identity={
            "manifest_path": tmp_path / "cache_manifest.json",
            "manifest_sha256": "a" * 64,
            "cache_identity": "cache-identity",
        },
        latest_resume_path=tmp_path / "latest_resume.json",
        objective_mode="standalone_memory",
    )
    assert train_log["objective_mode"] == "standalone_memory"
    assert train_log["arm"] == "standalone_memory"
