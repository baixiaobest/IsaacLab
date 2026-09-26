"""Pure contract tests for Research Agent native RSL-RL continuation inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from research_agent_continuation import (  # noqa: E402
    ResumeLineage,
    ResumeValidationError,
    inspect_native_checkpoint,
    sha256_file,
    write_resume_lineage,
)


def test_native_checkpoint_validation_rejects_jit_and_records_iteration(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_42.pt"
    checkpoint.write_bytes(b"native-rsl-rl")
    inspected = inspect_native_checkpoint(
        checkpoint,
        expected_sha256=sha256_file(checkpoint),
        checkpoint_loader=lambda _path: {
            "actor_state_dict": {}, "critic_state_dict": {}, "optimizer_state_dict": {}, "iter": 42,
        },
    )
    assert inspected.iteration == 42
    assert inspected.sha256 == sha256_file(checkpoint)

    jit = tmp_path / "policy_jit.pt"
    jit.write_bytes(b"jit")
    with pytest.raises(ResumeValidationError, match="not JIT"):
        inspect_native_checkpoint(jit, checkpoint_loader=lambda _path: {})


def test_resume_lineage_is_immutable_json_input(tmp_path: Path) -> None:
    path = tmp_path / "params" / "resume_lineage.json"
    write_resume_lineage(
        path,
        ResumeLineage(
            parent_experiment_id="parent", parent_checkpoint_id=None, parent_checkpoint_path="/staged/model.pt",
            parent_checkpoint_sha256="a" * 64, parent_checkpoint_iteration=42,
            task="Isaac-Locomotion-Vel-Unitree-Go2-Robust-v1", workflow="rsl_rl",
            target_branch="main", target_commit="deadbeef", seed=17, additional_iterations=2000,
        ),
    )
    content = path.read_text(encoding="utf-8")
    assert '"parent_experiment_id": "parent"' in content
    assert '"target_commit": "deadbeef"' in content
