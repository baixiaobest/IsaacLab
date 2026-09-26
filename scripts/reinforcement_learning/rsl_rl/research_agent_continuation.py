"""Research Agent contract helpers for native RSL-RL training continuations.

This module deliberately has no Isaac Sim dependency.  The Research Agent
control plane can use it while planning a continuation, and ``train.py`` uses
the same checks immediately before restoring the staged checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


class ResumeValidationError(ValueError):
    """Raised when a requested training continuation is not safe to launch."""


@dataclass(frozen=True)
class NativeCheckpoint:
    """Validated native RSL-RL checkpoint metadata."""

    path: str
    sha256: str
    iteration: int


@dataclass(frozen=True)
class ResumeLineage:
    """Immutable provenance written into a continuation run directory."""

    parent_experiment_id: str
    parent_checkpoint_id: str | None
    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    parent_checkpoint_iteration: int
    task: str
    workflow: str
    target_branch: str
    target_commit: str
    seed: int
    additional_iterations: int


def is_native_rsl_rl_checkpoint(path: str | Path) -> bool:
    """Return whether *path* names a native checkpoint rather than an inference export."""
    checkpoint = Path(path)
    return checkpoint.suffix == ".pt" and not checkpoint.name.endswith("_jit.pt")


def sha256_file(path: str | Path) -> str:
    """Calculate a stable SHA-256 digest without loading the checkpoint into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_native_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    checkpoint_loader: Callable[[str], Any] | None = None,
) -> NativeCheckpoint:
    """Validate a staged native RSL-RL checkpoint before passing it to a runner.

    The final compatibility authority remains ``runner.load`` on the selected
    commit.  This inexpensive structural check makes artifact and JIT mistakes
    fail before Isaac Sim starts.
    """
    checkpoint = Path(path).expanduser().resolve()
    if not is_native_rsl_rl_checkpoint(checkpoint):
        raise ResumeValidationError("Resume checkpoints must be native RSL-RL .pt files, not JIT exports.")
    if not checkpoint.is_file():
        raise ResumeValidationError(f"Staged resume checkpoint does not exist: {checkpoint}")

    checksum = sha256_file(checkpoint)
    if expected_sha256 and checksum.lower() != expected_sha256.lower():
        raise ResumeValidationError(
            f"Resume checkpoint SHA-256 mismatch for {checkpoint}: expected {expected_sha256}, got {checksum}."
        )

    if checkpoint_loader is None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - only possible outside the Isaac runtime.
            raise ResumeValidationError("PyTorch is required to inspect a native RSL-RL checkpoint.") from error
        checkpoint_loader = lambda filename: torch.load(filename, map_location="cpu", weights_only=False)

    try:
        payload = checkpoint_loader(str(checkpoint))
    except Exception as error:
        raise ResumeValidationError(f"Unable to read staged resume checkpoint {checkpoint}: {error}") from error
    if not isinstance(payload, dict):
        raise ResumeValidationError("Resume checkpoint payload must be a native RSL-RL dictionary.")
    # ``model_state_dict`` is the older RSL-RL layout.  ``train.py`` delegates
    # its conversion to Isaac Lab's existing version-compatibility helper.
    native_layout = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict"}
    if not native_layout.issubset(payload) and "model_state_dict" not in payload:
        raise ResumeValidationError("Resume checkpoint is not a supported native RSL-RL training checkpoint.")
    iteration = payload.get("iter", 0)
    if not isinstance(iteration, int) or iteration < 0:
        raise ResumeValidationError("Resume checkpoint has an invalid RSL-RL iteration counter.")
    return NativeCheckpoint(path=str(checkpoint), sha256=checksum, iteration=iteration)


def resolve_branch_commit(repository: str | Path, branch: str, commit: str) -> str:
    """Resolve *commit* and require it to be reachable from the selected branch."""
    repo = Path(repository).resolve()
    if not branch or not commit:
        raise ResumeValidationError("A target branch and commit are required for a training continuation.")

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ResumeValidationError(f"Git validation failed: {detail}")
        return result.stdout.strip()

    resolved_commit = git("rev-parse", "--verify", f"{commit}^{{commit}}")
    # Accept a checked-out local branch or an origin-tracking branch.  Do not
    # concatenate shell strings; branch names are always argv values.
    references = (f"refs/heads/{branch}", f"refs/remotes/origin/{branch}")
    branch_ref = next((ref for ref in references if _git_ref_exists(repo, ref)), None)
    if branch_ref is None:
        raise ResumeValidationError(f"Target branch {branch!r} does not exist in {repo}.")
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", resolved_commit, branch_ref],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ResumeValidationError(f"Commit {resolved_commit} is not reachable from selected branch {branch!r}.")
    return resolved_commit


def assert_checkout_matches(repository: str | Path, expected_commit: str) -> None:
    """Require the worker checkout to be exactly the commit persisted at planning time."""
    result = subprocess.run(
        ["git", "-C", str(Path(repository).resolve()), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    actual_commit = result.stdout.strip()
    if result.returncode or actual_commit != expected_commit:
        raise ResumeValidationError(
            f"Worker source revision mismatch: expected {expected_commit}, found {actual_commit or 'unavailable'}."
        )


def write_resume_lineage(path: str | Path, lineage: ResumeLineage) -> None:
    """Write an immutable, JSON-serializable continuation record."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(lineage), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_ref_exists(repository: Path, reference: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(repository), "show-ref", "--verify", "--quiet", reference], check=False
        ).returncode
        == 0
    )
