# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Best-effort Research Agent progress reporting for navigation evaluation."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

class EvaluationProgressReporter:
    """Best-effort live evaluation telemetry for the Research Agent UI.

    Evaluation is deliberately independent of W&B: a telemetry outage must
    never slow down or fail a benchmark.  The reporter writes flat summary
    keys so the control plane can read them from the normal run summary while
    the Pod is still executing.
    """

    _INTERVAL_SECONDS = 30.0
    _PREFIX = "research_agent_evaluation_"

    def __init__(self, *, profile_count: int, episodes_per_profile: int, seed_count: int):
        self.total_episodes = profile_count * episodes_per_profile
        self.profile_count = profile_count
        self.episodes_per_profile = episodes_per_profile
        self.seed_count = seed_count
        self.started_at = time.monotonic()
        self.last_report_at = 0.0
        self.run = None
        experiment_id = os.environ.get("RESEARCH_EXPERIMENT_ID")
        project = os.environ.get("WANDB_PROJECT")
        # Each newly submitted evaluation owns an isolated W&B run.  Keep the
        # durable remote-attempt identity in every update so the control plane
        # never mistakes an earlier attempt's telemetry for this one.
        self.evaluation_attempt_id = os.environ.get("RESEARCH_AGENT_EVALUATION_ATTEMPT_ID")
        self.bootstrap_id = os.environ.get("RESEARCH_AGENT_BOOTSTRAP_ID")
        # The bootstrap switches WANDB_RUN_ID before starting the evaluator,
        # but prefer the explicit value as well: it makes this reporter robust
        # when it is launched outside that shell wrapper.  Falling back to the
        # experiment ID preserves the legacy shared-run behavior for old Pods.
        self.wandb_run_id = (
            os.environ.get("RESEARCH_AGENT_EVALUATION_WANDB_RUN_ID")
            or os.environ.get("WANDB_RUN_ID")
            or experiment_id
        )
        if not experiment_id or not project:
            return
        try:
            import wandb

            self.run = wandb.init(
                project=project,
                entity=os.environ.get("WANDB_ENTITY") or None,
                id=self.wandb_run_id,
                resume="allow",
            )
        except Exception as error:
            print(f"[WARN] Research Agent evaluation telemetry disabled: {error}", flush=True)

    def report(
        self,
        accepted_episodes: int,
        *,
        seed: int | None,
        seed_index: int | None,
        status: str,
        force: bool = False,
    ) -> None:
        """Publish a throttled progress snapshot without affecting evaluation."""
        now = time.monotonic()
        if not force and now - self.last_report_at < self._INTERVAL_SECONDS:
            return
        self.last_report_at = now
        elapsed_seconds = max(0.0, now - self.started_at)
        rate = accepted_episodes / elapsed_seconds if elapsed_seconds else 0.0
        remaining_episodes = max(0, self.total_episodes - accepted_episodes)
        remaining_seconds = remaining_episodes / rate if rate > 0.0 else None
        percent = round(100.0 * accepted_episodes / self.total_episodes, 1) if self.total_episodes else 100.0
        eta = f"{remaining_seconds:.0f}s" if remaining_seconds is not None else "n/a"
        seed_desc = f"seed {seed} ({seed_index}/{self.seed_count})" if seed is not None else "finalizing"
        # Mirror the throttled snapshot to stdout: the Pod console stream is
        # captured by the Research Agent pod-log logger, so live evaluation
        # progress is visible even if W&B telemetry is unavailable.
        print(
            f"[EVAL] {status}: {accepted_episodes}/{self.total_episodes} episodes ({percent}%), "
            f"{seed_desc}, elapsed {elapsed_seconds:.0f}s, ETA {eta}",
            flush=True,
        )
        if self.run is None:
            return
        snapshot = {
            f"{self._PREFIX}status": status,
            f"{self._PREFIX}accepted_episodes": int(accepted_episodes),
            f"{self._PREFIX}total_episodes": int(self.total_episodes),
            f"{self._PREFIX}percent": round(100.0 * accepted_episodes / self.total_episodes, 1)
            if self.total_episodes
            else 100.0,
            f"{self._PREFIX}current_seed": seed,
            f"{self._PREFIX}seed_index": seed_index,
            f"{self._PREFIX}seed_count": self.seed_count,
            f"{self._PREFIX}profile_count": self.profile_count,
            f"{self._PREFIX}episodes_per_profile": self.episodes_per_profile,
            f"{self._PREFIX}elapsed_seconds": round(elapsed_seconds, 1),
            f"{self._PREFIX}estimated_remaining_seconds": round(remaining_seconds, 1)
            if remaining_seconds is not None
            else None,
            f"{self._PREFIX}updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if self.evaluation_attempt_id:
            snapshot["research_agent_evaluation_attempt_id"] = self.evaluation_attempt_id
        if self.bootstrap_id:
            snapshot["research_agent_bootstrap_id"] = self.bootstrap_id
        try:
            self.run.summary.update(snapshot)
            # ``log`` causes W&B to sync the just-updated summary during a
            # long-running evaluation, rather than only when the process exits.
            self.run.log(snapshot, commit=True)
        except Exception as error:
            # Do not repeatedly add expensive failed network calls to a live
            # evaluation. A later W&B reconnect is handled by its own client.
            print(f"[WARN] Research Agent evaluation telemetry update skipped: {error}", flush=True)
            self.run = None

    def close(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as error:
            print(f"[WARN] Research Agent evaluation telemetry shutdown skipped: {error}", flush=True)
        finally:
            self.run = None


