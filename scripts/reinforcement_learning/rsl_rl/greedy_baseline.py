# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Backward-compatible re-export.

``GreedyLidarController`` now lives in ``isaaclab_tasks`` (the installed package) so that the
in-environment ``GreedyPreTrainedPolicyAction`` action term can import it too, without depending on
this ``scripts/`` directory being on ``sys.path``. This shim keeps ``evaluate_baseline.py``'s
``from greedy_baseline import GreedyLidarController`` working unchanged.
"""

from __future__ import annotations

from isaaclab_tasks.manager_based.navigation.mdp.reactive_controllers import GreedyLidarController  # noqa: F401
