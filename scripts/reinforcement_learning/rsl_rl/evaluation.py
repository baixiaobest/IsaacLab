# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility imports for the navigation evaluator.

Implementations live in :mod:`navigation_evaluation`. Existing script imports stay valid.
"""

from navigation_evaluation.common import (
    SCENARIO_ORDER,
    SCENARIO_LABELS,
    STATIC_OBSTACLE_SCENARIO,
    ROOM_CORRIDOR_SCENARIOS,
    EVALUATION_LEVEL_COUNTS,
    GOAL_REGION_COLLISION_RADIUS_M,
    GOAL_REGION_TAG,
    INTERESTING_INTERACTION_TAG,
    SUCCESS_INTERACTION_DISTANCE_M,
    INTERACTION_ENTER_CLEARANCE_M,
    INTERACTION_EXIT_CLEARANCE_M,
    INTERACTION_RISK_CLEARANCE_M,
    INTERACTION_RISK_HORIZON_S,
    INTERACTION_PRE_SPEED_WINDOW_S,
    INTERACTION_MIN_DURATION_S,
    INTERACTION_MIN_BASELINE_SPEED_MPS,
    INTERACTION_YIELD_SPEED_RATIO,
    INTERACTION_ASSERT_SPEED_RATIO,
    INTERACTION_OVERTAKE_LONGITUDINAL_MARGIN_M,
    INTERACTION_FRONT_CROSS_CLEARANCE_MARGIN_M,
    INTERACTION_FRONT_CROSS_LATERAL_HYSTERESIS_M,
    INTERACTION_AGAINST_FLOW_ENCOUNTER_RANGE_M,
    INTERACTION_AGAINST_FLOW_CONE_HALF_ANGLE_RAD,
    INTERACTION_AGAINST_FLOW_CONE_LATERAL_BUFFER_M,
    INTERACTION_AGAINST_FLOW_CONE_PERSISTENCE_S,
    INTERACTION_AGAINST_FLOW_SIDESTEP_LATERAL_M,
    INTERACTION_AGAINST_FLOW_REAR_PASS_MARGIN_M,
    INTERACTION_LABELS,
    INTERACTION_SCENARIO_ORDER,
    LEADER_OUTCOME_SCENARIOS,
    LEADER_SLOT,
    LEADER_OVERTAKE_MARGIN_M,
    LEADER_OUTCOME_LABELS,
    SLOW_LEADER_SCENARIO,
    SLOW_LEADER_SLOT,
    SLOW_LEADER_OVERTAKE_MARGIN_M,
    SLOW_LEADER_OUTCOME_LABELS,
    BenchmarkProfile,
    _json_safe,
    _write_json_atomically,
    terminal_goal_region_collision_ids,
    terminal_collision_ids,
    _sample_standard_deviation,
)
from navigation_evaluation.profiles import (
    dynamic_crowd_profiles,
    room_corridor_profiles,
    select_scenario_profiles,
    indoor_dynamic_profiles,
    fixed_grid_profile_indices,
)
from navigation_evaluation.metrics import (
    _flat_list,
    _ids,
    completed_environment_ids,
    EpisodeVelocityAccumulator,
    EpisodeMetricsCollector,
    _result_row,
    _aggregate_rows_from_counts,
)
from navigation_evaluation.replays import (
    CollisionReplayRecorder,
    InteractionEventReplayRecorder,
)
from navigation_evaluation.interactions import (
    classify_speed_interaction,
    interaction_speed_diagnostics,
    classify_crossing_interaction,
    front_crossing_longitudinal_m,
    rear_crossing_longitudinal_m,
    front_lateral_side,
    classify_against_flow_interaction,
    InteractionEventCollector,
    LeaderOutcomeCollector,
    SlowLeaderOutcomeCollector,
)
from navigation_evaluation.artifacts import (
    print_results,
    save_artifacts,
    save_interaction_event_artifacts,
    save_leader_outcome_artifacts,
    print_leader_outcomes,
    _save_interaction_histogram,
    _save_leader_outcome_plot,
    save_slow_leader_outcome_artifacts,
    print_slow_leader_outcomes,
    _save_summary_plot,
    _save_failure_histogram,
)
