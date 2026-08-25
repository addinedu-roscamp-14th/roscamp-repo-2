from tire_vision.wear_measurement import (
    TireDecisionState,
    decision_state_args_from_config,
)


def make_state():
    return TireDecisionState(
        window_size=15,
        min_valid_frames=10,
        bad_max_ratio=0.28,
        good_min_ratio=0.32,
        max_ratio_std=0.02,
        no_tire_reset_frames=3,
        marker_hold_frames=3,
        marker_recheck_frames=5,
    )


def add_ratios(state, ratio, count):
    snapshots = []
    for _ in range(count):
        state.update_tire_detection(True)
        snapshots.append(
            state.update_marker_detection(
                True, ratio=ratio, gap_px=ratio * 100, radius_px=100
            )
        )
    return snapshots


def test_good_tire_collects_then_stabilizes():
    state = make_state()
    snapshots = add_ratios(state, 0.431, 10)

    assert all(item["result"] == "COLLECTING" for item in snapshots[:9])
    assert snapshots[-1]["result"] == "GOOD"
    assert snapshots[-1]["stable_result"] == "GOOD"
    assert snapshots[-1]["valid_count"] == 10


def test_three_no_tire_frames_clear_every_decision_value():
    state = make_state()
    add_ratios(state, 0.431, 10)

    first = state.update_tire_detection(False)
    second = state.update_tire_detection(False)
    cleared = state.update_tire_detection(False)

    assert first["result"] == second["result"] == "GOOD"
    assert first["current_tire"] == "LOST"
    assert first["gap_px"] is None
    assert cleared["result"] == "NO_TIRE"
    assert cleared["valid_count"] == 0
    assert cleared["stable_result"] is None
    assert cleared["median_ratio"] is None
    assert cleared["std_ratio"] is None
    assert cleared["current_marker"] == "UNKNOWN"
    assert cleared["marker_miss_count"] == 0
    assert cleared["error_code"] == "NO_TIRE"


def test_new_bad_tire_never_reuses_previous_good_history():
    state = make_state()
    add_ratios(state, 0.431, 10)
    old_session = state.session_id
    for _ in range(3):
        state.update_tire_detection(False)

    first_bad = add_ratios(state, 0.18, 1)[0]
    assert state.session_id == old_session + 1
    assert first_bad["result"] == "COLLECTING"
    assert first_bad["valid_count"] == 1
    assert first_bad["stable_result"] is None
    assert list(state.ratio_history) == [0.18]

    final_bad = add_ratios(state, 0.18, 9)[-1]
    assert final_bad["result"] == "BAD"
    assert final_bad["std_ratio"] == 0.0


def test_marker_hold_does_not_append_false_valid_frames():
    state = make_state()
    add_ratios(state, 0.431, 10)

    for miss_count in range(1, 4):
        held = state.update_marker_detection(False)
        assert held["result"] == "GOOD"
        assert held["stable_result"] == "GOOD"
        assert held["marker_miss_count"] == miss_count
        assert held["valid_count"] == 10
        assert held["gap_px"] is None
        assert held["radius_px"] is None
        assert held["gap_ratio"] is None

    for miss_count in range(4, 6):
        recheck = state.update_marker_detection(False)
        assert recheck["result"] == "RECHECK"
        assert recheck["marker_miss_count"] == miss_count
        assert recheck["valid_count"] == 10

    error = state.update_marker_detection(False)
    assert error["result"] == "ERROR"
    assert error["error_code"] == "MARKER_LOST"
    assert error["valid_count"] == 10


def test_manual_or_request_reset_clears_complete_session():
    state = make_state()
    add_ratios(state, 0.18, 10)
    state.update_marker_detection(False)
    old_session = state.session_id

    reset = state.reset_session()

    assert reset["session_id"] == old_session + 1
    assert reset["result"] == "NO_TIRE"
    assert reset["current_tire"] == "UNKNOWN"
    assert reset["current_marker"] == "UNKNOWN"
    assert reset["valid_count"] == 0
    assert reset["stable_result"] is None
    assert reset["marker_miss_count"] == 0
    assert reset["no_tire_count"] == 0
    assert reset["retry_count"] == 0
    assert reset["median_ratio"] is None
    assert reset["std_ratio"] is None


def test_boundary_and_high_std_results_are_recheck():
    state = make_state()
    add_ratios(state, 0.30, 10)
    assert state.snapshot()["result"] == "RECHECK"

    state.reset_session()
    for ratio in [0.28, 0.34] * 5:
        state.update_tire_detection(True)
        state.update_marker_detection(True, ratio=ratio)
    snapshot = state.snapshot()
    assert snapshot["std_ratio"] > 0.02
    assert snapshot["result"] == "RECHECK"


def test_side_threshold_config_overrides_legacy_flat_defaults():
    config = {
        "bad_max_ratio": 0.28,
        "good_min_ratio": 0.32,
        "decision_thresholds": {
            "left": {"bad_max_ratio": 0.27, "good_min_ratio": 0.29},
            "right": {"bad_max_ratio": 0.30, "good_min_ratio": 0.35},
        },
    }

    left = decision_state_args_from_config(config, "left")
    right = decision_state_args_from_config(config, "right")

    assert (left["bad_max_ratio"], left["good_min_ratio"]) == (0.27, 0.29)
    assert (right["bad_max_ratio"], right["good_min_ratio"]) == (0.30, 0.35)


def camera_config():
    return {
        "left_camera": {
            "bad_max_ratio": 0.28,
            "good_min_ratio": 0.32,
            "max_ratio_std": 0.03,
            "marker_hold_frames": 5,
            "marker_recheck_frames": 8,
            "window_size": 15,
            "min_valid_frames": 10,
        },
        "right_camera": {
            "bad_max_gap_px": 10.5,
            "good_min_gap_px": 11.5,
            "bad_max_ratio": 0.25,
            "good_min_ratio": 0.27,
            "max_ratio_std": 0.035,
            "marker_hold_frames": 5,
            "marker_recheck_frames": 8,
            "window_size": 15,
            "min_valid_frames": 10,
        },
    }


def classify_camera_ratio(side, ratio):
    state = TireDecisionState(**decision_state_args_from_config(camera_config(), side))
    return add_ratios(state, ratio, 10)[-1]


def test_measured_left_and_right_samples_use_camera_thresholds():
    assert classify_camera_ratio("left", 0.400)["result"] == "GOOD"
    assert classify_camera_ratio("left", 0.288)["result"] == "RECHECK"
    assert classify_camera_ratio("right", 0.24)["result"] == "BAD"
    assert classify_camera_ratio("right", 0.425)["result"] == "GOOD"


def test_right_uses_gap_or_ratio_for_bad_and_both_for_good():
    cases = [
        (10.5, 0.40, "BAD"),
        (20.0, 0.25, "BAD"),
        (11.5, 0.27, "GOOD"),
        (11.0, 0.26, "RECHECK"),
        (12.0, 0.26, "RECHECK"),
    ]

    for gap_px, ratio, expected in cases:
        state = TireDecisionState(
            **decision_state_args_from_config(camera_config(), "right")
        )
        for _ in range(10):
            state.update_tire_detection(True)
            snapshot = state.update_marker_detection(
                True, ratio=ratio, gap_px=gap_px, radius_px=100
            )
        assert snapshot["result"] == expected


def test_right_ratio_std_overrides_good_gap_and_ratio():
    state = TireDecisionState(
        **decision_state_args_from_config(camera_config(), "right")
    )
    for ratio in [0.33, 0.41] * 5:
        state.update_tire_detection(True)
        snapshot = state.update_marker_detection(
            True, ratio=ratio, gap_px=16.0, radius_px=100
        )
    assert snapshot["std_ratio"] > 0.035
    assert snapshot["result"] == "RECHECK"


def test_left_keeps_ratio_only_behavior_without_gap_thresholds():
    state = TireDecisionState(
        **decision_state_args_from_config(camera_config(), "left")
    )
    for _ in range(10):
        state.update_tire_detection(True)
        snapshot = state.update_marker_detection(
            True, ratio=0.40, gap_px=1.0, radius_px=100
        )
    assert state.uses_gap_thresholds is False
    assert snapshot["result"] == "GOOD"


def test_right_gap_history_misses_and_reset_do_not_reuse_samples():
    state = TireDecisionState(
        **decision_state_args_from_config(camera_config(), "right")
    )
    for _ in range(10):
        state.update_tire_detection(True)
        snapshot = state.update_marker_detection(
            True, ratio=0.40, gap_px=16.0, radius_px=100
        )
    valid_count = snapshot["valid_count"]
    state.update_marker_detection(False)
    assert len(state.gap_history) == valid_count
    assert state.snapshot()["valid_count"] == valid_count
    reset = state.reset_session()
    assert len(state.gap_history) == 0
    assert reset["median_gap_px"] is None
    assert reset["stable_result"] is None


def test_configured_marker_miss_boundaries_hold_recheck_then_error():
    state = TireDecisionState(
        **decision_state_args_from_config(camera_config(), "left")
    )
    add_ratios(state, 0.400, 10)

    for miss_count in range(1, 6):
        snapshot = state.update_marker_detection(False)
        assert snapshot["result"] == "GOOD"
        assert snapshot["marker_miss_count"] == miss_count
        assert snapshot["valid_count"] == 10

    for miss_count in range(6, 9):
        snapshot = state.update_marker_detection(False)
        assert snapshot["result"] == "RECHECK"
        assert snapshot["stable_result"] == "GOOD"
        assert snapshot["marker_miss_count"] == miss_count
        assert snapshot["valid_count"] == 10

    snapshot = state.update_marker_detection(False)
    assert snapshot["result"] == "ERROR"
    assert snapshot["error_code"] == "MARKER_LOST"
    assert snapshot["marker_miss_count"] == 9
    assert snapshot["valid_count"] == 10


def test_right_camera_uses_0035_std_limit():
    state = TireDecisionState(
        **decision_state_args_from_config(camera_config(), "right")
    )
    for ratio in [0.36, 0.428] * 5:
        state.update_tire_detection(True)
        within_limit = state.update_marker_detection(
            True, ratio=ratio, gap_px=16.0
        )

    assert within_limit["std_ratio"] < 0.035
    assert within_limit["result"] == "GOOD"

    state.reset_session()
    for ratio in [0.35, 0.43] * 5:
        state.update_tire_detection(True)
        over_limit = state.update_marker_detection(
            True, ratio=ratio, gap_px=16.0
        )

    assert over_limit["std_ratio"] > 0.035
    assert over_limit["result"] == "RECHECK"


def test_left_measured_bad_and_reset_do_not_reuse_previous_good():
    state = TireDecisionState(
        **decision_state_args_from_config(camera_config(), "left")
    )
    add_ratios(state, 0.400, 10)
    assert state.snapshot()["stable_result"] == "GOOD"

    reset = state.reset_session()
    assert reset["stable_result"] is None
    assert reset["valid_count"] == 0
    assert reset["median_ratio"] is None
    assert reset["std_ratio"] is None

    snapshots = []
    for ratio in [0.267, 0.279] * 5:
        state.update_tire_detection(True)
        snapshots.append(state.update_marker_detection(True, ratio=ratio))

    assert snapshots[0]["result"] == "COLLECTING"
    assert snapshots[0]["stable_result"] is None
    assert round(snapshots[-1]["median_ratio"], 3) == 0.273
    assert round(snapshots[-1]["std_ratio"], 3) == 0.006
    assert snapshots[-1]["result"] == "BAD"
    assert snapshots[-1]["stable_result"] == "BAD"
