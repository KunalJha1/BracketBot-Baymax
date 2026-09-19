"""Regression tests for the fall-detection concept check.

``ground_safety.assess_ground_pose`` shipped without test coverage.  These tests
pin the behaviour the concept check measured, so a threshold change cannot
silently turn a cleared posture into an alert.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "bbapps" / "emotion_greeter"))

from fall_concept_check import (  # noqa: E402
    FALLEN_SUPINE,
    LYING_SOFA,
    SEGMENT_LIMITS,
    build_scenarios,
    depth_path,
    horizontal_pose,
    implausible_segment,
    monocular_path,
    project,
)


def scenarios_by_name():
    return {name: (truth, joints) for name, truth, joints in build_scenarios()}


def test_every_on_floor_scenario_is_detected_by_both_paths():
    rng = np.random.default_rng(7)
    for name, should_alert, joints in build_scenarios():
        if not should_alert:
            continue
        depth_assessment, _visible = depth_path(joints, rng)
        mono_assessment, _pixels = monocular_path(joints)
        assert depth_assessment.suspected, f"depth path missed {name}"
        assert mono_assessment.suspected, f"monocular path missed {name}"


@pytest.mark.parametrize(
    "name",
    ["standing_2m", "standing_3.5m", "sitting_chair_2.5m", "crouching_2m", "bending_over_2.2m"],
)
def test_ordinary_upright_postures_stay_clear(name):
    rng = np.random.default_rng(7)
    _truth, joints = scenarios_by_name()[name]
    depth_assessment, _visible = depth_path(joints, rng)
    mono_assessment, _pixels = monocular_path(joints)
    assert not depth_assessment.suspected
    assert not mono_assessment.suspected


def test_standing_clears_by_a_wide_margin_not_a_hair():
    """A safety threshold that only just clears standing is not tuned, it is lucky."""
    rng = np.random.default_rng(7)
    _truth, joints = scenarios_by_name()["standing_2m"]
    assessment, _visible = depth_path(joints, rng)
    assert assessment.confidence < 0.3


def test_person_on_a_sofa_is_not_reported_as_being_on_the_ground():
    """The monocular floor-contact assumption fails here; it must abstain, not alert."""
    _truth, joints = scenarios_by_name()["lying_on_sofa_2.5m"]
    assessment, _pixels = monocular_path(joints)
    assert assessment.state == "unknown"
    assert "implausible" in assessment.reason


def test_scale_gate_rejects_an_inflated_body_and_accepts_a_real_one():
    sofa = {}
    fallen = {}
    for index, values in LYING_SOFA.items():
        sofa[index] = np.array([values[1], 2.5 - values[0], values[2]])
    for index, values in FALLEN_SUPINE.items():
        fallen[index] = np.array([values[1], 2.5 - values[0], values[2]])

    assert implausible_segment(fallen) is None
    inflated = {index: point * 1.45 for index, point in fallen.items()}
    assert implausible_segment(inflated) is not None


def test_segment_limits_are_above_real_adult_anatomy():
    """The gate must never reject a genuine fall, so limits stay generous."""
    for (first, second), limit in SEGMENT_LIMITS.items():
        assert 0.4 < limit < 0.7, f"segment {first}-{second} limit {limit} is not plausible"


def test_occluded_lower_body_still_alerts():
    """Legs behind furniture is the common real case; upper body must be enough."""
    rng = np.random.default_rng(7)
    _truth, joints = scenarios_by_name()["fallen_prone_occluded"]
    assessment, _visible = depth_path(joints, rng)
    assert assessment.suspected
    assert assessment.depth_keypoints >= 4


def test_projection_reproduces_the_documented_camera_geometry():
    """docs/robot-facts.md: 33 deg down from 1.55 m puts the image centre about
    2.4 m ahead on the floor.  The projection here must independently agree."""
    from fall_concept_check import CX, CY

    pixel = project(np.array([0.0, 2.4, 0.0]))
    assert pixel is not None
    u, v = pixel
    assert u == pytest.approx(CX, abs=2.0)
    assert v == pytest.approx(CY, abs=3.0)


def test_floor_nearer_than_the_centre_distance_images_lower():
    from fall_concept_check import CY

    near = project(np.array([0.0, 1.2, 0.0]))
    assert near is not None
    assert near[1] > CY, "floor closer than 2.4 m should image below centre"


def test_a_point_behind_the_robot_is_not_projected():
    assert project(np.array([0.0, -2.0, 0.5])) is None
