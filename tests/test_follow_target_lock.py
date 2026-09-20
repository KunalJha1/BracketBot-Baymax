"""Identity regressions and physical-motion simulations; no hardware evidence."""

from dataclasses import replace
import math

import numpy as np
import pytest

from follow_core import FollowConfig, FollowLoop, LockOn, Perception, PersonObservation, Tracker, TickInputs, Pose2D, FOLLOWING, LOST, SEARCHING, hist_distance
from follow_perception import clothing_histogram, find_people
from robot_follow import perceive, point_colors
from test_follow_perception import standing_person
from test_follow_sim import Scenario, run


CFG = FollowConfig()
RED = clothing_histogram(np.tile(np.array([190, 35, 35], np.uint8), (50, 1)))
BLUE = clothing_histogram(np.tile(np.array([35, 35, 190], np.uint8), (50, 1)))


def obs(x=1.2, y=0, hist=RED):
    return PersonObservation(x, y, hist=hist)


def tick(loop, t, people):
    return loop.tick(TickInputs(t, 0, False, 0, 0, 0, 0,
                               Perception(t, tuple(people), np.empty((0, 3)))))


def test_initial_lock_does_not_pick_the_older_candidate_when_another_person_arrives():
    lock = LockOn(CFG)
    for i in range(5):
        assert lock.update(i * .1, [obs()], [(1.2, 0)]) is None
    assert lock.update(.5, [obs(), obs(1.4, .3)], [(1.2, 0), (1.4, .3)]) is None


def test_initial_lock_cannot_use_a_candidate_missing_from_the_current_frame():
    lock = LockOn(CFG)
    for i in range(5):
        assert lock.update(i * .1, [obs()], [(1.2, 0)]) is None
    assert lock.update(.5, [], []) is None


def test_initial_lock_requires_real_frames_not_a_long_timestamp_gap_or_duplicates():
    lock = LockOn(CFG)
    for t in [0, 0, 0, 1, 1, 1, 2, 2, 2]:
        assert lock.update(t, [obs()], [(1.2, 0)]) is None


def test_exact_nearest_match_does_not_hide_another_near_identical_candidate():
    tracker = Tracker(CFG)
    tracker.start(0, (1.2, 0), RED)
    assert tracker.update(.1, [obs(), obs(1.2, .03)], [(1.2, 0), (1.2, .03)]) == "ambiguous"
    assert tracker.uncertain
    assert tracker.last_update == 0


def test_clothing_cue_selects_target_even_when_bystander_is_closer_to_prediction():
    tracker = Tracker(CFG)
    tracker.start(0, (1.2, 0), RED)
    people = [obs(hist=BLUE), obs(1.22, .04)]
    assert tracker.update(.1, people, [(1.2, 0), (1.22, .04)]) == "updated"
    assert tracker.kf.s[1] > 0
    np.testing.assert_array_equal(tracker.ref_hist, RED)


def test_missing_colour_never_silently_weakens_an_existing_identity_match():
    tracker = Tracker(CFG)
    tracker.start(0, (1.2, 0), RED)
    assert tracker.update(.1, [obs(hist=None)], [(1.2, 0)]) == "coasted"
    assert tracker.last_update == 0


def test_reacquisition_needs_several_consistent_observations_with_original_colour():
    tracker = Tracker(CFG)
    tracker.start(0, (1.2, 0), RED)
    tracker.mark_lost()
    assert tracker.update(2, [obs(hist=BLUE)], [(1.2, 0)]) == "coasted"
    for t in (2.1, 2.2, 2.3):
        assert tracker.update(t, [obs()], [(1.2, 0)]) == "confirming"
        assert tracker.last_update == 0
    assert tracker.update(2.4, [obs()], [(1.2, 0)]) == "updated"


def test_long_loss_never_auto_locks_a_new_person_even_if_they_match_colour():
    loop = FollowLoop(CFG)
    for i in range(11):
        out = tick(loop, i * .1, [obs()])
    assert out.state == FOLLOWING
    for i in range(11, 160):
        out = tick(loop, i * .1, [])
    for i in range(160, 180):
        out = tick(loop, i * .1, [obs()])
        assert out.state == LOST
        assert out.association == "restart-required"
        assert out.v == out.omega == 0


def test_relock_after_loss_searches_again_and_follows_whoever_stands_in_front():
    loop = FollowLoop(replace(CFG, relock_after_loss=True))
    for i in range(11):
        out = tick(loop, i * .1, [obs()])
    assert out.state == FOLLOWING
    for i in range(11, 160):
        out = tick(loop, i * .1, [])
        assert out.v == out.omega == 0 or out.state != SEARCHING
    assert out.state == SEARCHING
    for i in range(160, 180):
        out = tick(loop, i * .1, [obs(hist=BLUE)])
    assert out.state == FOLLOWING


def test_geometry_only_does_not_switch_when_one_person_disappears_after_an_ambiguous_crossing():
    tracker = Tracker(CFG)
    tracker.start(0, (1.2, 0), None)
    assert tracker.update(.1, [obs(hist=None), obs(1.2, .03, None)], [(1.2, 0), (1.2, .03)]) == "ambiguous"
    assert tracker.update(.2, [obs(hist=None)], [(1.2, 0)]) == "identity-required"


def test_colours_stay_aligned_after_body_mask_and_person_clustering():
    first = standing_person(1.2, -.4)
    second = standing_person(1.5, .5)
    local = np.vstack([first, second])
    base = np.column_stack([-local[:, 1], local[:, 0], local[:, 2]])
    colors = np.vstack([np.tile(np.array([190, 35, 35], np.uint8), (len(first), 1)),
                        np.tile(np.array([35, 35, 190], np.uint8), (len(second), 1))])
    from follow_calibration import Calibration
    frame = perceive(base, 0, colors=colors, calibration=Calibration(self_mask=((.3, 1.4, -.7, -.1, 0, 2),)))
    assert len(frame.people) == 1
    np.testing.assert_allclose(frame.people[0].hist, BLUE)


def test_unknown_colour_format_is_not_guessed():
    points = standing_person(1.2)
    assert find_people(points, colors=np.ones_like(points))[0].hist is None
    assert point_colors({"colors": np.ones((10, 3))}, 10) is None


def test_modest_exposure_change_does_not_break_the_same_clothing_cue():
    brighter = clothing_histogram(np.tile(np.array([209, 39, 39], np.uint8), (50, 1)))
    assert hist_distance(RED, brighter) < CFG.hist_max_distance
    assert hist_distance(RED, BLUE) > CFG.hist_max_distance


def test_target_angular_feedforward_compensates_for_robot_translation():
    tracker = Tracker(CFG)
    tracker.start(0, (1, 1), RED)
    tracker.kf.s[2:] = [0, .4]
    track = tracker.track(0, Pose2D(), robot_v=.2)
    assert track.omega_feedforward == pytest.approx(.3)


@pytest.mark.parametrize("seed", [0, 7, 19])
def test_rotate_only_follows_a_continuous_arc_at_ten_hz(seed):
    def path(t):
        angle = .35 * max(0, t - 2)
        return 1.2 * math.cos(angle), 1.2 * math.sin(angle)

    result = run(Scenario(path, 12, seed=seed, frame_period=.1, appearance=True),
                 cfg=replace(CFG, v_max=0))
    settled = result.samples(4)
    assert all(o.state == FOLLOWING for *_, o in settled)
    assert all(o.v == 0 for o in result.out)
    assert np.percentile([abs(math.degrees(b)) for _, _, b, _ in settled], 95) < 10
    assert not any(o.exit for o in result.out)


def test_slow_sideways_walk_remains_locked_while_robot_turns_and_moves():
    result = run(Scenario(lambda t: (1.3 + .08 * max(0, t - 2), .12 * max(0, t - 2)),
                          14, frame_period=.1, appearance=True), cfg=CFG)
    assert all(o.state == FOLLOWING for *_, o in result.samples(3))
    assert max(abs(math.degrees(b)) for _, _, b, _ in result.samples(4)) < 10


def test_brief_detection_dropout_recovers_with_colour_and_never_changes_target():
    result = run(Scenario(lambda t: (1.2, .08 * max(0, t - 2)), 10,
                          visible=lambda t: not 4 <= t < 4.4, appearance=True, frame_period=.1),
                 cfg=replace(CFG, v_max=0))
    assert all(o.state == FOLLOWING for *_, o in result.samples(6))
    assert result.out[-1].association == "updated"
