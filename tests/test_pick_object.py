import threading
import time

import numpy as np
import pytest

from scripts.pick_object import MIN_PHASE_SECONDS, phase_seconds, pose_at


def ramp(rotary_turns, lift_turns=0.0, count=50):
    poses = np.zeros((count, 8))
    poses[:, 0] = np.linspace(0.0, lift_turns, count)
    poses[:, 3] = np.linspace(0.0, rotary_turns, count)
    return poses


def test_short_move_runs_at_the_floor_not_the_ceiling():
    assert phase_seconds(ramp(0.02), 3.0, MIN_PHASE_SECONDS["descend"]) == 1.5


def test_long_move_is_never_slower_or_faster_than_the_ceiling():
    assert phase_seconds(ramp(0.9, lift_turns=1.0), 6.0, 3.0) == 6.0


def test_mid_move_keeps_the_peak_joint_speed():
    # 0.25 turns at a 0.15 turn/s smoothstep peak (1.5x mean) needs 2.5 s.
    assert phase_seconds(ramp(0.25), 6.0, 1.0) == pytest.approx(2.5)


def test_gripper_travel_does_not_slow_a_phase():
    poses = ramp(0.02)
    poses[:, 7] = np.linspace(0.0, 5.0, len(poses))
    assert phase_seconds(poses, 3.0, 1.5) == 1.5


def test_pose_at_interpolates_and_reports_the_passed_index():
    poses = ramp(1.0, count=5)
    pose, index = pose_at(poses, 0.375)
    assert index == 1
    assert pose[3] == pytest.approx(0.375)
    pose, index = pose_at(poses, 1.0)
    assert index == 4 and pose[3] == 1.0
    assert pose_at(poses, -1.0)[1] == 0


def test_replay_finds_the_can_and_box_on_a_tilted_table():
    from scripts.pick_replay import replay, synthetic_cloud

    frames = [synthetic_cloud(can=(0.40, 0.10), seed=seed, table_points=40000)
              for seed in range(3)]
    plane, item, box, _ = replay(frames, quiet=True)
    assert plane.tilt_degrees == pytest.approx(9.0, abs=0.5)
    assert item.center[:2] == pytest.approx((0.40, 0.10), abs=0.01)
    assert item.top == pytest.approx(0.12, abs=0.015)
    assert item.width == pytest.approx(0.066, abs=0.015)
    assert box is not None and box.center[:2] == pytest.approx((0.47, -0.22), abs=0.03)


def test_replay_follows_a_hint_to_the_far_can():
    from scripts.pick_replay import synthetic_cloud
    from scripts.pick_object import measure_frame

    cloud = synthetic_cloud(can=(0.55, 0.20), box=None, table_points=40000)
    _, item, box = measure_frame(cloud, near=(0.55, 0.20), report=lambda *_: None)
    assert item.center[:2] == pytest.approx((0.55, 0.20), abs=0.01)
    assert box is None


def test_last_accepted_option_is_tried_first():
    from scripts.pick_object import ordered_options

    options = ordered_options((25.0, 35.0), remembered=(35.0, 0.5))
    assert options[0] == (35.0, 0.5)
    assert sorted(options) == sorted(ordered_options((25.0, 35.0)))
    assert ordered_options((25.0,), remembered=(99.0, 1.0))[0] == (25.0, 1.0)


def test_lean_settles_only_after_the_base_has_moved_and_stopped():
    from scripts.pick_object import lean_settled

    assert not lean_settled([0.1] * 20, initial=0.0, degrees=4.0)       # never leaned
    assert not lean_settled(list(np.linspace(0, 4, 20)), 0.0, 4.0)      # still moving
    assert lean_settled(list(np.linspace(0, 4, 20)) + [4.0, 4.1] * 5, 0.0, 4.0)
    assert lean_settled([-3.9, -4.0] * 5, initial=0.0, degrees=4.0)     # either sign
    assert not lean_settled([4.0] * 3, 0.0, 4.0)                        # too few samples


def test_deferred_runs_in_the_background_and_reraises():
    from scripts.pick_object import Deferred

    assert Deferred(lambda: (["attempt"], None)).result() == (["attempt"], None)
    with pytest.raises(RuntimeError, match="boom"):
        Deferred(lambda: (_ for _ in ()).throw(RuntimeError("boom"))).result()


def test_warm_server_runs_jobs_survives_bad_ones_and_retires(tmp_path, monkeypatch):
    import scripts.pick_object as po

    ran = []

    def fake_execute(**kwargs):
        ran.append(kwargs)
        po.log("complete", f"place={kwargs['place']}")
        assert (tmp_path / "pick.pid").exists()      # 'stop' can find the job
        if len(ran) == 2:
            po.shutdown_event.set()

    monkeypatch.setattr(po, "execute", fake_execute)
    monkeypatch.setattr(po.tr, "_load_bbos", lambda: None)
    po.shutdown_event.clear()

    def submit():
        # Like pick_lab.sh: wait for the server, then drop jobs in atomically.
        while not (tmp_path / "server.pid").exists():
            time.sleep(0.01)
        for number, words in enumerate(
                ("--execute --adjust", "--no-such-flag", "--execute --place --near 0.4 0.1")):
            incoming = tmp_path / "jobs" / ".incoming"
            incoming.write_text(words + "\n")
            incoming.rename(tmp_path / "jobs" / f"{number}.job")

    client = threading.Thread(target=submit)
    client.start()
    try:
        assert po.serve(po.build_parser(), tmp_path) == 0
    finally:
        client.join()
        po.shutdown_event.clear()
        po.cancel_event.clear()
    assert [job["place"] for job in ran] == [False, True]
    assert ran[1]["adjust"] is False and po.scan.near == (0.4, 0.1)
    assert "place=True" in (tmp_path / "pick.log").read_text()
    assert not (tmp_path / "pick.pid").exists() and not (tmp_path / "server.pid").exists()


def test_a_can_sized_thing_in_front_of_the_table_is_never_the_target():
    from scripts.pick_object import measure_frame
    from scripts.pick_replay import synthetic_cloud

    cloud = synthetic_cloud(can=(0.50, 0.10), box=None, table_points=40000, near_edge=0.30)
    rng = np.random.default_rng(3)
    # A chair-back post 10 cm in front of the table edge, nearer than the can.
    post = np.column_stack((rng.normal(0.20, 0.012, 600), rng.normal(0.30, 0.012, 600),
                            rng.uniform(0.76, 0.84, 600)))
    chair = np.column_stack((-post[:, 1], post[:, 0], post[:, 2])).astype(np.float32)
    _, item, _ = measure_frame(np.vstack((cloud, chair)), report=lambda *_: None)
    assert item.center[:2] == pytest.approx((0.50, 0.10), abs=0.01)


def test_view_is_steady_only_when_recent_table_fits_agree():
    from types import SimpleNamespace as Plane
    from scripts.pick_object import view_steady

    moving = [Plane(tilt_degrees=t, near_edge=e, inliers=n) for t, e, n in
              ((8.5, 0.28, 577), (12.6, 0.22, 679), (1.6, 0.19, 668), (13.8, 0.16, 961))]
    still = [Plane(tilt_degrees=5.0 + 0.2 * i, near_edge=0.26, inliers=9000 + 50 * i)
             for i in range(4)]
    assert not view_steady(moving)
    assert not view_steady(still[:3])
    assert view_steady(moving + still)
    growing = [Plane(tilt_degrees=5.0, near_edge=0.26, inliers=n) for n in (3000, 5000, 7000, 9000)]
    assert not view_steady(growing)        # the table is still coming into view


class FakePoints:
    """A camera.points reader that serves a fixed list of clouds, one per read."""

    def __init__(self, clouds):
        self.clouds, self.index = clouds, -1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def ready(self):
        self.index = min(self.index + 1, len(self.clouds) - 1)
        return True

    @property
    def data(self):
        cloud = self.clouds[self.index]
        return {"timestamp": self.index, "num_points": len(cloud), "points": cloud}


def test_scan_drops_a_lock_made_while_the_robot_was_still_moving():
    import scripts.pick_object as po
    from scripts.pick_replay import synthetic_cloud

    # While settling the "can" appears to wander; once still it sits at (0.40, 0.10).
    wandering = [synthetic_cloud(can=(0.55 + 0.04 * i, -0.05 * i), box=None, seed=i,
                                 table_points=30000) for i in range(3)]
    still = [synthetic_cloud(can=(0.40, 0.10), box=None, seed=10 + i, table_points=30000)
             for i in range(8)]
    po.scan.near = po.scan.virtual = po.scan.record_dir = None
    po.cancel_event.clear()
    reader = FakePoints(wandering + still)
    _, item = po.scan(lambda *_, **__: reader)
    assert item.center[:2] == pytest.approx((0.40, 0.10), abs=0.01)


def test_lift_clears_the_box_rim_and_release_spots_stay_inside_the_walls():
    from scripts.pick_object import (LIFT_METRES, PLACE_INSIDE_WALL_METRES, lift_for_box,
                                     place_candidates)
    from scripts.tabletop_scene import TableObject

    box = TableObject((0.415, -0.125, 0.79), 0.084, 0.37, 0.35, 0.0, 5000)
    assert lift_for_box(None) == LIFT_METRES
    assert lift_for_box(box) >= box.top + 0.05 > LIFT_METRES
    lift_point = np.array([0.21, -0.33, 0.95])
    candidates = place_candidates(box, lift_point, 85.0, -0.8)
    assert candidates[0][0][:2] == pytest.approx(box.center[:2])    # centre first
    assert len(candidates) == 24                  # 3 spots x 4 pitches, level then sunk
    for position, quaternion, _ in candidates:
        assert lift_point[2] - 0.04 < position[2] <= lift_point[2]  # level, or sunk a little
        offset = np.abs(position[:2] - np.asarray(box.center[:2]))
        assert np.all(offset <= 0.35 / 2 - PLACE_INSIDE_WALL_METRES + 1e-9)
        assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    # later spots move toward the pick, shortening the reach
    nearer = [np.linalg.norm(c[0][:2] - lift_point[:2]) for c in candidates[:3]]
    assert nearer == sorted(nearer, reverse=True)


def test_the_nudge_that_held_last_time_is_tried_first():
    from scripts.pick_object import RETRY_NUDGES_CM, ordered_nudges

    assert ordered_nudges() == list(RETRY_NUDGES_CM)
    assert ordered_nudges((2.0, 0.0, 0.0))[0] == (2.0, 0.0, 0.0)
    assert sorted(ordered_nudges((2.0, 0.0, 0.0))) == sorted(RETRY_NUDGES_CM)
    assert ordered_nudges((9.0, 9.0, 9.0)) == list(RETRY_NUDGES_CM)


def test_a_rim_pinch_is_told_apart_from_a_full_hold():
    from scripts.pick_object import grip_is_pinched

    assert grip_is_pinched(0.189)                 # measured: slipped on the lift
    assert not grip_is_pinched(0.307)             # measured: carried to the box
    assert not grip_is_pinched(0.314)
    assert grip_is_pinched(0.30, expected=0.45)   # a wider object held last time


def test_cans_already_in_the_box_are_not_targets_and_drop_spots_rotate():
    from scripts.pick_object import inside_footprint, measure_frame, place_candidates
    from scripts.pick_replay import synthetic_cloud
    from scripts.tabletop_scene import TableObject

    box = TableObject((0.47, -0.22, 0.79), 0.10, 0.25, 0.25, 0.0, 5000)
    assert inside_footprint(box, (0.50, -0.20))
    assert not inside_footprint(box, (0.40, 0.10))
    turned = TableObject((0.47, -0.22, 0.79), 0.10, 0.40, 0.10, np.pi / 2, 5000)
    assert inside_footprint(turned, (0.47, -0.05)) and not inside_footprint(turned, (0.62, -0.22))

    cloud = synthetic_cloud(can=(0.40, 0.10), box=None, table_points=40000)
    everywhere = TableObject((0.40, 0.10, 0.79), 0.10, 0.30, 0.30, 0.0, 5000)
    assert measure_frame(cloud, report=lambda *_: None)[1] is not None
    assert measure_frame(cloud, report=lambda *_: None, filled_box=everywhere)[1] is None

    lift_point = np.array([0.21, -0.45, 0.95])
    first = place_candidates(box, lift_point, 85.0, 0.0, already_placed=0)[0][0]
    second = place_candidates(box, lift_point, 85.0, 0.0, already_placed=1)[0][0]
    assert not np.allclose(first[:2], second[:2])


def test_all_mode_rescans_after_a_drop_and_stops_when_the_table_is_clear(monkeypatch):
    import contextlib
    import scripts.pick_object as po
    from scripts.tabletop_scene import TableObject

    box = TableObject((0.47, -0.22, 0.79), 0.10, 0.25, 0.25, 0.0, 5000)
    scans, rounds = [], iter(["SLIPPED", "PLACED", "PLACED"])

    def fake_scan(_reader):
        scans.append(fake_scan.filled_box)
        if len(scans) > 3:
            raise RuntimeError("no steady graspable object within 12s")
        fake_scan.last_box = box
        return "plane", "item"

    fake_scan.last_box = fake_scan.filled_box = None
    fake_scan.others = []
    monkeypatch.setattr(po, "scan", fake_scan)
    def fake_pick_with(*args):
        result = next(rounds)
        fake_pick_with.placed += result == "PLACED"
        return result

    fake_pick_with.placed = 0
    monkeypatch.setattr(po, "pick_with", fake_pick_with)
    monkeypatch.setattr(po.tr, "_load_bbos",
                        lambda: (None, None, lambda *a, **k: contextlib.nullcontext(), None, None))
    monkeypatch.setattr(po.tr, "fresh", lambda _reader: {"rpy": [0.0, 0.0, 0.0]})
    po.cancel_event.clear()
    assert po.execute(plan_only=False, adjust=True, allow_lean=False, place=True,
                      everything=True) == "PLACED"
    # scan 1: can dropped -> scan 2 re-finds and delivers it -> scan 3 delivers the next
    # -> scan 4 sees nothing left. The box is remembered once something is inside it.
    assert scans == [None, None, box, box]
    assert po.pick_with.placed == 2
