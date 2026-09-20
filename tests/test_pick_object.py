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
