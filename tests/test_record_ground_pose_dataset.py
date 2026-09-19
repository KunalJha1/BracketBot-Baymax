from scripts.record_ground_pose_dataset import make_record


def test_record_keeps_ground_evidence_without_expression_or_images():
    status = {
        "frame": 42,
        "people": 1,
        "track_ids": [7],
        "ground_status": "clear",
        "depth_aligned": True,
        "camera_age_ms": 120,
        "scan_fps": 4.0,
        "expression": "neutral",
        "map": {
            "map_epoch": 5,
            "robot_position": [1.0, 2.0],
            "robot_heading": 0.3,
        },
        "ground_observations": [
            {"track_id": 7, "torso_height_m": 1.2, "state": "clear"}
        ],
    }

    record = make_record(status, "standing")

    assert record["label"] == "standing"
    assert record["map_epoch"] == 5
    assert record["ground_observations"][0]["torso_height_m"] == 1.2
    assert "expression" not in record
    assert "jpeg" not in record
