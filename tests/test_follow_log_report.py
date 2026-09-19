import csv

import pytest

from follow_log_report import load, summarise
from robot_follow import CSV_FIELDS


def write_log(path, rows):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def row(t, state="FOLLOWING", error=0.0, bearing=0.0, v=0.1, measured_v=0.1, omega=0.0,
        measured_omega=0.0, rule="ok"):
    return {"t": t, "state": state, "rule": rule, "gap": 1.0,
            "range": None if error is None else 1.0 + error, "error": error, "bearing": bearing,
            "v": v, "measured_v": measured_v, "omega": omega, "measured_omega": measured_omega}


def test_report_on_a_synthetic_log(tmp_path):
    rows = [row(0.00, state="SEARCHING", error=None, bearing=None, v=0.0, measured_v=0.0, rule="no-track")]
    rows += [row(0.05 * i, error=0.05, bearing=0.02) for i in range(1, 19)]
    rows += [row(0.95, error=0.30, bearing=0.5)]  # one out-of-band, off-axis sample
    rows += [row(1.00, state="BLOCKED", v=0.06, rule="blocked"), row(1.05, state="BLOCKED", v=0.0, rule="blocked")]
    rows += [row(1.10, v=0.2, measured_v=-0.1, omega=0.5, measured_omega=0.4, rule="heartbeat")]
    path = tmp_path / "log.csv"
    write_log(path, rows)

    report = summarise(load(path))

    assert report["samples"] == len(rows)
    assert report["following_samples"] == 20
    assert report["in_band_fraction"] == pytest.approx(19 / 20)
    assert report["abs_error_p95_m"] == pytest.approx(0.05)
    assert report["bearing_within_10deg_fraction"] == pytest.approx(19 / 20)
    assert report["v_sign_agreement"] == pytest.approx(20 / 21)
    assert report["omega_sign_agreement"] == 1.0
    assert report["blocked_to_stop_s"] == pytest.approx(0.05)
    assert report["rules"]["blocked"] == 2
    assert report["last_rule"] == "heartbeat"


def test_report_handles_a_log_with_no_following(tmp_path):
    path = tmp_path / "log.csv"
    write_log(path, [row(0.0, state="SEARCHING", error=None, bearing=None, v=0.0, rule="no-track")])
    report = summarise(load(path))
    assert report["in_band_fraction"] is None
    assert report["v_sign_agreement"] is None
    assert report["blocked_to_stop_s"] is None
