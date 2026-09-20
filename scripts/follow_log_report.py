"""Summarise a robot_follow CSV log into the numbers the robot gates check.

    scp 'bot:/tmp/baymax_follow_*.csv' artifacts/follow/
    python scripts/follow_log_report.py artifacts/follow/baymax_follow_20260920_101500.csv

Standard library only. Fractions are over FOLLOWING samples; sign agreement
compares what was sent with wheel feedback while the command was clearly nonzero.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
import sys


def load(path):
    with open(path, newline="") as source:
        return list(csv.DictReader(source))


def num(row, key):
    value = row.get(key, "")
    return None if value in ("", "None") else float(value)


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q / 100 * len(ordered)) - 1)]


def sign_agreement(rows, sent, measured, threshold):
    pairs = [(num(r, sent), num(r, measured)) for r in rows if abs(num(r, sent) or 0.0) >= threshold]
    if not pairs:
        return None
    return sum(a * b > 0 for a, b in pairs) / len(pairs)


def blocked_to_stop(rows):
    """Seconds from the first BLOCKED sample until the sent speed reached zero."""
    for i, row in enumerate(rows):
        if row["state"] == "BLOCKED":
            start = num(row, "t")
            for later in rows[i:]:
                if (num(later, "v") or 0.0) == 0.0:
                    return round(num(later, "t") - start, 3)
            return None
    return None


def summarise(rows, band=0.20):
    following = [r for r in rows if r["state"] == "FOLLOWING" and num(r, "error") is not None]
    errors = [num(r, "error") for r in following]
    bearings = [abs(math.degrees(num(r, "bearing"))) for r in following if num(r, "bearing") is not None]
    return {
        "samples": len(rows),
        "following_samples": len(following),
        "in_band_fraction": None if not errors else round(sum(abs(e) <= band for e in errors) / len(errors), 3),
        "abs_error_p95_m": None if not errors else round(percentile([abs(e) for e in errors], 95), 3),
        "bearing_within_10deg_fraction": None if not bearings else round(sum(b <= 10 for b in bearings) / len(bearings), 3),
        "v_sign_agreement": sign_agreement(rows, "v", "measured_v", 0.05),
        "omega_sign_agreement": sign_agreement(rows, "omega", "measured_omega", 0.2),
        "blocked_to_stop_s": blocked_to_stop(rows),
        "rules": dict(Counter(r["rule"] for r in rows)),
        "associations": dict(Counter(r["association"] for r in rows if r.get("association"))),
        "last_rule": rows[-1]["rule"] if rows else None,
    }


def main(argv=None):
    paths = (argv if argv is not None else sys.argv[1:])
    if not paths:
        raise SystemExit("usage: follow_log_report.py LOG.csv [LOG.csv ...]")
    for path in paths:
        print(path)
        print(json.dumps(summarise(load(path)), indent=2))


if __name__ == "__main__":
    main()
