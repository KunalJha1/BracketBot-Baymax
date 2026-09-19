# BracketBot Baymax project instructions

## Dashboard development

- Treat `http://127.0.0.1:8020/` as the canonical local robot dashboard.
- Keep hot reload enabled by default for `scripts/robot_dashboard.py`: saved dashboard source changes should restart the server once the robot is idle, and open dashboard pages should then reload automatically.
- Never interrupt an active robot action or lean mode merely to apply a development reload; wait for a safe idle/balance state.
- After dashboard changes, run `python3 -m py_compile scripts/robot_dashboard.py` and `python3 -m pytest -q tests/test_robot_dashboard.py`.
