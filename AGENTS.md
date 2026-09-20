# BracketBot Baymax project instructions

## Dashboard development

- Treat `http://127.0.0.1:8020/` as the canonical local robot dashboard.
- Keep hot reload enabled by default for `scripts/robot_dashboard.py`: saved dashboard source changes should restart the server once the robot is idle, and open dashboard pages should then reload automatically.
- Never interrupt an active robot action or lean mode merely to apply a development reload; wait for a safe idle/balance state.
- After dashboard changes, run `python3 -m py_compile scripts/robot_dashboard.py` and `python3 -m pytest -q tests/test_robot_dashboard.py`.

## Talking to the robot

- Use `scripts/bot` for every robot command instead of hand-written `ssh`/`scp` lines. It finds the live route (USB `bot`, mDNS, `botwifi`), caches it, and reuses one kept-alive multiplexed connection shared by all terminals and agent sessions.
- `scripts/bot <command>` runs a shell command, `scripts/bot py <args>` runs the BBOS venv python, `scripts/bot push <files> [--to DIR]` and `scripts/bot pull <remote> [local]` copy files, `scripts/bot status` / `close` manage the connection. Set `BOT_HOST` to skip discovery.
- If it prints "no route to the robot", the robot is offline: say so rather than retrying other ssh variants.
