"""Local, accessible web dashboard for BracketBot's recorded gestures.

The server binds to localhost by default. It discovers the first reachable SSH
alias (``botwifi`` then ``bot``), copies the safe gesture runner to the robot,
and executes only gestures from the fixed allowlist below.

    python3 scripts/robot_dashboard.py
    open http://127.0.0.1:8020
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "gesture_test.py"
REMOTE_RUNNER = "/tmp/gesture_test.py"
REMOTE_PID_FILE = "/tmp/bracketbot-gesture.pid"
SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=3",
    "-o", "ConnectionAttempts=1",
)

ACTIONS = {
    "wave": {
        "label": "Wave",
        "description": "Friendly left-arm wave",
        "file": "wave.json",
        "key": "1",
    },
    "handshake": {
        "label": "Handshake",
        "description": "Offer and shake the right hand",
        "file": "handshake.json",
        "key": "2",
    },
    "fist bump": {
        "label": "Fist bump",
        "description": "Offer a right-handed fist bump",
        "file": "fist bump.json",
        "key": "3",
    },
    "hug": {
        "label": "Hug",
        "description": "Open both arms for a hug",
        "file": "hug.json",
        "key": "4",
    },
}


class DashboardState:
    def __init__(self, ssh_hosts):
        self.ssh_hosts = tuple(ssh_hosts)
        self.lock = threading.Lock()
        self.host = None
        self.checking = False
        self.running = False
        self.action = None
        self.phase = "Starting connection check"
        self.error = None
        self.log = []
        self.cancel_requested = False
        self.process = None

    def snapshot(self):
        with self.lock:
            return {
                "host": self.host,
                "checking": self.checking,
                "connected": self.host is not None,
                "running": self.running,
                "action": self.action,
                "phase": self.phase,
                "error": self.error,
                "log": list(self.log[-24:]),
                "candidates": list(self.ssh_hosts),
                "actions": ACTIONS,
            }

    def add_log(self, line):
        line = line.strip()
        if not line:
            return
        with self.lock:
            self.log.append(line)
            del self.log[:-80]
            self.phase = line


class RobotController:
    def __init__(self, ssh_hosts):
        self.state = DashboardState(ssh_hosts)
        self._discover_lock = threading.Lock()
        self._stop_monitor = threading.Event()

    @staticmethod
    def _probe(host):
        command = [
            "ssh", *SSH_OPTIONS, host,
            'test -x "$HOME/.local/bin/uv" && '
            'test -d "$HOME/bbos"',
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def discover(self):
        if not self._discover_lock.acquire(blocking=False):
            return
        with self.state.lock:
            if self.state.running:
                self._discover_lock.release()
                return
            self.state.checking = True
            self.state.phase = "Checking robot connections…"
            self.state.error = None

        try:
            reachable = set()
            with ThreadPoolExecutor(max_workers=len(self.state.ssh_hosts)) as pool:
                futures = {
                    pool.submit(self._probe, host): host for host in self.state.ssh_hosts
                }
                for future in as_completed(futures):
                    if future.result():
                        reachable.add(futures[future])

            # Preserve the configured order when more than one interface works.
            selected = next(
                (host for host in self.state.ssh_hosts if host in reachable), None
            )
            with self.state.lock:
                self.state.host = selected
                if selected:
                    self.state.phase = f"Ready — connected through {selected}"
                else:
                    self.state.phase = "Robot not found"
                    self.state.error = (
                        "No configured SSH connection is reachable. Check Wi-Fi/USB, "
                        "then choose Reconnect."
                    )
        finally:
            with self.state.lock:
                self.state.checking = False
            self._discover_lock.release()

    def discover_async(self):
        threading.Thread(target=self.discover, name="robot-discovery", daemon=True).start()

    def start_monitor(self):
        self.discover_async()

        def monitor():
            while not self._stop_monitor.wait(15):
                with self.state.lock:
                    host = self.state.host
                    busy = self.state.running or self.state.checking
                if busy:
                    continue
                if host is None or not self._probe(host):
                    self.discover()

        threading.Thread(target=monitor, name="robot-monitor", daemon=True).start()

    def run_action(self, action):
        if action not in ACTIONS:
            return False, "Unknown command"

        with self.state.lock:
            if self.state.running:
                return False, f"{self.state.action} is already running"
            host = self.state.host
            if host is None:
                return False, "Robot is not connected; choose Reconnect"
            self.state.running = True
            self.state.action = ACTIONS[action]["label"]
            self.state.phase = f"Preparing {ACTIONS[action]['label']}…"
            self.state.error = None
            self.state.log = []
            self.state.cancel_requested = False

        threading.Thread(
            target=self._run_action,
            args=(host, action),
            name=f"gesture-{action}",
            daemon=True,
        ).start()
        return True, f"Started {ACTIONS[action]['label']}"

    def _run_action(self, host, action):
        info = ACTIONS[action]
        try:
            self.state.add_log(f"Connecting through {host}")
            movement_path = ROOT / "bbapps" / "greeter" / "movements" / info["file"]
            deploy = subprocess.run(
                [
                    "scp", "-q", *SSH_OPTIONS,
                    str(RUNNER), str(movement_path), f"{host}:/tmp/",
                ],
                capture_output=True,
                text=True,
                timeout=12,
            )
            if deploy.returncode != 0:
                detail = deploy.stderr.strip() or "copy failed"
                raise RuntimeError(f"Could not send the gesture runner: {detail}")

            with self.state.lock:
                if self.state.cancel_requested:
                    self.state.phase = "Cancelled before motion started"
                    return

            movement = info["file"].replace('"', '')
            label = info["label"].replace("'", "")
            remote_command = (
                'export PATH="$HOME/.local/bin:$PATH"; '
                f'exec "$HOME/.local/bin/uv" run --no-sync --project "$HOME/bbos" '
                f'python {REMOTE_RUNNER} '
                f'"/tmp/{movement}" '
                f"--name '{label}' --speed 0.6 --execute --pid-file {REMOTE_PID_FILE}"
            )
            process = subprocess.Popen(
                ["ssh", *SSH_OPTIONS, host, remote_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self.state.lock:
                self.state.process = process

            assert process.stdout is not None
            for line in process.stdout:
                self.state.add_log(line)
            return_code = process.wait()

            with self.state.lock:
                stopped = self.state.cancel_requested
            if return_code != 0:
                raise RuntimeError(f"Robot command exited with status {return_code}")
            with self.state.lock:
                self.state.phase = (
                    "Stopped safely — torque off" if stopped
                    else f"{info['label']} complete — torque off"
                )
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            with self.state.lock:
                self.state.error = str(exc)
                self.state.phase = "Command failed"
                self.state.host = None
        finally:
            with self.state.lock:
                self.state.running = False
                self.state.action = None
                self.state.process = None
                self.state.cancel_requested = False

    def stop(self):
        with self.state.lock:
            if not self.state.running:
                return False, "No gesture is running"
            host = self.state.host
            self.state.cancel_requested = True
            self.state.phase = "Stop requested — returning arms safely…"

        if host:
            def request_stop():
                subprocess.run(
                    [
                        "ssh", *SSH_OPTIONS, host,
                        "for attempt in 1 2 3 4 5 6 7 8 9 10; do "
                        f"if test -s {REMOTE_PID_FILE}; then "
                        f"xargs kill -INT < {REMOTE_PID_FILE}; exit 0; fi; "
                        "sleep 0.25; done; exit 1",
                    ],
                    capture_output=True,
                    timeout=6,
                )

            threading.Thread(target=request_stop, name="gesture-stop", daemon=True).start()
        return True, "Stop requested"


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>BracketBot controls</title>
<style>
:root { --bg:#101319; --card:#1a2029; --text:#f7f8fa; --muted:#bac4d1;
  --line:#3c4858; --accent:#63d8ff; --good:#72e6a1; --warn:#ffd166; --danger:#ff6577; }
* { box-sizing:border-box; }
body { margin:0; min-height:100vh; background:var(--bg); color:var(--text);
  font:18px/1.5 system-ui,-apple-system,sans-serif; }
main { width:min(760px,calc(100% - 28px)); margin:0 auto; padding:28px 0 48px; }
h1 { margin:0 0 4px; font-size:clamp(1.8rem,5vw,2.6rem); }
.intro { color:var(--muted); margin:0 0 22px; }
.status { border:2px solid var(--line); background:var(--card); border-radius:16px;
  padding:16px 18px; margin-bottom:20px; }
.status-line { display:flex; align-items:center; gap:12px; font-weight:750; }
.dot { width:14px; height:14px; flex:none; border-radius:50%; background:var(--warn); }
.dot.good { background:var(--good); } .dot.bad { background:var(--danger); }
#detail { color:var(--muted); margin:5px 0 0 26px; }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
button { min-height:94px; border:2px solid var(--line); border-radius:16px; padding:14px;
  text-align:left; background:var(--card); color:var(--text); font:inherit; cursor:pointer; }
button:hover:not(:disabled) { border-color:var(--accent); transform:translateY(-1px); }
button:focus-visible { outline:4px solid var(--accent); outline-offset:3px; }
button:disabled { opacity:.48; cursor:not-allowed; }
.label { display:block; font-size:1.15rem; font-weight:800; }
.desc { display:block; color:var(--muted); font-size:.9rem; margin-top:3px; }
.key { float:right; border:1px solid var(--line); border-radius:6px; padding:1px 7px;
  color:var(--muted); font-size:.8rem; }
.controls { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:14px; }
.secondary { min-height:58px; text-align:center; }
.stop { min-height:58px; text-align:center; background:#491923; border-color:var(--danger); font-weight:850; }
.log-wrap { margin-top:22px; }
.log-wrap summary { cursor:pointer; color:var(--muted); }
pre { white-space:pre-wrap; overflow-wrap:anywhere; max-height:220px; overflow:auto;
  background:#090b0f; padding:12px; border-radius:10px; color:#d7e0ea; font-size:.78rem; }
.error { color:#ff9ca8; font-weight:700; margin-top:10px; }
.sr-only { position:absolute; width:1px; height:1px; padding:0; margin:-1px;
  overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }
@media (max-width:560px) { .grid { grid-template-columns:1fr; } main { padding-top:18px; } }
@media (prefers-reduced-motion:reduce) { * { transition:none!important; scroll-behavior:auto!important; }
  button:hover:not(:disabled) { transform:none; } }
@media (prefers-contrast:more) { :root { --line:#eef3f8; --muted:#eef3f8; } }
</style>
</head>
<body>
<main>
  <h1>BracketBot controls</h1>
  <p class="intro">Choose one gesture. The dashboard finds the active robot connection automatically.</p>
  <section class="status" aria-labelledby="connection-title">
    <h2 id="connection-title" class="sr-only">Robot connection</h2>
    <div class="status-line"><span id="dot" class="dot" aria-hidden="true"></span><span id="status">Connecting…</span></div>
    <p id="detail" aria-live="polite">Checking botwifi and bot</p>
    <p id="error" class="error" role="alert" hidden></p>
  </section>
  <section class="grid" aria-label="Robot gestures">
    <button data-action="wave"><span class="key">1</span><span class="label">Wave</span><span class="desc">Friendly left-arm wave</span></button>
    <button data-action="handshake"><span class="key">2</span><span class="label">Handshake</span><span class="desc">Offer and shake the right hand</span></button>
    <button data-action="fist bump"><span class="key">3</span><span class="label">Fist bump</span><span class="desc">Offer a right-handed fist bump</span></button>
    <button data-action="hug"><span class="key">4</span><span class="label">Hug</span><span class="desc">Open both arms for a hug</span></button>
  </section>
  <div class="controls">
    <button id="reconnect" class="secondary">Reconnect</button>
    <button id="stop" class="stop" disabled><span class="key">Esc</span>Stop motion</button>
  </div>
  <details class="log-wrap"><summary>Technical details</summary><pre id="log">No activity yet.</pre></details>
</main>
<script>
const buttons=[...document.querySelectorAll('[data-action]')];
const statusEl=document.getElementById('status'), detail=document.getElementById('detail');
const dot=document.getElementById('dot'), error=document.getElementById('error');
const stop=document.getElementById('stop'), reconnect=document.getElementById('reconnect');
const log=document.getElementById('log'); let current={};
async function post(path, body={}) {
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  return response.json();
}
async function refresh() {
  try {
    current=await (await fetch('/api/status',{cache:'no-store'})).json();
    const ready=current.connected&&!current.running&&!current.checking;
    buttons.forEach(b=>b.disabled=!ready); stop.disabled=!current.running;
    reconnect.disabled=current.running||current.checking;
    dot.className='dot '+(current.connected?'good':current.checking?'':'bad');
    statusEl.textContent=current.running?`${current.action} in progress`:current.connected?`Connected: ${current.host}`:current.checking?'Connecting…':'Robot offline';
    detail.textContent=current.phase;
    error.hidden=!current.error; error.textContent=current.error||'';
    log.textContent=current.log.length?current.log.join('\n'):'No activity yet.';
  } catch (_) {
    dot.className='dot bad'; statusEl.textContent='Dashboard connection lost';
    detail.textContent='Reload this page to reconnect.';
  }
}
buttons.forEach(button=>button.addEventListener('click',()=>post('/api/run',{action:button.dataset.action}).then(refresh)));
reconnect.addEventListener('click',()=>post('/api/discover').then(refresh));
stop.addEventListener('click',()=>post('/api/stop').then(refresh));
document.addEventListener('keydown',event=>{
  if(event.repeat||event.target.matches('input,textarea,select')) return;
  if(event.key==='Escape'&&current.running){ event.preventDefault(); stop.click(); return; }
  const map={'1':'wave','2':'handshake','3':'fist bump','4':'hug'};
  if(map[event.key]) { const button=document.querySelector(`[data-action="${map[event.key]}"]`); if(!button.disabled) button.click(); }
});
setInterval(refresh,500); refresh();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    controller = None

    def log_message(self, format, *args):
        return

    def _send(self, body, status=HTTPStatus.OK, content_type="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self):
        length = min(int(self.headers.get("Content-Length", "0")), 4096)
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        if self.path == "/":
            self._send(PAGE, content_type="text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._send(self.controller.state.snapshot())
        else:
            self._send({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path == "/api/discover":
            self.controller.discover_async()
            self._send({"ok": True, "message": "Connection check started"})
        elif self.path == "/api/run":
            ok, message = self.controller.run_action(self._json_body().get("action"))
            self._send({"ok": ok, "message": message}, HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT)
        elif self.path == "/api/stop":
            ok, message = self.controller.stop()
            self._send({"ok": ok, "message": message}, HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT)
        else:
            self._send({"error": "Not found"}, HTTPStatus.NOT_FOUND)


def parse_hosts(value):
    hosts = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not hosts:
        raise argparse.ArgumentTypeError("provide at least one SSH host")
    return hosts


def main():
    parser = argparse.ArgumentParser(description="Accessible local BracketBot command dashboard")
    parser.add_argument("--bind", default="127.0.0.1", help="listen address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--ssh-hosts", type=parse_hosts, default=("botwifi", "bot"),
                        help="comma-separated SSH aliases in priority order")
    args = parser.parse_args()

    controller = RobotController(args.ssh_hosts)
    Handler.controller = controller
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    controller.start_monitor()
    print(f"BracketBot dashboard: http://{args.bind}:{args.port}", flush=True)
    print(f"SSH candidates: {', '.join(args.ssh_hosts)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        controller._stop_monitor.set()
        controller.stop()
        deadline = time.monotonic() + 8.0
        while controller.state.snapshot()["running"] and time.monotonic() < deadline:
            time.sleep(0.1)
        server.server_close()


if __name__ == "__main__":
    main()
