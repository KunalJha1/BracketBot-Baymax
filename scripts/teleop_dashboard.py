#!/usr/bin/env python3
"""One-page WASD drive test at http://127.0.0.1:8021/ — can the base move at all?

Pushes ``robot_teleop.py`` to the robot and pipes key state to it over the
shared ``scripts/bot`` ssh connection. Hold a key to move; releasing it, closing
the tab, or killing this script stops the base (the robot side has a 0.4 s
deadman of its own).

    python3 scripts/teleop_dashboard.py
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading

ROOT = Path(__file__).resolve().parent
BOT = str(ROOT / "bot")
REMOTE = "/tmp/robot_teleop.py"

PAGE = """<!doctype html><meta charset=utf-8><title>WASD drive test</title>
<style>
body{font:16px system-ui;background:#111;color:#eee;display:grid;place-items:center;min-height:100vh;margin:0}
main{text-align:center;max-width:460px;padding:16px}
.keys{display:grid;grid-template-columns:repeat(3,72px);gap:8px;justify-content:center;margin:20px 0}
.k{height:72px;border-radius:12px;background:#222;border:2px solid #444;display:grid;place-items:center;
   font-size:24px;font-weight:700;user-select:none;touch-action:none}
.k.on{background:#2a7;border-color:#5fb}
#stop{grid-column:1/4;height:52px;background:#a22;border-color:#f66}
pre{background:#1a1a1a;padding:12px;border-radius:8px;text-align:left;min-height:88px;white-space:pre-wrap}
label{display:block;margin:10px 0}
</style>
<main>
<h2>WASD drive test</h2>
<div>Hold W/S to drive, A/D to turn. Space = stop. Click this page first so it has keyboard focus.</div>
<label>speed <input id=speed type=range min=0.05 max=0.30 step=0.05 value=0.15> <b id=speedv>0.15</b> m/s</label>
<label>turn <input id=turn type=range min=0.2 max=1.0 step=0.1 value=0.6> <b id=turnv>0.6</b> rad/s</label>
<div class=keys>
<span></span><div class=k data-k=w>W</div><span></span>
<div class=k data-k=a>A</div><div class=k data-k=s>S</div><div class=k data-k=d>D</div>
<div class=k id=stop>STOP (space)</div>
</div>
<pre id=out>connecting...</pre>
</main>
<script>
const held=new Set(), $=id=>document.getElementById(id);
const paint=()=>document.querySelectorAll('.k[data-k]').forEach(e=>e.classList.toggle('on',held.has(e.dataset.k)));
function twist(){const s=+$('speed').value,t=+$('turn').value;
  return {v:(held.has('w')?s:0)-(held.has('s')?s:0), w:(held.has('a')?t:0)-(held.has('d')?t:0)};}
const send=()=>fetch('/cmd',{method:'POST',body:JSON.stringify(twist())}).catch(()=>{});
function release(){held.clear();paint();send();}
addEventListener('keydown',e=>{const k=e.key.toLowerCase();
  if(k===' '){e.preventDefault();release();return;}
  if('wasd'.includes(k)&&k.length===1){e.preventDefault();held.add(k);paint();send();}});
addEventListener('keyup',e=>{held.delete(e.key.toLowerCase());paint();send();});
addEventListener('blur',release);
document.addEventListener('visibilitychange',()=>{if(document.hidden)release();});
document.querySelectorAll('.k[data-k]').forEach(e=>{
  e.addEventListener('pointerdown',ev=>{ev.preventDefault();held.add(e.dataset.k);paint();send();});
  for(const n of ['pointerup','pointerleave','pointercancel'])e.addEventListener(n,()=>{held.delete(e.dataset.k);paint();send();});});
$('stop').addEventListener('pointerdown',release);
for(const id of ['speed','turn'])$(id).addEventListener('input',()=>$(id+'v').textContent=$(id).value);
setInterval(()=>{if(held.size)send();},100);   // keeps the robot-side deadman fed while a key is held
setInterval(async()=>{try{const s=await(await fetch('/status')).json();
  const t=s.telemetry||{};
  $('out').textContent=`link: ${s.link}\\ncommand: v=${t.v??'-'} m/s  w=${t.w??'-'} rad/s\\nwheels:  ${t.wheel_v??'-'} m/s   pitch: ${t.pitch??'-'} deg\\n${s.log.join('\\n')}`;
}catch(e){$('out').textContent='dashboard script not reachable';}},250);
</script>"""


class Link:
    """The ssh pipe to robot_teleop.py."""

    def __init__(self, v_max: float):
        self.lock = threading.Lock()
        self.telemetry: dict = {}
        self.log: list[str] = []
        self.state = "starting"
        subprocess.run([BOT, "push", str(ROOT / "robot_teleop.py"), "--to", "/tmp"], check=True)
        self.process = subprocess.Popen(
            [BOT, "py", REMOTE, "--v-max", str(v_max)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        threading.Thread(target=self._pump, name="teleop-stdout", daemon=True).start()

    def _pump(self):
        for line in self.process.stdout:
            line = line.strip()
            if line.startswith("TELEOP "):
                try:
                    self.telemetry = json.loads(line[7:])
                    self.state = "driving" if self.telemetry.get("live") else "connected (idle)"
                except ValueError:
                    pass
            elif line:
                print(line, flush=True)
                self.log = (self.log + [line])[-4:]
        self.state = "robot script exited - restart this dashboard"

    def send(self, v: float, w: float) -> None:
        with self.lock:
            try:
                self.process.stdin.write(json.dumps({"v": v, "w": w}) + "\n")
                self.process.stdin.flush()
            except (OSError, ValueError):
                self.state = "robot link lost - restart this dashboard"

    def close(self) -> None:
        self.send(0.0, 0.0)
        try:
            self.process.stdin.close()          # EOF: the robot script zeroes the twist and exits
            self.process.wait(timeout=3.0)
        except (OSError, subprocess.TimeoutExpired):
            self.process.kill()


def handler(link: Link):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _reply(self, body: bytes, kind: str):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/status":
                body = json.dumps({"link": link.state, "telemetry": link.telemetry, "log": link.log})
                self._reply(body.encode(), "application/json")
            else:
                self._reply(PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            try:
                message = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                link.send(float(message["v"]), float(message["w"]))
            except (ValueError, KeyError, TypeError):
                pass
            self._reply(b"{}", "application/json")

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--v-max", type=float, default=0.30)
    args = parser.parse_args()
    link = Link(args.v_max)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(link))
    print(f"WASD drive test: http://127.0.0.1:{args.port}/  (Ctrl-C stops the robot and exits)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        link.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
