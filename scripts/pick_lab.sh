#!/usr/bin/env bash
# One command for the whole pick loop: sync, look, plan, pick, recover.
#
#   scripts/pick_lab.sh status            # what the robot sees now (no motion)
#   scripts/pick_lab.sh preview [--watch] # camera image with the can + box drawn on it
#   scripts/pick_lab.sh plan  [args]      # full validated plan, no motion
#   scripts/pick_lab.sh pick  [args]      # real pick with auto-retry, put back
#   scripts/pick_lab.sh place [args]      # pick and drop into the box beside it
#   scripts/pick_lab.sh clear [args]      # every can beside the box goes into it, one by one
#   scripts/pick_lab.sh warm | cold       # start / retire the warm server (auto-started)
#   scripts/pick_lab.sh hover [args]      # go to the hover pose and come back
#   scripts/pick_lab.sh rest              # lower both arms to their rest pose
#   scripts/pick_lab.sh lean on|off       # plug-and-play lean mode (default 4 deg)
#   scripts/pick_lab.sh space             # back up until the target is graspable
#   scripts/pick_lab.sh fix-gripper SIDE  # drive a gripper back into range
#   scripts/pick_lab.sh stop              # safe-stop a running attempt
#   scripts/pick_lab.sh log               # full log of the last attempt
#   scripts/pick_lab.sh pull              # fetch the last run's depth frames
#   scripts/pick_lab.sh replay [args]     # NO ROBOT: rerun perception locally on them
#
# Extra args pass straight to pick_object.py, e.g.
#   scripts/pick_lab.sh pick --near 0.40 0.15 --grip-torque 0.9
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE=/tmp/pick
PY='~/bbos/.venv/bin/python'
# One multiplexed SSH connection, reused by every call here and kept alive
# between runs: connection setup was costing more than the robot work.
SSH_OPTS=(-o ControlMaster=auto -o ControlPath=/tmp/pick-lab-%C -o ControlPersist=600
          -o ConnectTimeout=5 -o BatchMode=yes
          # A robot that reboots or drops off wifi leaves a master that still
          # says "running" and hangs every command; let it notice and die.
          -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
HOST_CACHE=/tmp/pick-lab-host
SSH() { ssh "${SSH_OPTS[@]}" "$@"; }
FILTER='timing\]|complete\]|median|lean\]|reach\]|state\] side|startup|accepted pitch|retry\]|place\]|space\]|grip\]|evidence|torque\]|cleanup|fatal|staged|rest\]|gripper\]|motion\] stage|complete\]'

host() {
  if [ -s "$HOST_CACHE" ]; then
    local cached; cached=$(cat "$HOST_CACHE")
    # A live control socket answers without a network round trip.
    if ssh "${SSH_OPTS[@]}" -O check "$cached" 2>/dev/null || SSH "$cached" true 2>/dev/null; then echo "$cached"; return 0; fi
  fi
  for h in bot bracketbot@bracketbot-184.local botwifi; do
    if SSH "$h" true 2>/dev/null; then echo "$h" | tee "$HOST_CACHE"; return 0; fi
  done
  echo "no route to the robot (tried bot, mDNS, botwifi)" >&2; return 1
}

# Copy only what actually changed: one checksum round trip beats four scps.
sync_files() {
  local files=(pick_object.py tabletop_scene.py table_rest.py camera_geometry.py)
  local stamp=/tmp/pick-lab-synced newest
  # Nothing edited since the last sync over this still-open connection: skip
  # the round trip. (A robot reboot wipes $REMOTE but also drops the socket.)
  newest=$(cd "$ROOT/scripts" && ls -t "${files[@]}" | head -1)
  if ssh "${SSH_OPTS[@]}" -O check "$1" 2>/dev/null && [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$1" ] && [ ! "$ROOT/scripts/$newest" -nt "$stamp" ]; then
    return 0
  fi
  local local_sums remote_sums changed=()
  local_sums=$(cd "$ROOT/scripts" && md5 -q "${files[@]}" 2>/dev/null || md5sum "${files[@]}" | cut -d" " -f1)
  remote_sums=$(SSH "$1" "mkdir -p $REMOTE; cd $REMOTE && md5sum ${files[*]} 2>/dev/null | cut -d' ' -f1")
  local i=1
  for f in "${files[@]}"; do
    local l r
    l=$(echo "$local_sums" | sed -n "${i}p"); r=$(echo "$remote_sums" | sed -n "${i}p")
    [ "$l" = "$r" ] || changed+=("$ROOT/scripts/$f")
    i=$((i + 1))
  done
  if [ ${#changed[@]} -gt 0 ]; then
    scp -q "${SSH_OPTS[@]}" "${changed[@]}" "$1:$REMOTE/" || return 1
    # New code: retire the warm server (it leaves once idle; never mid-job).
    SSH "$1" "cd $REMOTE; [ -f server.pid ] || exit 0; \
      if [ -f pick.pid ]; then echo 'a job is running: it keeps the OLD code' >&2; exit 0; fi; \
      kill -TERM \$(cat server.pid) 2>/dev/null; \
      for _ in \$(seq 1 50); do [ -f server.pid ] || break; sleep 0.1; done; rm -f server.pid"
  fi
  echo "$1" > "$stamp"
}

# Remote snippet: make sure the warm server (pick_object.py --serve) is up.
# It keeps bbos imported and the IK initialised, so a job starts in well under
# a second instead of paying Python + bbos start-up on every run.
ENSURE_SERVER="cd $REMOTE; \
  if ! { [ -f server.pid ] && kill -0 \$(cat server.pid) 2>/dev/null; }; then \
    rm -f server.pid; \
    setsid nohup $PY -u pick_object.py --serve $REMOTE > server.log 2>&1 < /dev/null & \
    for _ in \$(seq 1 300); do [ -f server.pid ] && break; sleep 0.1; done; \
    [ -f server.pid ] || { echo 'warm server failed to start:'; tail -5 server.log; exit 1; }; \
  fi"

# Submit one job to the warm server and stream its log until it finishes, all
# in a single SSH call. The server is detached, so an SSH drop cannot kill a
# motion; 'stop' still works because pick.pid exists while the job runs.
run_detached() {
  local h="$1"; shift
  SSH "$h" "$ENSURE_SERVER; rm -f pick.log; \
      printf '%s\n' '$*' > jobs/.incoming && mv jobs/.incoming jobs/\$(date +%s%N)-\$\$.job; \
      for _ in \$(seq 1 100); do [ -f pick.log ] && break; sleep 0.05; done; \
      tail -n +1 -f pick.log & TAIL=\$!; \
      for _ in \$(seq 1 3000); do sleep 0.1; \
        [ -f pick.pid ] || ls jobs/*.job >/dev/null 2>&1 || break; done; \
      sleep 0.15; kill \$TAIL 2>/dev/null" | grep --line-buffered -E "$FILTER" | cut -c1-170
}

CMD=${1:-status}; shift || true
FRAMES="$ROOT/artifacts/pick/latest"
if [ "$CMD" = replay ]; then
  # No robot needed: rerun perception on the last pulled frames (or a
  # synthetic table when there are none) in a couple of seconds.
  if [ -d "$FRAMES" ] && [ "${1:-}" != "--synthetic" ]; then
    exec python3 "$ROOT/scripts/pick_replay.py" "$FRAMES" "$@"
  fi
  exec python3 "$ROOT/scripts/pick_replay.py" "$@"
fi
H=$(host) || exit 1
sync_files "$H" || exit 1

case "$CMD" in
  status)
    SSH "$H" "cd $REMOTE && $PY -c \"
import sys, time, numpy as np; sys.path.insert(0, '$REMOTE')
import table_rest as tr
from tabletop_scene import fit_table_plane, find_objects, points_to_arm
_, Config, Reader, _, _ = tr._load_bbos()
with Reader('imu.orientation', keeptime=False) as i:
    while not i.ready(): time.sleep(0.02)
    print('lean/pitch %.1f deg' % float(np.asarray(i.data['rpy'])[1]))
for side in ('left', 'right'):
    cfg = Config('arm_' + side)
    with Reader('arm_%s.state' % side, keeptime=False) as r:
        t0 = time.time()
        while not r.ready() and time.time() - t0 < 3: time.sleep(0.02)
        pos = np.asarray(r.data['pos'], float)
    rad = float(np.asarray(cfg.q2urdf(pos.copy()), float)[7])
    print('%-5s gripper %.2f rad %s' % (side, rad, 'OK' if -0.3 <= rad <= 1.2 else 'OUT OF RANGE'))
with Reader('camera.points', keeptime=False) as r:
    while not r.ready(): time.sleep(0.02)
    n = int(r.data['num_points'])
    P = points_to_arm(np.asarray(r.data['points'])[:n].astype(float))
try:
    pl = fit_table_plane(P)
except RuntimeError as exc:
    raise SystemExit('no table in view: %s' % exc)
print('table %.3f m, tilt %.1f deg, near edge %.3f m  %s' % (pl.height_at(0.4, 0.0), pl.tilt_degrees, pl.near_edge, '' if pl.near_edge >= 0.13 else '(close to the table: raise climbs behind the edge)'))
print('%-22s %6s %6s %6s  %s' % ('object (fwd, left)', 'top', 'width', 'reach', 'verdict'))
for o in find_objects(P, pl)[:12]:
    reach = min(np.hypot(o.center[0], o.center[1] - 0.0975), np.hypot(o.center[0], o.center[1] + 0.0975))
    if o.width > 0.12 or not 0.04 < o.top < 0.32: continue
    verdict = 'GRASPABLE' if reach <= 0.48 else ('needs lean' if reach <= 0.68 else 'too far')
    print('%-22s %6.3f %6.3f %6.3f  %s' % ('%.3f, %.3f' % (o.center[0], o.center[1]), o.top, o.width, reach, verdict))
from tabletop_scene import find_box, select_graspable
objs = find_objects(P, pl)
target = select_graspable(objs, max_reach=0.68)
box = find_box(P, pl, objs, exclude=target)
print('target:', 'none' if target is None else '%.3f, %.3f' % target.center[:2])
print('box   :', 'none' if box is None else '%.3f, %.3f  rim %.3f  %.2fx%.2f m' % (box.center[0], box.center[1], box.top, box.length, box.width))
\"" ;;
  watch)
    # Poll the scene until a graspable target (and box, with --box) is in reach.
    want_box=0; [ "${1:-}" = "--box" ] && want_box=1
    for _ in $(seq 1 60); do
      out=$("$0" status 2>/dev/null | tail -3)
      echo "$out" | tr '\n' ' '; echo
      tgt=$(echo "$out" | grep '^target:' | grep -v none)
      box=$(echo "$out" | grep '^box   :' | grep -v none)
      if [ -n "$tgt" ] && { [ "$want_box" = 0 ] || [ -n "$box" ]; }; then
        echo "READY"; exit 0
      fi
      sleep 5
    done
    echo "still not ready"; exit 1 ;;
  plan)  SSH "$H" "rm -rf $REMOTE/frames"
         run_detached "$H" --record $REMOTE/frames "$@" ;;
  place) SSH "$H" "rm -rf $REMOTE/frames"
         run_detached "$H" --execute --adjust --auto-space --place --record $REMOTE/frames "$@" ;;
  clear) SSH "$H" "rm -rf $REMOTE/frames"
         run_detached "$H" --execute --adjust --auto-space --place --all --record $REMOTE/frames "$@" ;;
  warm)  SSH "$H" "$ENSURE_SERVER; echo \"warm server pid \$(cat server.pid)\"; tail -1 server.log" ;;
  cold)  SSH "$H" "cd $REMOTE; [ -f server.pid ] && kill -TERM \$(cat server.pid) && echo 'server retiring' || echo 'no server'" ;;
  pick)  SSH "$H" "rm -rf $REMOTE/frames"
         run_detached "$H" --execute --adjust --auto-space --record $REMOTE/frames "$@" ;;
  preview)
    # What the pick sees, drawn on the head camera: can, box, table edge.
    # Read-only on the robot. 'preview --watch' keeps refreshing.
    mkdir -p "$ROOT/artifacts/pick"
    watch=0; [ "${1:-}" = "--watch" ] && { watch=1; shift; }
    scp -q "${SSH_OPTS[@]}" "$ROOT/scripts/pick_preview.py" "$H:$REMOTE/" || exit 1
    while :; do
      SSH "$H" "cd $REMOTE && $PY pick_preview.py --capture $REMOTE/preview.npz" | grep -v '^\[preview\] saved' 
      scp -q "${SSH_OPTS[@]}" "$H:$REMOTE/preview.npz" "$ROOT/artifacts/pick/preview.npz" || exit 1
      python3 "$ROOT/scripts/pick_preview.py" "$ROOT/artifacts/pick/preview.npz" \
        --out "$ROOT/artifacts/pick/preview.jpg" "$@" || exit 1
      [ "${opened:-0}" = 1 ] || { open "$ROOT/artifacts/pick/preview.jpg" 2>/dev/null; opened=1; }
      [ "$watch" = 1 ] || break
    done ;;
  pull)
    # Bring the last run's depth frames home for 'replay'.
    rm -rf "$FRAMES"; mkdir -p "$FRAMES"
    SSH "$H" "cd $REMOTE/frames 2>/dev/null && tar cf - ." | tar xf - -C "$FRAMES" \
      && echo "$(ls "$FRAMES" | wc -l | tr -d ' ') frames in $FRAMES" ;;
  space) run_detached "$H" --space "$@" ;;
  hover) run_detached "$H" --execute --stop-at pregrasp "$@" ;;
  rest)  SSH "$H" "cd $REMOTE && $PY -u pick_object.py --rest 2>&1 | grep -E '$FILTER'" ;;
  fix-gripper) SSH "$H" "cd $REMOTE && $PY -u pick_object.py --fix-gripper $* 2>&1 | grep -E '$FILTER'" ;;
  lean)
    # Plug-and-play lean: "lean on [deg]" holds it in the background, "lean off" restores balance.
    case "${1:-status}" in
      on)  scp -q "${SSH_OPTS[@]}" "$ROOT/scripts/robot_base_mode.py" "$H:$REMOTE/" && \
           SSH "$H" "cd $REMOTE; [ -f lean.pid ] && kill -INT \$(cat lean.pid) 2>/dev/null; sleep 0.3; \
             setsid nohup $PY robot_base_mode.py --angle ${2:-4} --pid-file $REMOTE/lean.pid > lean.log 2>&1 < /dev/null & sleep 1.2; tail -2 $REMOTE/lean.log" ;;
      off) SSH "$H" "cd $REMOTE; if [ -f lean.pid ]; then kill -INT \$(cat lean.pid) && sleep 0.8 && tail -2 lean.log; else echo 'lean is not held by pick_lab'; fi" ;;
      *)   SSH "$H" "cd $REMOTE; if [ -f lean.pid ] && kill -0 \$(cat lean.pid) 2>/dev/null; then echo 'lean ON'; tail -1 lean.log; else echo 'lean OFF'; fi" ;;
    esac ;;
  stop)  SSH "$H" "if [ -f $REMOTE/pick.pid ]; then kill -INT \$(cat $REMOTE/pick.pid) && echo 'safe stop requested'; else echo 'nothing running'; fi" ;;
  log)   SSH "$H" "cat $REMOTE/pick.log" ;;
  *) echo "unknown command: $CMD"; sed -n '2,20p' "$0"; exit 2 ;;
esac
