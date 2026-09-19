# /// script
# dependencies = [
#   "bbos",
#   "fastapi",
#   "uvicorn",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
import asyncio
import contextlib
import threading
import time
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn
from bbos import Reader
import socket

CAMS = ["head", "left", "right"]
WAIT_POLL_S = 0.005     # How often a client checks for a newer frame.
FRAME_WAIT_S = 1.0      # Longest /frame waits for a newer frame.

# Newest (sequence, jpeg) per camera, replaced whole so readers never see a
# torn pair. Every client keeps its own last sequence: a shared queue would
# let two viewers steal frames from each other and freeze one of them.
latest = {cam: (0, None) for cam in CAMS}


# One loop for every Reader: Loop's global pacing state is not thread-safe.
def camera_reader():
    with contextlib.ExitStack() as stack:
        readers = {c: stack.enter_context(Reader(f"camera.{c}.jpeg")) for c in CAMS}
        while True:
            for cam, r in readers.items():
                if not r.ready():
                    continue
                jpeg_bytes = bytes(r.data['jpeg'][:r.data['jpeg_len']])
                latest[cam] = (latest[cam][0] + 1, jpeg_bytes)


async def next_frame(cam, after, timeout):
    """Newest (seq, jpeg) newer than `after`, or (after, None) on timeout."""
    deadline = time.monotonic() + timeout
    while True:
        seq, frame = latest[cam]
        if seq != after and frame is not None:
            return seq, frame
        if time.monotonic() >= deadline:
            return after, None
        await asyncio.sleep(WAIT_POLL_S)


app = FastAPI()


def _make_routes(cam):
    @app.get(f"/{cam}/stream")
    async def stream():
        async def generate():
            seq = 0
            while True:
                seq, frame = await next_frame(cam, seq, FRAME_WAIT_S)
                if frame is None:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n'
                       + frame + b'\r\n')

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get(f"/{cam}/frame")
    async def frame(after: int = -1):
        seq, f = await next_frame(cam, after, FRAME_WAIT_S)
        if f is None:
            return Response(status_code=503, headers={"Cache-Control": "no-store"})
        return Response(content=f, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store", "X-Frame-Seq": str(seq)})


for _cam in CAMS:
    _make_routes(_cam)


# The page pulls one frame at a time per camera instead of holding an endless
# MJPEG <img> open. Browsers allow ~6 connections per host, so a couple of tabs
# of three endless streams used to leave new streams hanging on a frozen image,
# and an <img> stream never reconnects after the server restarts.
INDEX = '''
<html>
<head>
    <title>Camera Streams</title>
    <style>
        body { margin: 0; padding: 20px; background: #000; color: #fff; font-family: sans-serif; }
        div { margin-bottom: 20px; }
        h2 { margin: 5px 0; }
        h2 span { font-size: 14px; font-weight: normal; color: #8f8; }
        h2 span.stale { color: #f66; }
        img { max-width: 100%; height: auto; display: block; }
    </style>
</head>
<body>__IMGS__
<script>
async function run(cam) {
    const img = document.getElementById(cam), info = document.getElementById(cam + '-info');
    let seq = -1, frames = 0, lastFrame = performance.now(), shown = null;
    setInterval(() => {
        const age = (performance.now() - lastFrame) / 1000;
        info.className = age > 1 ? 'stale' : '';
        info.textContent = age > 1 ? `no new frame for ${age.toFixed(0)} s, retrying` : `${frames} fps`;
        frames = 0;
    }, 1000);
    while (true) {
        try {
            const r = await fetch(`/${cam}/frame?after=${seq}`, {cache: 'no-store'});
            if (r.ok) {
                seq = Number(r.headers.get('X-Frame-Seq'));
                const url = URL.createObjectURL(await r.blob());
                await new Promise(done => { img.onload = img.onerror = done; img.src = url; });
                if (shown) URL.revokeObjectURL(shown);
                shown = url; frames++; lastFrame = performance.now();
            }
        } catch (_) {
            await new Promise(done => setTimeout(done, 500));  // Server restarting.
        }
    }
}
__CAMS__.forEach(run);
</script>
</body>
</html>
'''


@app.get("/")
async def index():
    imgs = "\n".join(
        f'<div><h2>{cam} <span id="{cam}-info"></span></h2><img id="{cam}" /></div>'
        for cam in CAMS
    )
    cams = "[" + ", ".join(f'"{cam}"' for cam in CAMS) + "]"
    html = INDEX.replace("__IMGS__", imgs).replace("__CAMS__", cams)
    return Response(content=html, media_type="text/html",
                    headers={"Cache-Control": "no-store"})


def main():
    threading.Thread(target=camera_reader, daemon=True).start()

    host = socket.gethostname()
    print(f"[+] Streaming all cameras on http://{host}.local:8004/", flush=True)

    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="error",
                access_log=False, timeout_graceful_shutdown=1)

if __name__ == "__main__":
    main()
