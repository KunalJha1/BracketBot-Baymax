"""Hand one utterance to whichever process owns ``speaker.audio``.

``speaker.audio`` accepts a single writer process at a time, and the always-on
voice assistant holds that writer open for its entire lifetime. Any other app
that tries to speak while it runs dies on ``RuntimeError: Writer for
speaker.audio already exists`` -- the emotion greeter used to detect a
distressed face, compose its opening line, and then fall over silently.

So the assistant is the one speaker owner, and everybody else posts a request
here. A requester drops a JSON file into the spool directory and waits for the
matching ``.done`` marker; the assistant polls the spool from its main loop and
plays each request on the speaker it already holds. This is the same atomic
file-interlock shape the ground-safety alert already uses between the greeter
and the navigator, so it needs no new BBOS topic.

Requesters should try their own ``Writer`` first and only fall back to the
relay: when nothing else owns the speaker, writing directly is simpler and does
not depend on the assistant running at all.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import uuid
from typing import Any, Callable


SPOOL_DIR = Path("/tmp/bracketbot_speech")
# A request nobody serves is dropped rather than spoken minutes later, out of
# context. Longer than the assistant's own slowest reply, short enough that a
# stale file never surprises the person.
REQUEST_TTL_S = 45.0
# How long a requester waits for the owner to finish speaking before giving up.
DEFAULT_TIMEOUT_S = 30.0
# led.ctrl has the same single-writer rule as the speaker, so a relayed
# conversation also posts its state here for the owner's LEDs to mirror. The
# hint expires on its own in case the requester dies mid-conversation.
LED_STATUS_FILE = "led_status"
LED_STATUS_TTL_S = 30.0


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON so a reader never observes a half-written request."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def request(
    text: str | None = None,
    wav: str | Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    spool: Path = SPOOL_DIR,
    poll_s: float = 0.05,
) -> bool:
    """Ask the speaker's owner to say ``text`` (or play ``wav``).

    Returns True once the owner reports it finished, False on timeout -- which
    means nothing is serving the spool, so the caller should not assume the
    person heard anything.
    """

    if not text and wav is None:
        raise ValueError("speech relay needs text or a wav path")
    request_id = uuid.uuid4().hex
    payload: dict[str, Any] = {"id": request_id, "created": time.time()}
    if text:
        payload["text"] = text
    if wav is not None:
        payload["wav"] = str(wav)

    pending = spool / f"{request_id}.json"
    done = spool / f"{request_id}.done"
    _write_atomic(pending, payload)

    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if done.exists():
                return True
            time.sleep(poll_s)
        return False
    finally:
        pending.unlink(missing_ok=True)
        done.unlink(missing_ok=True)


def post_led_status(
    status: str | None,
    ttl: float = LED_STATUS_TTL_S,
    spool: Path = SPOOL_DIR,
) -> None:
    """Ask the LED owner to show ``status``; None or "idle" releases the LEDs."""

    path = spool / LED_STATUS_FILE
    try:
        if not status or status == "idle":
            path.unlink(missing_ok=True)
        else:
            _write_atomic(path, {"status": status, "expires": time.time() + ttl})
    except OSError:
        pass


def read_led_status(
    spool: Path = SPOOL_DIR, now: Callable[[], float] = time.time
) -> str | None:
    """Return the status another app asked the LEDs to show, if still fresh."""

    try:
        payload = json.loads((spool / LED_STATUS_FILE).read_text())
        if now() < float(payload["expires"]):
            return str(payload["status"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _pending_requests(spool: Path) -> list[Path]:
    if not spool.is_dir():
        return []
    return sorted(spool.glob("*.json"), key=lambda item: item.name)


def serve_pending(
    speak_text: Callable[[str], None],
    play_wav: Callable[[Path], None] | None = None,
    spool: Path = SPOOL_DIR,
    log: Callable[[str], None] = lambda _message: None,
    now: Callable[[], float] = time.time,
) -> int:
    """Play every queued request. Call this from the speaker owner's loop.

    Returns how many requests were spoken. Never raises: a malformed or
    unplayable request is dropped and logged, because the owner's own job
    (listening for the wake word) must not stop because another app asked for
    something odd.
    """

    spoken = 0
    for path in _pending_requests(spool):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            continue

        request_id = str(payload.get("id") or path.stem)
        created = float(payload.get("created") or 0.0)
        if created and now() - created > REQUEST_TTL_S:
            log(f"[speech-relay] dropped stale request {request_id}")
            path.unlink(missing_ok=True)
            continue

        try:
            if payload.get("text"):
                log(f"[speech-relay] speaking for another app: {payload['text']}")
                speak_text(str(payload["text"]))
                spoken += 1
            elif payload.get("wav") and play_wav is not None:
                log(f"[speech-relay] playing {payload['wav']}")
                play_wav(Path(str(payload["wav"])))
                spoken += 1
        except Exception as error:  # noqa: BLE001 - never kill the owner's loop
            log(f"[speech-relay] request {request_id} failed: {error!r}")
        finally:
            path.unlink(missing_ok=True)
            # The marker is what unblocks the requester, so write it even when
            # playback failed: it waited for an answer, not for success.
            try:
                (path.parent / f"{request_id}.done").touch()
            except OSError:
                pass
    return spoken


__all__ = [
    "REQUEST_TTL_S",
    "SPOOL_DIR",
    "post_led_status",
    "read_led_status",
    "request",
    "serve_pending",
]
