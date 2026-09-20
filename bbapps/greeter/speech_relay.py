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
import math
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
# A fall alert outranks every conversation color, so it has its own file: a
# check-in posting "listening" must not erase the emergency. The short TTL
# means the flashing stops by itself if the posting app dies.
LED_EMERGENCY_FILE = "led_emergency"
LED_EMERGENCY_TTL_S = 3.0


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
    *,
    cancel=None,
    require_success: bool = False,
    probe: bool = False,
) -> bool:
    """Ask the speaker's owner to say ``text`` (or play ``wav``).

    A strict caller requires an explicit successful playback receipt. A probe
    verifies the owner's cancellation/receipt support without making a sound.
    Cancellation removes the request; an updated owner checks it between chunks.
    """

    if not text and wav is None and not probe:
        raise ValueError("speech relay needs text or a wav path")
    if cancel is not None and cancel.is_set():
        return False
    request_id = uuid.uuid4().hex
    payload: dict[str, Any] = {
        "id": request_id, "created": time.time(), "expires": time.time() + timeout,
        "cancellable": cancel is not None, "probe": probe,
    }
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
            if cancel is not None and cancel.is_set():
                return False
            if done.exists():
                receipt = done.read_text()
                if not receipt.strip():
                    return not require_success  # Legacy owners only touched .done.
                try:
                    return json.loads(receipt).get("ok") is True
                except (ValueError, AttributeError):
                    return False
            if cancel is None:
                time.sleep(poll_s)
            else:
                cancel.wait(poll_s)
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


def post_led_emergency(
    active: bool,
    ttl: float = LED_EMERGENCY_TTL_S,
    spool: Path = SPOOL_DIR,
) -> None:
    """Ask the LED owner to flash the emergency pattern; False releases it."""

    path = spool / LED_EMERGENCY_FILE
    try:
        if active:
            _write_atomic(path, {"expires": time.time() + ttl})
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass


def read_led_emergency(
    spool: Path = SPOOL_DIR, now: Callable[[], float] = time.time
) -> bool:
    """Return whether another app's emergency request is still fresh."""

    try:
        payload = json.loads((spool / LED_EMERGENCY_FILE).read_text())
        return now() < float(payload["expires"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


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
    speak_cancellable: Callable[[str, Callable[[], bool]], None] | None = None,
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
            created = float(payload.get("created") or 0.0)
            expires = float(payload.get("expires", created + REQUEST_TTL_S if created else now() + REQUEST_TTL_S))
            if not math.isfinite(created) or not math.isfinite(expires):
                raise ValueError("invalid request time")
        except (OSError, ValueError, TypeError, AttributeError):
            path.unlink(missing_ok=True)
            continue

        request_id = path.stem
        if now() >= expires or (created and now() - created > REQUEST_TTL_S):
            log(f"[speech-relay] dropped stale request {request_id}")
            path.unlink(missing_ok=True)
            continue

        def cancelled():
            return not path.exists() or now() >= expires

        ok, error = False, None
        try:
            if payload.get("cancellable") and speak_cancellable is None:
                raise RuntimeError("owner does not support cancellable playback")
            if payload.get("probe"):
                ok = True
            elif payload.get("text"):
                log(f"[speech-relay] speaking for another app: {payload['text']}")
                if payload.get("cancellable"):
                    speak_cancellable(str(payload["text"]), cancelled)
                else:
                    speak_text(str(payload["text"]))
                if cancelled():
                    raise RuntimeError("playback cancelled or expired")
                ok = True
                spoken += 1
            elif payload.get("wav") and play_wav is not None:
                if payload.get("cancellable"):
                    raise RuntimeError("cancellable wav playback is unavailable")
                log(f"[speech-relay] playing {payload['wav']}")
                play_wav(Path(str(payload["wav"])))
                ok = True
                spoken += 1
        except Exception as exc:  # noqa: BLE001 - never kill the owner's loop
            error = str(exc)
            log(f"[speech-relay] request {request_id} failed: {exc!r}")
        finally:
            try:
                # Do not leave a receipt after the requester cancelled/timed out.
                if path.exists():
                    _write_atomic(path.with_suffix(".done"), {"ok": ok, "error": error})
            except OSError:
                pass
            path.unlink(missing_ok=True)
    return spoken


__all__ = [
    "REQUEST_TTL_S",
    "SPOOL_DIR",
    "post_led_emergency",
    "post_led_status",
    "read_led_emergency",
    "read_led_status",
    "request",
    "serve_pending",
]
