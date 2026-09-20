"""Prepare speech before driving, then play once while the control loop holds zero."""

from contextlib import ExitStack
import threading
import time

from ground_approach import GROUND_LINE


class GroundSpeech:
    def __init__(self, config, open_writer, chunks, *, relay_request=None):
        self.config, self.open_writer, self.chunks = config, open_writer, chunks
        self.relay_request = relay_request
        self.resources = ExitStack()
        self.speaker = None
        self.relay = False
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.thread = None
        self.error = None

    def reserve(self):
        """Verify speech before motion; retain an available writer until cleanup."""
        if self.speaker is not None or self.relay:
            return
        try:
            self.speaker = self.resources.enter_context(self.open_writer())
        except RuntimeError as exc:
            if "Writer for speaker.audio already exists" not in str(exc) or self.relay_request is None:
                raise
            if not self.relay_request(probe=True, cancel=self.cancel, require_success=True, timeout=3.0):
                raise RuntimeError(
                    "speaker is owned, but a cancellable speech relay is unavailable; "
                    "deploy/restart the updated voice assistant while idle, or stop it before check-in"
                ) from exc
            self.relay = True

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._play, name="ground-check-in", daemon=True)
            self.thread.start()

    def _play(self):
        try:
            if self.cancel.is_set():
                return
            self.reserve()
            if self.relay:
                confirmed = self.relay_request(text=GROUND_LINE, cancel=self.cancel, require_success=True)
                if not confirmed and not self.cancel.is_set():
                    raise RuntimeError("speech playback was not confirmed by the voice assistant")
                return
            due = time.monotonic()
            for chunk in self.chunks:
                if self.cancel.is_set():
                    return
                with self.speaker.buf() as frame:
                    frame["audio"] = chunk.reshape(-1, self.config.channels)
                due += self.config.chunk_size / self.config.sample_rate
                if self.cancel.wait(max(0, due - time.monotonic())):
                    return
            self.cancel.wait(0.4)
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.done.set()

    def close(self):
        self.cancel.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.resources.close()


def prepare_speech(Config, Writer, Type):
    from local_voice import EspeakSynthesizer, speaker_chunks
    from speech_relay import request

    cfg = Config("speaker")
    audio = EspeakSynthesizer().synthesize(GROUND_LINE, cfg.sample_rate)
    speech = GroundSpeech(
        cfg, lambda: Writer("speaker.audio", Type("speaker_audio"), keeptime=False),
        speaker_chunks(audio, cfg.chunk_size, cfg.channels),
        relay_request=request,
    )
    try:
        speech.reserve()
    except Exception:
        speech.close()
        raise
    return speech
