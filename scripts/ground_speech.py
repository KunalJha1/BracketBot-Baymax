"""Prepare speech before driving, then play once while the control loop holds zero."""

import threading
import time

from ground_approach import GROUND_LINE


class GroundSpeech:
    def __init__(self, config, open_writer, chunks):
        self.config, self.open_writer, self.chunks = config, open_writer, chunks
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.thread = None
        self.error = None

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._play, name="ground-check-in", daemon=True)
            self.thread.start()

    def _play(self):
        try:
            with self.open_writer() as speaker:
                due = time.monotonic()
                for chunk in self.chunks:
                    if self.cancel.is_set():
                        return
                    with speaker.buf() as frame:
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


def prepare_speech(Config, Writer, Type):
    from local_voice import EspeakSynthesizer, speaker_chunks

    cfg = Config("speaker")
    audio = EspeakSynthesizer().synthesize(GROUND_LINE, cfg.sample_rate)
    return GroundSpeech(
        cfg, lambda: Writer("speaker.audio", Type("speaker_audio"), keeptime=False),
        speaker_chunks(audio, cfg.chunk_size, cfg.channels),
    )
