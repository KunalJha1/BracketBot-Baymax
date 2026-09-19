import io
import json
from types import SimpleNamespace
import wave

import numpy as np

from bbapps.greeter.local_voice import (
    EspeakSynthesizer,
    FallbackSynthesizer,
    HttpTtsSynthesizer,
    LocalVoiceError,
    SpeechSegmenter,
    WhisperCppTranscriber,
    speaker_chunks,
)


class FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.payload


def test_segmenter_keeps_preroll_and_stops_after_trailing_silence():
    segmenter = SpeechSegmenter(
        100,
        threshold_db=-30,
        pre_roll_s=0.2,
        min_utterance_s=0.2,
        trailing_silence_s=0.2,
        max_utterance_s=2.0,
    )
    quiet = np.zeros(10, dtype=np.int16)
    speech = np.full(10, 10000, dtype=np.int16)

    assert segmenter.push(quiet) is None
    assert segmenter.push(speech, triggered=True) is None
    assert segmenter.recording is True
    assert segmenter.push(speech) is None
    assert segmenter.push(quiet) is None
    utterance = segmenter.push(quiet)

    assert utterance is not None
    assert len(utterance) == 50
    assert np.max(utterance) == 10000
    assert segmenter.recording is False


def test_whisper_cpp_writes_wav_and_returns_clean_transcript(tmp_path):
    binary = tmp_path / "whisper-cli"
    model = tmp_path / "ggml-base.en.bin"
    binary.write_text("")
    model.write_text("")
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        wav_path = command[command.index("--file") + 1]
        with wave.open(wav_path, "rb") as wav:
            assert wav.getframerate() == 16000
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
        return SimpleNamespace(returncode=0, stdout="  Hey BracketBot, weather please.\n", stderr="")

    transcriber = WhisperCppTranscriber(
        str(binary),
        str(model),
        threads=3,
        runner=runner,
    )
    transcript = transcriber.transcribe(np.ones(320, dtype=np.int16), 16000)

    assert transcript == "Hey BracketBot, weather please."
    assert calls[0][0][calls[0][0].index("--threads") + 1] == "3"
    assert "--no-timestamps" in calls[0][0]
    assert calls[0][0][calls[0][0].index("--best-of") + 1] == "1"
    assert calls[0][0][calls[0][0].index("--beam-size") + 1] == "1"
    assert "--no-fallback" in calls[0][0]


def test_espeak_wav_is_resampled_to_robot_rate(tmp_path):
    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(np.full(2205, 500, dtype=np.int16).tobytes())

    def runner(command, **kwargs):
        assert kwargs["input"] == b"Hello from BracketBot."
        return SimpleNamespace(returncode=0, stdout=wav_bytes.getvalue(), stderr=b"")

    binary = tmp_path / "espeak-ng"
    binary.write_text("")
    synth = EspeakSynthesizer(str(binary), runner=runner)

    pcm = synth.synthesize("Hello from BracketBot.", 16000)

    assert pcm.dtype == np.int16
    assert len(pcm) == 1600


def test_http_tts_posts_text_and_decodes_wav():
    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.full(1600, 700, dtype=np.int16).tobytes())
    requests = []

    def opener(http_request, timeout):
        requests.append((http_request, timeout))
        return FakeHttpResponse(wav_bytes.getvalue())

    synth = HttpTtsSynthesizer(
        "http://192.168.55.100:8900/tts", timeout=3.0, opener=opener
    )
    pcm = synth.synthesize("Natural voice.", 16000)

    assert len(pcm) == 1600
    assert requests[0][1] == 3.0
    assert json.loads(requests[0][0].data) == {"text": "Natural voice."}


def test_natural_voice_falls_back_when_service_is_unavailable():
    class BrokenSynth:
        def synthesize(self, text, target_rate):
            raise LocalVoiceError("offline")

    class WorkingSynth:
        def synthesize(self, text, target_rate):
            return np.array([1, 2, 3], dtype=np.int16)

    pcm = FallbackSynthesizer(BrokenSynth(), WorkingSynth()).synthesize(
        "Hello", 16000
    )

    assert pcm.tolist() == [1, 2, 3]


def test_speaker_chunks_add_lead_in_and_pad():
    chunks = speaker_chunks(
        np.array([1, 2, 3], dtype=np.int16),
        chunk_size=4,
        channels=1,
        lead_chunks=1,
    )

    assert [chunk.tolist() for chunk in chunks] == [
        [0, 0, 0, 0],
        [1, 2, 3, 0],
    ]
