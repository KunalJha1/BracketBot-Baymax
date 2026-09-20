import io
import json
from types import SimpleNamespace
import wave
from contextlib import contextmanager

import numpy as np

from bbapps.greeter.local_voice import (
    EspeakSynthesizer,
    FallbackSynthesizer,
    FallbackTranscriber,
    HttpTtsSynthesizer,
    clarify_speech,
    LocalVoiceError,
    SpeechSegmenter,
    WhisperCppTranscriber,
    WhisperServerTranscriber,
    clean_transcript,
    condition_utterance,
    play_dance_music,
    speaker_chunks,
    split_sentences,
    trim_silence,
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
        wake_tail_s=0.0,
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


def test_segmenter_does_not_close_during_post_wake_grace():
    segmenter = SpeechSegmenter(
        100,
        threshold_db=-30,
        pre_roll_s=0.2,
        min_utterance_s=0.1,
        trailing_silence_s=0.2,
        start_grace_s=0.6,
        max_utterance_s=2.0,
        wake_tail_s=0.0,
    )
    speech = np.full(10, 10000, dtype=np.int16)
    quiet = np.zeros(10, dtype=np.int16)

    assert segmenter.push(speech, triggered=True) is None
    # The wake phrase in pre-roll counts as speech, but 0.2 seconds of silence
    # must not immediately close the query during the 0.6-second grace period.
    assert segmenter.push(quiet) is None
    assert segmenter.push(quiet) is None
    assert segmenter.recording is True
    assert segmenter.push(speech) is None
    assert segmenter.push(quiet) is None
    assert segmenter.push(quiet) is None
    assert segmenter.push(quiet) is not None


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
    assert json.loads(requests[0][0].data) == {
        "text": "Natural voice.",
        "sample_rate": 16000,
    }


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


def test_dance_music_streams_through_existing_speaker_writer(tmp_path):
    path = tmp_path / "dance.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.array([1000, -1000, 500, -500], dtype=np.int16).tobytes())

    class FakeWriter:
        def __init__(self):
            self.frames = []

        @contextmanager
        def buf(self):
            frame = {}
            yield frame
            self.frames.append(np.asarray(frame["audio"]).copy())

    writer = FakeWriter()
    calls = 0

    def is_dancing():
        nonlocal calls
        calls += 1
        return calls == 1

    play_dance_music(
        writer,
        path,
        SimpleNamespace(sample_rate=16000, chunk_size=4, channels=1),
        is_dancing,
        volume=0.5,
    )

    assert len(writer.frames) == 1
    assert writer.frames[0].reshape(-1).tolist() == [500, -500, 250, -250]


def test_non_speech_annotations_never_reach_the_router():
    assert clean_transcript(" [BLANK_AUDIO] ") == ""
    assert clean_transcript("(wind blowing)") == ""
    assert clean_transcript("*laughs*") == ""
    assert clean_transcript("[MUSIC] What is a robot? (coughs)") == "What is a robot?"
    assert clean_transcript("  Hey BracketBot,   wave  ") == "Hey BracketBot, wave"


def test_silent_turns_transcribe_to_nothing():
    def runner(command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="[BLANK_AUDIO]\n", stderr="")

    transcriber = WhisperCppTranscriber(
        "/bin/sh", __file__, runner=runner
    )
    assert transcriber.transcribe(np.zeros(16, dtype=np.int16), 16000) == ""


def test_conditioning_normalizes_quiet_speech_without_clipping_loud_speech():
    quiet = (np.sin(np.linspace(0, 40, 4000)) * 3000).astype(np.int16)
    loud = (np.sin(np.linspace(0, 40, 4000)) * 20000).astype(np.int16)

    lifted = condition_utterance(quiet)
    assert np.max(np.abs(lifted)) > 15000

    kept = condition_utterance(loud)
    peak = int(np.max(np.abs(kept)))
    # Room is left below full scale, and nothing is squared off at the rails.
    assert peak <= 32000
    assert np.count_nonzero(np.abs(kept) >= 32767) == 0


def test_conditioning_will_not_amplify_a_whisper_into_room_noise():
    # A very quiet turn is lifted by the allowed gain and no further, so a
    # near-silent room is never inflated to look like speech.
    barely_there = (np.sin(np.linspace(0, 40, 4000)) * 200).astype(np.int16)
    lifted = condition_utterance(barely_there, max_gain=8.0)
    assert 1400 <= int(np.max(np.abs(lifted))) <= 1700


def test_conditioning_removes_dc_offset_and_survives_silence():
    offset = np.full(2000, 5000, dtype=np.int16)
    assert np.max(np.abs(condition_utterance(offset))) == 0
    assert condition_utterance(np.zeros(0, dtype=np.int16)).size == 0


def test_trimming_drops_room_silence_around_the_words():
    sample_rate = 16000
    silence = np.zeros(sample_rate, dtype=np.int16)
    speech = (np.sin(np.linspace(0, 400, sample_rate // 2)) * 8000).astype(np.int16)
    trimmed = trim_silence(np.concatenate([silence, speech, silence]), sample_rate)

    assert trimmed.size < sample_rate * 2
    assert trimmed.size >= speech.size
    # An entirely silent turn is returned untouched for the caller to reject.
    assert trim_silence(silence, sample_rate).size == silence.size


def test_detection_gain_never_clips_the_stored_utterance():
    loud = np.full(400, 12000, dtype=np.int16)
    segmenter = SpeechSegmenter(
        100,
        threshold_db=-30,
        pre_roll_s=0.0,
        min_utterance_s=0.5,
        trailing_silence_s=0.5,
        level_gain=8.0,
    )
    segmenter.push(loud, triggered=True)
    segmenter.push(loud)
    utterance = segmenter.push(np.zeros(200, dtype=np.int16))

    assert utterance is not None
    # An 8x detection gain would have driven 12000 into the rails.
    assert int(np.max(np.abs(utterance))) == 12000


def test_sentences_are_split_for_incremental_speech():
    assert split_sentences(
        "Rinse the cut under cool water. Then cover it with a bandage."
    ) == ["Rinse the cut under cool water.", "Then cover it with a bandage."]
    # A short fragment is merged forward instead of being spoken alone.
    assert split_sentences("Okay. I will wave now, and then go limp.") == [
        "Okay. I will wave now, and then go limp."
    ]
    assert split_sentences("   ") == []


def test_whisper_server_is_used_when_available_and_falls_back_when_not():
    payload = json.dumps({"text": " Hello there. "}).encode("utf-8")
    seen = {}

    def opener(api_request, timeout):
        seen["url"] = api_request.full_url
        seen["body"] = api_request.data
        return FakeHttpResponse(payload)

    server = WhisperServerTranscriber("http://127.0.0.1:8910/", opener=opener)
    assert server.transcribe(np.zeros(320, dtype=np.int16), 16000) == "Hello there."
    assert seen["url"] == "http://127.0.0.1:8910/inference"
    assert b'name="file"; filename="utterance.wav"' in seen["body"]
    assert b"RIFF" in seen["body"]

    def failing_opener(api_request, timeout):
        raise OSError("connection refused")

    def runner(command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="From the CLI.\n", stderr="")

    fallback = FallbackTranscriber(
        WhisperServerTranscriber("http://127.0.0.1:8910", opener=failing_opener),
        WhisperCppTranscriber("/bin/sh", __file__, runner=runner),
    )
    assert fallback.transcribe(np.zeros(320, dtype=np.int16), 16000) == "From the CLI."


def test_clarify_speech_lifts_consonant_band_and_controls_peak():
    rate = 16000
    t = np.arange(rate) / rate
    body = 8000 * np.sin(2 * np.pi * 250 * t)
    consonants = 800 * np.sin(2 * np.pi * 3200 * t)
    rumble = 4000 * np.sin(2 * np.pi * 50 * t)
    pcm = (body + consonants + rumble).astype(np.int16)

    out = clarify_speech(pcm, rate)

    def band(signal, freq):
        return np.abs(np.fft.rfft(signal.astype(np.float64)))[freq]

    assert out.dtype == np.int16
    assert len(out) == len(pcm)
    assert np.max(np.abs(out)) <= int(0.9 * 32768)
    before = band(pcm, 3200) / band(pcm, 250)
    after = band(out, 3200) / band(out, 250)
    assert after > 2.0 * before
    assert band(out, 50) / band(out, 250) < 0.2 * band(pcm, 50) / band(pcm, 250)


def test_clarify_speech_passes_silence_through():
    silence = np.zeros(1600, dtype=np.int16)
    assert np.array_equal(clarify_speech(silence, 16000), silence)


def test_whisper_audio_context_tracks_utterance_length():
    from bbapps.greeter.local_voice import whisper_audio_context

    # The floor is 768: large-v3-turbo hallucinates on a tighter window.
    assert whisper_audio_context(16_000 * 2, 16_000) == 768
    assert whisper_audio_context(16_000 * 8, 16_000) == 768
    assert whisper_audio_context(16_000 * 20, 16_000) == 1152
    assert whisper_audio_context(16_000 * 40, 16_000) == 1500


def test_segmenter_offers_speculation_mid_pause_and_tracks_resumed_speech():
    rate = 16000
    segmenter = SpeechSegmenter(
        rate, threshold_db=-38.0, pre_roll_s=0.0, min_utterance_s=0.2,
        trailing_silence_s=0.6, wake_tail_s=0.0,
    )
    loud = (np.ones(1600) * 8000).astype(np.int16)
    quiet = np.zeros(1600, dtype=np.int16)
    segmenter.push(loud, triggered=True)
    for _ in range(3):
        assert segmenter.push(loud) is None
    assert not segmenter.speculation_ready(0.25)
    for _ in range(3):
        assert segmenter.push(quiet) is None
    assert segmenter.speculation_ready(0.25)
    heard = segmenter.loud_chunks
    assert len(segmenter.snapshot()) == 1600 * 6

    # Talking again must invalidate a guess made during the pause.
    segmenter.push(loud)
    assert segmenter.loud_chunks == heard + 1
    assert not segmenter.speculation_ready(0.25)


def test_segmenter_ends_turn_in_a_room_louder_than_the_fixed_threshold():
    rate = 16000
    rng = np.random.default_rng(0)
    segmenter = SpeechSegmenter(
        rate, threshold_db=-38.0, pre_roll_s=0.5, min_utterance_s=0.2,
        trailing_silence_s=0.6, max_utterance_s=8.0,
    )
    noise = lambda: (rng.standard_normal(1600) * 1000).astype(np.int16)  # ~-30 dB
    speech = lambda: (rng.standard_normal(1600) * 9000).astype(np.int16)  # ~-11 dB
    for _ in range(30):
        assert segmenter.push(noise()) is None
    segmenter.push(speech(), triggered=True)
    for _ in range(10):
        assert segmenter.push(speech()) is None
    result = None
    pushed = 0
    while result is None and pushed < 60:
        result = segmenter.push(noise())
        pushed += 1
    # Ends on the 0.6 s pause, not at the 8 s cap.
    assert result is not None and pushed <= 7


def test_trimming_follows_a_noisy_room_floor():
    rng = np.random.default_rng(0)
    rate = 16_000
    # Room noise at about -39 dBFS, above the fixed -45 dB threshold.
    audio = rng.normal(0, 370, rate * 4)
    audio[rate : rate * 2] += np.sin(np.arange(rate) * 0.3) * 8000
    trimmed = trim_silence(audio.astype(np.int16), rate)
    assert rate <= len(trimmed) <= rate * 1.6


def test_repeated_sentences_collapse_to_one():
    assert clean_transcript("Follow me. Follow me. Follow me.") == "Follow me."
    assert clean_transcript("Stop. Follow me.") == "Stop. Follow me."


def test_fallback_voice_is_held_instead_of_flipping_mid_reply():
    now = [0.0]
    calls = []

    class Broken:
        def synthesize(self, text, rate):
            calls.append(text)
            raise LocalVoiceError("down")

    class Working:
        def synthesize(self, text, rate):
            return np.zeros(4, dtype=np.int16)

    voice = FallbackSynthesizer(Broken(), Working(), retry_after_s=60, clock=lambda: now[0])
    voice.synthesize("one", 16_000)
    voice.synthesize("two", 16_000)
    assert calls == ["one"]
    now[0] = 61.0
    voice.synthesize("three", 16_000)
    assert calls == ["one", "three"]


def _wake_segmenter():
    return SpeechSegmenter(
        16_000, pre_roll_s=1.0, trailing_silence_s=0.6, start_grace_s=0.9
    )


def test_wake_phrase_tail_and_a_pause_do_not_end_the_turn_before_the_question():
    segmenter = _wake_segmenter()
    loud = (np.ones(1600) * 8000).astype(np.int16)
    quiet = np.zeros(1600, dtype=np.int16)
    segmenter.push(loud, triggered=True)
    segmenter.push(loud)  # the end of "Hey BracketBot"
    for _ in range(14):  # 1.4 s waiting for the listening light
        assert segmenter.push(quiet) is None
    for _ in range(10):
        assert segmenter.push(loud) is None
    result = None
    for _ in range(6):
        result = segmenter.push(quiet)
    assert result is not None
    assert segmenter.last_had_speech


def test_a_wake_with_nothing_after_it_gives_up_without_speech():
    segmenter = _wake_segmenter()
    quiet = np.zeros(1600, dtype=np.int16)
    segmenter.push(quiet, triggered=True)
    results = [segmenter.push(quiet) for _ in range(50)]
    assert results[-1] is not None and all(r is None for r in results[:-1])
    assert not segmenter.last_had_speech
