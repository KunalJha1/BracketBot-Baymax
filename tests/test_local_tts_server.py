"""Tests for the TTS bridge's spoken-audio cache.

Nothing here invokes the macOS speech engine; synthesis is always replaced so
the tests measure caching behaviour alone.
"""

import pytest

from local_tts_server import SpeechCache, TtsServer, prewarm


WAV = b"RIFF....WAVEfake"


@pytest.fixture
def server(tmp_path):
    instance = TtsServer(
        ("127.0.0.1", 0), "Test Voice", 178, 4, SpeechCache(tmp_path / "tts")
    )
    try:
        yield instance
    finally:
        instance.server_close()


def fake_synthesis(server, audio=WAV):
    calls = []

    def synthesize(text):
        calls.append(text)
        return audio

    server.synthesize = synthesize
    return calls


def test_repeated_line_is_synthesized_once(server):
    calls = fake_synthesis(server)

    assert server.speak("Hello there.") == WAV
    assert server.speak("Hello there.") == WAV
    assert server.speak("Something else.") == WAV

    assert calls == ["Hello there.", "Something else."]


def test_cache_survives_a_new_server_over_the_same_directory(server, tmp_path):
    fake_synthesis(server)
    server.speak("Hello there.")

    replacement = TtsServer(
        ("127.0.0.1", 0), "Test Voice", 178, 4, SpeechCache(tmp_path / "tts")
    )
    try:
        calls = fake_synthesis(replacement)
        assert replacement.speak("Hello there.") == WAV
        assert calls == []
    finally:
        replacement.server_close()


def test_voice_settings_are_part_of_the_cache_key(tmp_path):
    cache = SpeechCache(tmp_path / "tts")
    cache.put("Hello there.", "Test Voice", 178, 4, WAV)

    assert cache.get("Hello there.", "Test Voice", 178, 4) == WAV
    assert cache.get("Hello there.", "Other Voice", 178, 4) is None
    assert cache.get("Hello there.", "Test Voice", 150, 4) is None
    assert cache.get("Hello there.", "Test Voice", 178, 0) is None
    assert cache.get("Hello again.", "Test Voice", 178, 4) is None


def test_disabled_cache_always_synthesizes(tmp_path):
    instance = TtsServer(("127.0.0.1", 0), "Test Voice", 178, 4, SpeechCache(None))
    try:
        calls = fake_synthesis(instance)
        instance.speak("Hello there.")
        instance.speak("Hello there.")
        assert calls == ["Hello there.", "Hello there."]
        assert instance.cache.enabled is False
    finally:
        instance.server_close()


def test_an_unwritable_cache_directory_degrades_instead_of_failing(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_bytes(b"")
    cache = SpeechCache(blocker / "tts")

    assert cache.enabled is False
    cache.put("Hello there.", "Test Voice", 178, 4, WAV)
    assert cache.get("Hello there.", "Test Voice", 178, 4) is None


def test_prewarm_renders_new_lines_only(server, tmp_path):
    calls = fake_synthesis(server)
    server.speak("Already spoken.")
    calls.clear()

    script = tmp_path / "lines.txt"
    script.write_text(
        "\n".join(
            [
                "# a comment is not a line",
                "",
                "Already spoken.",
                "A fresh line.",
                "  Another fresh line.  ",
            ]
        ),
        encoding="utf-8",
    )

    assert prewarm(server, script) == 2
    assert calls == ["A fresh line.", "Another fresh line."]
    assert prewarm(server, script) == 0


def test_prewarm_keeps_going_after_one_failure(server, tmp_path):
    attempted = []

    def synthesize(text):
        attempted.append(text)
        if "bad" in text:
            raise OSError("engine unavailable")
        return WAV

    server.synthesize = synthesize
    script = tmp_path / "lines.txt"
    script.write_text("a bad line\na good line\n", encoding="utf-8")

    assert prewarm(server, script) == 1
    assert attempted == ["a bad line", "a good line"]
    assert server.cache.get("a good line", "Test Voice", 178, 4) == WAV
