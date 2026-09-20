"""Guards on the robot's fixed spoken lines and their rendered audio."""

import json
import wave

from canned_lines import LINES, LINES_BY_ID, LINES_DIR, missing_assets


MANIFEST = LINES_DIR / "manifest.json"


def test_line_catalog_is_unique_and_single_key():
    assert len(LINES) == len(LINES_BY_ID)
    keys = [line.key for line in LINES]
    assert len(keys) == len(set(keys))
    for line in LINES:
        assert line.id.startswith("line-")
        assert len(line.key) == 1 and line.key.isalpha() and line.key.islower()
        assert line.label and line.summary
        assert len(line.text.split()) >= 5


def test_every_line_is_rendered():
    assert missing_assets() == (), "run python3 scripts/generate_line_assets.py"


def test_rendered_audio_matches_the_recorded_text():
    """A line edited without re-rendering would otherwise go unnoticed."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest) == set(LINES_BY_ID)
    for line in LINES:
        entry = manifest[line.id]
        assert entry["text"] == line.text, f"{line.id} needs re-rendering"
        assert entry["file"] == line.filename
        assert entry["bytes"] == line.path.stat().st_size


def test_rendered_audio_plays_on_the_robot_speaker():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for line in LINES:
        with wave.open(str(line.path)) as audio:
            # robot_effect.py refuses anything that is not the speaker's own
            # mono 16 kHz format.
            assert audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            assert audio.getframerate() == 16_000
            assert audio.getframerate() == manifest[line.id]["sample_rate"]
            assert 2.0 < audio.getnframes() / audio.getframerate() < 30.0


def test_lines_never_claim_medical_authority():
    for line in LINES:
        spoken = line.text.lower()
        for claim in ("i am a doctor", "i am a nurse", "diagnose", "prescribe"):
            assert claim not in spoken, f"{line.id} must not say {claim!r}"
