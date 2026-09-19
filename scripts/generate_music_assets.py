"""Generate the repository's small original PCM music cues deterministically."""

from __future__ import annotations

from array import array
import math
from pathlib import Path
import wave


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "bbapps" / "play_sound" / "wavs"
RATE = 16_000


def hz(semitones_from_a4):
    return 440.0 * 2 ** (semitones_from_a4 / 12)


NOTES = {
    "C3": hz(-21), "D3": hz(-19), "E3": hz(-17), "F3": hz(-16),
    "G3": hz(-14), "A3": hz(-12), "B3": hz(-10),
    "C4": hz(-9), "D4": hz(-7), "E4": hz(-5), "F4": hz(-4),
    "G4": hz(-2), "A4": hz(0), "B4": hz(2),
    "C5": hz(3), "D5": hz(5), "E5": hz(7), "G5": hz(10),
}


def render(path, bpm, melody, chords, *, bright=False):
    step_seconds = 60.0 / bpm / 2.0
    step_samples = round(step_seconds * RATE)
    output = array("h")
    for index, note_name in enumerate(melody):
        note = NOTES.get(note_name, 0.0)
        chord = chords[(index // 8) % len(chords)]
        for sample_index in range(step_samples):
            local_t = sample_index / RATE
            absolute_t = (index * step_samples + sample_index) / RATE
            attack = min(1.0, sample_index / max(1, int(0.035 * RATE)))
            release = min(1.0, (step_samples - sample_index) / max(1, int(0.08 * RATE)))
            envelope = attack * release
            value = 0.0
            if note:
                value += 0.34 * math.sin(2 * math.pi * note * local_t)
                value += 0.08 * math.sin(2 * math.pi * note * 2 * local_t)
            for chord_note in chord:
                value += 0.075 * math.sin(2 * math.pi * NOTES[chord_note] * absolute_t)
            bass = NOTES[chord[0]] / 2
            value += 0.12 * math.sin(2 * math.pi * bass * absolute_t)
            if bright and index % 2 == 0:
                # A short deterministic click gives the upbeat cue a pulse.
                value += 0.06 * math.sin(2 * math.pi * 120 * local_t) * math.exp(-18 * local_t)
            sample = max(-1.0, min(1.0, value * envelope))
            output.append(round(sample * 22_000))

    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(RATE)
        target.writeframes(output.tobytes())


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    render(
        OUTPUT / "baymax_calm.wav",
        70,
        (
            "E4", "G4", "C5", "G4", "D4", "G4", "B4", "G4",
            "E4", "A4", "C5", "A4", "D4", "F4", "A4", "F4",
        ) * 2,
        (("C4", "E4", "G4"), ("G3", "B3", "D4"),
         ("A3", "C4", "E4"), ("F3", "A3", "C4")),
    )
    render(
        OUTPUT / "baymax_celebration.wav",
        112,
        (
            "C4", "E4", "G4", "C5", "G4", "E4", "D4", "G4",
            "A4", "C5", "E5", "C5", "G4", "B4", "D5", "G5",
            "E5", "D5", "C5", "G4", "A4", "C5", "D5", "E5",
            "G5", "E5", "D5", "B4", "C5", "G4", "E4", "C4",
        ),
        (("C4", "E4", "G4"), ("G3", "B3", "D4"),
         ("A3", "C4", "E4"), ("F3", "A3", "C4")),
        bright=True,
    )
    for path in (OUTPUT / "baymax_calm.wav", OUTPUT / "baymax_celebration.wav"):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
