"""The robot's pre-rendered spoken lines, shared by the dashboard and generator.

These are the few sentences BracketBot says the same way every time: the ones
that carry a demo and must not depend on the network, the wake word, the
speech recognizer, or a model round trip.  The text lives here, in code, and
``scripts/generate_line_assets.py`` renders each one to a WAV that the existing
allowlisted sound path plays.  Nothing here is generated at run time.

Every line must stay honest about what the robot is: warm, plainly spoken, and
never claiming medical authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINES_DIR = ROOT / "assets" / "lines"


@dataclass(frozen=True)
class CannedLine:
    """One fixed spoken line and the dashboard button that plays it."""

    id: str
    label: str
    summary: str
    key: str
    text: str

    @property
    def filename(self) -> str:
        return f"{self.id}.wav"

    @property
    def path(self) -> Path:
        return LINES_DIR / self.filename


LINES = (
    CannedLine(
        id="line-intro",
        label="Introduce yourself",
        summary="BracketBot says who it is and what it is for",
        key="i",
        text=(
            "Hello. I am BracketBot, an at home care companion. "
            "I can talk with you, keep track of your reminders, "
            "and help with small tasks around the room. "
            "I am not a doctor, so for anything medical "
            "I will always point you to a real one."
        ),
    ),
    CannedLine(
        id="line-capabilities",
        label="What I can do",
        summary="A short spoken tour of the robot's real abilities",
        key="a",
        text=(
            "Here is what I can do. I can set a reminder and keep it running "
            "while I work on something else. I can answer questions out loud. "
            "I can wave, shake your hand, or give you a hug if you ask me for one. "
            "And I can pick things up from a table and put them away."
        ),
    ),
    CannedLine(
        id="line-comfort",
        label="Comfort line",
        summary="Offers company and a hug, without assuming how someone feels",
        key="e",
        text=(
            "You seem a little quiet. I am right here with you, "
            "and I am happy to just keep you company for a while. "
            "If you would like a hug, ask me for one."
        ),
    ),
    CannedLine(
        id="line-farewell",
        label="Farewell line",
        summary="Closes the visit and points back at the pending reminder",
        key="y",
        text=(
            "Thank you for spending time with me today. "
            "Take good care of yourself, and do not forget your reminder. "
            "Goodbye for now."
        ),
    ),
)

LINES_BY_ID = {line.id: line for line in LINES}


def missing_assets() -> tuple[CannedLine, ...]:
    """Lines whose rendered audio is not on disk yet."""
    return tuple(line for line in LINES if not line.path.is_file())
