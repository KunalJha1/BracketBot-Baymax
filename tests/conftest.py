import sys
from pathlib import Path
import subprocess

import pytest

# Robot-side scripts are deployed flat to /tmp and import their siblings by bare
# name (``import follow_core``); put scripts/ on the path so tests do the same.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture
def python_instead_of_uv(monkeypatch):
    """Launch real test child processes without a POSIX-only shell wrapper."""
    popen = subprocess.Popen

    def launch(command, **kwargs):
        assert command[:3] == ["test-uv", "run", "--quiet"]
        return popen([sys.executable, *command[3:]], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)
    return "test-uv"
