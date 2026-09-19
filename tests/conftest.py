import sys
from pathlib import Path

# Robot-side scripts are deployed flat to /tmp and import their siblings by bare
# name (``import follow_core``); put scripts/ on the path so tests do the same.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
