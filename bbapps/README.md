# bbapps

The apps and scripts that run on a BracketBot, collected in one place so everyone can read, run, and document them.

Each app lives in its own folder with a `main.py` entry point. The standalone scripts sit at the top level.

## Standalone scripts

| Script | Purpose |
| --- | --- |
| `teleop.py` | Manual driving |
| `leader_follower_teleop.py` | Leader/follower arm control |
| `tune_odrive.py` | Motor controller (ODrive) tuning |
| `depth_calibration_check.py` | Depth sensor calibration check |
| `slam_debug_web.py` | SLAM / mapping debug web view |

Hidden files: `.autostart` is probably what the bot launches on boot. `.python-version` pins the Python version.

## App folders

| Folder | What's in it |
| --- | --- |
| `examples/` | 11 small `view_*.py` demos, one each for arms, camera, depth, IK, IMU, LED, mic, Quest, speaker, USB, and wakeword |
| `greeter/` | Local Whisper/eSpeak or optional Gemini mic/speaker transport, deterministic voice-command routing, OpenRouter/Browserbase conversation, and wave/hug/handshake/fist-bump movements |
| `emotion_greeter/` | Robot-head-camera YOLO + visible-expression cue with a local BBOS speaker response |
| `mimic/` | `main.py` + `recordings/` (dance, wave `.json`) |
| `nav/` | `main.py` (~161 KB, the biggest file) + `planner.py`, `reloc_planner.py`, `reloc_geom.py` |
| `quest_teleop/` | `main.py` + `scripts/` (homing, tracking, quest, quat, sound) + `wavs/` for mode sounds |
| `play_sound/` | `main.py`, `web_ui.py` + `wavs/` (about 30 sound clips) |
| `low_battery/` | `main.py` + `wavs/low_battery.wav` |
| `inference/` | Policy/VLM setup: `live_inference.py`, `vlm.py`, `vlm_inference.py`, `bracketbot_adapter.py`, `run_inference.sh`, `prompt.md`, plus the `bb_relay/` and `policy_client/` gRPC packages. Has its own `pyproject.toml` and `uv.lock`. |

## Secrets

No keys are committed. Apps read them from the environment:

- `greeter/` needs `OPENROUTER_API_KEY` for non-action speech and `BROWSERBASE_API_KEY` for live web search; local Whisper/eSpeak voice needs no speech API key
- `inference/` needs `--api-key`, `$BB_API_KEY`, or `/etc/BB_API_KEY`

## Browsing the tree

```sh
find . -type f -not -path '*/__pycache__/*' | sort
```

## Contributing docs

Doc improvements are welcome. Good places to start: a short README inside each app folder that covers what it does, how to launch it, and which topics or hardware it uses.

## License

MIT. See [LICENSE](LICENSE).
