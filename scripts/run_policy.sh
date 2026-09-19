#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
# GLFW uses Python's main thread on macOS.
exec uv run --locked python -m bbsim "$@"
