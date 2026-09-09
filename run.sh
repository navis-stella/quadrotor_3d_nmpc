#!/usr/bin/env bash
# run.sh — run Python inside the WSL acados venv from Windows (Git Bash / Claude Code).
#
# The acados C libraries and the acados_template package live in WSL (Ubuntu),
# not on Windows. The real work is done by ~/acados_run.sh *inside* WSL, which
# exports the acados env vars, activates ~/acados_env, and cd's to the project.
# This wrapper just forwards your arguments to it.
#
# Usage (from Git Bash on Windows):
#   ./run.sh simulate_dare.py
#   ./run.sh -m pytest -q
#
# For python -c one-liners with quotes, run inside a WSL shell instead:
#   wsl
#   ~/acados_run.sh -c "import acados_template; print('ok')"
set -euo pipefail
export MSYS_NO_PATHCONV=1
exec wsl.exe -e bash /home/celestialnavis/acados_run.sh "$@"
