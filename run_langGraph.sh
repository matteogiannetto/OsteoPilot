#!/bin/bash
# Simple script to run langgraph dev from the project directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/multiagent"

echo "📂 Changing directory to multiagent..."
cd "$PROJECT_DIR"

# Explicitly load the .env file for the langgraph command
# This ensures variables are available before Python starts (for langsmith)
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

if ! command -v langgraph >/dev/null 2>&1; then
    echo "❌ LangGraph CLI not found in the active environment."
    echo "   Install it from this directory with:"
    echo "   python -m pip install -e '.[dev]'"
    exit 127
fi

echo "🚀 Starting langgraph in dev mode..."
LOG_LEVEL=WARNING langgraph dev
