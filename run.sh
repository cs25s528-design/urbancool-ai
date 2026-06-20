#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

mkdir -p data/processed data/outputs models

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

exec python3 -m uvicorn app.backend.main:app --host "${HOST}" --port "${PORT}" --reload

