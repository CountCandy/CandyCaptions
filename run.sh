#!/usr/bin/env sh
# CandyCaption launcher for Linux. Needs uv (https://docs.astral.sh/uv/).
cd "$(dirname "$0")" || exit 1
echo "CandyCaption: http://localhost:8200  (Ctrl+C to stop)"
( for _ in $(seq 1 60); do
    if curl -fs http://localhost:8200/health >/dev/null 2>&1; then
      command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:8200 >/dev/null 2>&1
      exit 0
    fi
    sleep 1
  done ) &
exec uv run uvicorn candycaption.server:app --host 127.0.0.1 --port 8200
