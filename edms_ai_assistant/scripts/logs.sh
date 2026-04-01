#!/usr/bin/env bash
SERVICE="${1:-orchestrator}"
cd "$(dirname "${BASH_SOURCE[0]}")/.." 
docker compose logs --tail=200 -f "$SERVICE"
