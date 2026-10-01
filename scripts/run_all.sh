#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS=${SEEDS:-"4711 8123 2659 6011 7297 8837 9187 10427 11717 12829"}

for SEED in $SEEDS; do
  "$HERE/run_seed.sh" "$SEED"
done
