#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 TRAIN_SEED" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for ARM in grey stable unpredictable; do
  "$HERE/run_one.sh" "$ARM" "$1"
done
