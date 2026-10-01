#!/usr/bin/env bash
set -euo pipefail

: "${STABLEWM_HOME:?Set STABLEWM_HOME to the directory that will hold the PushT dataset.}"

mkdir -p "$STABLEWM_HOME"
hf download quentinll/lewm-pusht pusht_expert_train.h5.zst \
  --repo-type dataset \
  --local-dir "$STABLEWM_HOME"

if [[ ! -f "$STABLEWM_HOME/pusht_expert_train.h5" ]]; then
  unzstd "$STABLEWM_HOME/pusht_expert_train.h5.zst"
fi

echo "Dataset ready: $STABLEWM_HOME/pusht_expert_train.h5"
