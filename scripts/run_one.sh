#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {grey|stable|unpredictable} TRAIN_SEED" >&2
  exit 2
fi

ARM=$1
TRAIN_SEED=$2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/outputs}
EPOCHS=${EPOCHS:-2}
EVAL_SEEDS=${EVAL_SEEDS:-"42 4242"}

: "${STABLEWM_HOME:?Set STABLEWM_HOME to the directory containing pusht_expert_train.h5.}"

case "$ARM" in
  grey)
    CONDITION=neutral
    RHO=1.0
    ;;
  stable)
    CONDITION=markov
    RHO=1.0
    ;;
  unpredictable)
    CONDITION=markov
    RHO=0.0
    ;;
  *)
    echo "unknown arm: $ARM" >&2
    exit 2
    ;;
esac

RUN_NAME="${ARM}_seed${TRAIN_SEED}"
RUN_DIR="$OUTPUT_ROOT/runs/$RUN_NAME"
EVAL_DIR="$OUTPUT_ROOT/evals"
CHECKPOINT="$RUN_DIR/${RUN_NAME}_epoch_${EPOCHS}_object.ckpt"
mkdir -p "$RUN_DIR" "$EVAL_DIR"

export WANDB_MODE=disabled
export MUJOCO_GL=${MUJOCO_GL:-egl}
export TOKENIZERS_PARALLELISM=false

NUISANCE_ARGS=(
  nuisance.condition="$CONDITION"
  nuisance.mode=pad
  nuisance.geometry=patch
  nuisance.border=14
  nuisance.patch_size=14
  nuisance.n_patches=1
  nuisance.amplitude=1.0
  nuisance.rho="$RHO"
  nuisance.frame_unit=model
  nuisance.base_seed=20260922
)

if [[ ! -f "$CHECKPOINT" ]]; then
  "$PYTHON" -u "$ROOT/train.py" \
    --config-name=lewm \
    data=pusht \
    seed="$TRAIN_SEED" \
    trainer.max_epochs="$EPOCHS" \
    trainer.devices=1 \
    output_model_name="$RUN_NAME" \
    run_dir="$RUN_DIR" \
    cache_dir="$STABLEWM_HOME" \
    "${NUISANCE_ARGS[@]}"
else
  echo "Reusing checkpoint: $CHECKPOINT"
fi

for EVAL_SEED in $EVAL_SEEDS; do
  TAG="${RUN_NAME}_eval${EVAL_SEED}"
  REPORT="$EVAL_DIR/$TAG.json"
  if [[ -f "$REPORT" ]]; then
    echo "Reusing evaluation: $REPORT"
    continue
  fi
  "$PYTHON" -u "$ROOT/eval.py" \
    --config-name=pusht \
    policy="${CHECKPOINT%_object.ckpt}" \
    seed="$EVAL_SEED" \
    cache_dir="$STABLEWM_HOME" \
    nuisance.stream=1 \
    eval.goal_nuisance=matched \
    output.dir="$EVAL_DIR" \
    output.tag="$TAG" \
    output.filename="$TAG.txt" \
    "${NUISANCE_ARGS[@]}"
done
