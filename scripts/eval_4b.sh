#!/usr/bin/env bash
# Standalone evaluation: initial model, one checkpoint, or asynchronous watching.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
args=(--model "${MODEL_PATH:-$PROJECT_ROOT/../models/Qwen3-4B}"
      --data "${DATA_ROOT:-$PROJECT_ROOT/data}/math"
      --output "${EVAL_OUTPUT:-$PROJECT_ROOT/../rlopsd-evaluation/$(date +%Y%m%d-%H%M%S)}"
      --gpus "${EVAL_GPUS:-2}" --tensor-parallel-size "${EVAL_TP:-1}"
      --prompt-format native --no-thinking --n 12 --max-tokens 32000 --max-model-len 32768
      --gpu-memory-utilization "${EVAL_MEMORY_UTILIZATION:-0.8}"
      --max-num-seqs "${EVAL_MAX_NUM_SEQS:-8}")
exec "$PYTHON" -m local.math_m2.evaluate "${args[@]}" "$@"
