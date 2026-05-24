#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-ctf-one-shot-cf13ddd-$(date +%Y%m%d-%H%M%S)}"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/torchinductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/triton_cache}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

exec torchrun --standalone --nproc_per_node=8 train_gpt_ctf.py --run_id "$RUN_ID"
