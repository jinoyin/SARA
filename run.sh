#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
GPUS="${GPUS:-[0]}"
SEED="${SEED:-1024}"
CKPT="${CKPT:-${PROJECT_DIR}/ckpts/ultra_50g.pth}"

SYM_RATIO_THRESHOLD="${SYM_RATIO_THRESHOLD:-0.5}"
SYM_SCORE_THRESHOLD="${SYM_SCORE_THRESHOLD:-0.4}"
COOCCUR_RATIO_THRESHOLD="${COOCCUR_RATIO_THRESHOLD:-0.1}"
COOCCUR_SCORE_THRESHOLD="${COOCCUR_SCORE_THRESHOLD:-0.4}"
MIN_COOCCUR_SUPPORT="${MIN_COOCCUR_SUPPORT:-5}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2}"
PATIENCE="${PATIENCE:-3}"

CONFIGS=(
    "${PROJECT_DIR}/config/pretrain_nl_v1_ind.yaml"
    "${PROJECT_DIR}/config/pretrain_nl_v2_ind.yaml"
    "${PROJECT_DIR}/config/pretrain_nl_v3_ind.yaml"
    "${PROJECT_DIR}/config/pretrain_nl_v4_ind.yaml"
)

export CUDA_VISIBLE_DEVICES="${GPU}"
cd "${PROJECT_DIR}"

for config in "${CONFIGS[@]}"; do
    "${PYTHON_BIN}" -u script/self_evolution.py \
        -c "${config}" \
        -s "${SEED}" \
        --gpus "${GPUS}" \
        --ckpt "${CKPT}" \
        --sym_ratio_threshold "${SYM_RATIO_THRESHOLD}" \
        --sym_score_threshold "${SYM_SCORE_THRESHOLD}" \
        --cooccur_ratio_threshold "${COOCCUR_RATIO_THRESHOLD}" \
        --cooccur_score_threshold "${COOCCUR_SCORE_THRESHOLD}" \
        --min_cooccur_support "${MIN_COOCCUR_SUPPORT}" \
        --max_iterations "${MAX_ITERATIONS}" \
        --patience "${PATIENCE}"
done

