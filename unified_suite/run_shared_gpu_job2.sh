#!/bin/bash
set -euo pipefail

PYTHON=/global/homes/d/danieltm/.conda/envs/collider-env/bin/python
SCRIPT=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite/train_and_extract.py
LOGDIR=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs
mkdir -p "${LOGDIR}"

echo "[$(date)] Starting Job 2 on $(hostname)"

CUDA_VISIBLE_DEVICES=0 ${PYTHON} ${SCRIPT} --task train_ae_transformer --dim 8 --epochs 15 --gpu 0 > "${LOGDIR}/ae_transformer_d8.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 ${PYTHON} ${SCRIPT} --task train_ae_transformer --dim 32 --epochs 15 --gpu 0 > "${LOGDIR}/ae_transformer_d32.log" 2>&1 &
PID1=$!

wait ${PID0}
CUDA_VISIBLE_DEVICES=0 ${PYTHON} ${SCRIPT} --task train_ae_transformer --dim 64 --epochs 15 --gpu 0 > "${LOGDIR}/ae_transformer_d64.log" 2>&1 &
PID2=$!

wait ${PID1}
CUDA_VISIBLE_DEVICES=1 ${PYTHON} ${SCRIPT} --task train_ssl_tokens --dim 8 --epochs 12 --gpu 0 > "${LOGDIR}/ssl_tokens_d8.log" 2>&1
sleep 3600
wait ${PID2}
echo "[$(date)] Completed Job 2 on $(hostname)"
