#!/bin/bash
set -euo pipefail

PYTHON=/global/homes/d/danieltm/.conda/envs/collider-env/bin/python
SCRIPT=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite/train_and_extract.py
LOGDIR=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs
mkdir -p "${LOGDIR}"

TASK_ID=${SLURM_PROCID:-0}
LOCAL_GPU=${SLURM_LOCALID:-0}
export CUDA_VISIBLE_DEVICES=${LOCAL_GPU}

echo "[$(date)] Starting task ${TASK_ID} on $(hostname) GPU ${LOCAL_GPU}"

case ${TASK_ID} in
  0)
    ${PYTHON} ${SCRIPT} --task train_ssl_kinematic --latent-dim 8 --epochs 20 --batch-size 512 --gpu 0 > "${LOGDIR}/ssl_kinematic_d8.log" 2>&1
    ;;
  1)
    ${PYTHON} ${SCRIPT} --task train_ssl_kinematic --latent-dim 32 --epochs 20 --batch-size 512 --gpu 0 > "${LOGDIR}/ssl_kinematic_d32.log" 2>&1
    ;;
  2)
    ${PYTHON} ${SCRIPT} --task train_ssl_kinematic --latent-dim 64 --epochs 20 --batch-size 512 --gpu 0 > "${LOGDIR}/ssl_kinematic_d64.log" 2>&1
    ;;
  3)
    ${PYTHON} ${SCRIPT} --task train_ssl_kinematic_drop --latent-dim 64 --epochs 20 --batch-size 512 --gpu 0 > "${LOGDIR}/ssl_kinematic_drop_d64.log" 2>&1
    ;;
  4)
    ${PYTHON} ${SCRIPT} --task train_ae_transformer --latent-dim 8 --epochs 25 --batch-size 512 --gpu 0 > "${LOGDIR}/ae_transformer_d8.log" 2>&1
    ;;
  5)
    ${PYTHON} ${SCRIPT} --task train_ae_transformer --latent-dim 32 --epochs 25 --batch-size 512 --gpu 0 > "${LOGDIR}/ae_transformer_d32.log" 2>&1
    ;;
  6)
    ${PYTHON} ${SCRIPT} --task train_ae_transformer --latent-dim 64 --epochs 25 --batch-size 512 --gpu 0 > "${LOGDIR}/ae_transformer_d64.log" 2>&1
    ;;
  7)
    ${PYTHON} ${SCRIPT} --task train_ssl_tokens --latent-dim 8 --epochs 15 --batch-size 512 --gpu 0 > "${LOGDIR}/ssl_tokens_d8.log" 2>&1
    ${PYTHON} ${SCRIPT} --task train_ssl_tokens --latent-dim 32 --epochs 15 --batch-size 512 --gpu 0 > "${LOGDIR}/ssl_tokens_d32.log" 2>&1
    ;;
esac

echo "[$(date)] Completed task ${TASK_ID} on $(hostname) GPU ${LOCAL_GPU}"
