#!/bin/bash
#SBATCH --account=m4958_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --job-name=fm_p3_dbgB
#SBATCH --output=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/phase3_hardening/logs/dbgB_%j.out
#SBATCH --error=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/phase3_hardening/logs/dbgB_%j.err
#SBATCH --qos=debug
#SBATCH --time=00:29:30

set -euo pipefail

PY="/global/homes/d/danieltm/.conda/envs/collider-env/bin/python"
SUITE="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite"
LOGDIR="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/phase3_hardening/logs"
mkdir -p "${LOGDIR}"

echo "[$(date)] Starting Phase 3 Debug B on $(hostname) (Job ${SLURM_JOB_ID})"
nvidia-smi -L

# GPU 0 (resumable): pretrain_seed ssl_tokens seed 1
(
  echo "[$(date)] [P3B:GPU0] Running pretrain_seed ssl_tokens seed 1..."
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase3_referee_hardening.py" --step pretrain_seed --family ssl_tokens --seed-idx 1 --gpu 0 >> "${LOGDIR}/seed_tok_s1.log" 2>&1
  echo "[$(date)] [P3B:GPU0] DONE!"
) &
P0=$!

# GPU 1 (resumable): pretrain_seed ssl_tokens seed 2
(
  echo "[$(date)] [P3B:GPU1] Running pretrain_seed ssl_tokens seed 2..."
  CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase3_referee_hardening.py" --step pretrain_seed --family ssl_tokens --seed-idx 2 --gpu 0 >> "${LOGDIR}/seed_tok_s2.log" 2>&1
  echo "[$(date)] [P3B:GPU1] DONE!"
) &
P1=$!

# GPU 2 (~8m): bandwidth_zca_sweep
(
  echo "[$(date)] [P3B:GPU2] Running bandwidth_zca_sweep..."
  CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase3_referee_hardening.py" --step bandwidth_zca_sweep --gpu 0 >> "${LOGDIR}/bandwidth_zca_sweep.log" 2>&1
  echo "[$(date)] [P3B:GPU2] DONE!"
) &
P2=$!

# GPU 3 (~6m): finetune_curves (ssl_tokens)
(
  echo "[$(date)] [P3B:GPU3] Running finetune_curves (ssl_tokens)..."
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase3_referee_hardening.py" --step finetune_curves --family ssl_tokens --gpu 0 >> "${LOGDIR}/ft_ssl_tokens.log" 2>&1
  echo "[$(date)] [P3B:GPU3] DONE!"
) &
P3=$!

wait ${P0} ${P1} ${P2} ${P3}
${PY} "${SUITE}/phase3_referee_hardening.py" --step summarize_all >> "${LOGDIR}/summarize.log" 2>&1 || true
echo "[$(date)] Phase 3 Debug B COMPLETED!"
