#!/bin/bash
#SBATCH --account=m4958_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --job-name=fm_p3_dbgA
#SBATCH --output=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/phase3_hardening/logs/dbgA_%j.out
#SBATCH --error=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/phase3_hardening/logs/dbgA_%j.err
#SBATCH --qos=debug
#SBATCH --time=00:29:30

set -euo pipefail

PY="/global/homes/d/danieltm/.conda/envs/collider-env/bin/python"
SUITE="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite"
LOGDIR="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/phase3_hardening/logs"
mkdir -p "${LOGDIR}"

echo "[$(date)] Starting Phase 3 Debug A on $(hostname) (Job ${SLURM_JOB_ID})"
nvidia-smi -L

# GPU 0 (~11m): supcon_subspace (~3m) + finetune_curves ssl_kinematic (~8m)
(
  echo "[$(date)] [P3A:GPU0] Running supcon_subspace..."
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase3_referee_hardening.py" --step supcon_subspace --gpu 0 >> "${LOGDIR}/supcon_subspace.log" 2>&1
  echo "[$(date)] [P3A:GPU0] Running finetune_curves (ssl_kinematic)..."
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase3_referee_hardening.py" --step finetune_curves --family ssl_kinematic --gpu 0 >> "${LOGDIR}/ft_ssl_kinematic.log" 2>&1
  echo "[$(date)] [P3A:GPU0] DONE!"
) &
P0=$!

# GPU 1 (resumable): broken_aug_t128 (20 epochs + extract + probes + NPLM)
(
  echo "[$(date)] [P3A:GPU1] Running broken_aug_t128..."
  CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase3_referee_hardening.py" --step broken_aug_t128 --gpu 0 >> "${LOGDIR}/broken_aug_t128.log" 2>&1
  echo "[$(date)] [P3A:GPU1] DONE!"
) &
P1=$!

# GPU 2 (resumable): pretrain_seed ssl_kinematic seed 1
(
  echo "[$(date)] [P3A:GPU2] Running pretrain_seed ssl_kinematic seed 1..."
  CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase3_referee_hardening.py" --step pretrain_seed --family ssl_kinematic --seed-idx 1 --gpu 0 >> "${LOGDIR}/seed_kin_s1.log" 2>&1
  echo "[$(date)] [P3A:GPU2] DONE!"
) &
P2=$!

# GPU 3 (resumable): pretrain_seed ssl_kinematic seed 2
(
  echo "[$(date)] [P3A:GPU3] Running pretrain_seed ssl_kinematic seed 2..."
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase3_referee_hardening.py" --step pretrain_seed --family ssl_kinematic --seed-idx 2 --gpu 0 >> "${LOGDIR}/seed_kin_s2.log" 2>&1
  echo "[$(date)] [P3A:GPU3] DONE!"
) &
P3=$!

wait ${P0} ${P1} ${P2} ${P3}
${PY} "${SUITE}/phase3_referee_hardening.py" --step summarize_all >> "${LOGDIR}/summarize.log" 2>&1 || true
echo "[$(date)] Phase 3 Debug A COMPLETED!"
