#!/bin/bash
#SBATCH --account=m4958_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --job-name=fm_p2_w4B
#SBATCH --output=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/w4B_%j.out
#SBATCH --error=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/w4B_%j.err
#SBATCH --qos=debug
#SBATCH --time=00:29:30

set -euo pipefail

PY="/global/homes/d/danieltm/.conda/envs/collider-env/bin/python"
SUITE="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite"
LOGDIR="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2"

echo "[$(date)] Starting Wave 4B on $(hostname) (Job ${SLURM_JOB_ID})"
nvidia-smi -L

CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_controls_2x2_and_masking.py" \
  --models ae_mlp_d32 ae_transformer_d32 supcon_d64 supervised_e2e_d64 \
  --gpu 0 > "${LOGDIR}/w4_controls_0.log" 2>&1 &
C0=$!

CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_controls_2x2_and_masking.py" \
  --models ssl_tokens_d64 ssl_tokens_d32 ssl_kinematic_d64 ssl_kinematic_d32 \
  --gpu 0 > "${LOGDIR}/w4_controls_1.log" 2>&1 &
C1=$!

CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_controls_2x2_and_masking.py" \
  --models ssl_tokens_d8 vqvae_encoder_d128 vqvae_code_d8 vqvae_quant_d8 \
  --gpu 0 > "${LOGDIR}/w4_controls_2.log" 2>&1 &
C2=$!

CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_controls_2x2_and_masking.py" \
  --models raw_summary_d28 pca_summary_d8 phys_handcrafted_d6 mult_only_d1 random_init_d64 \
  --gpu 0 > "${LOGDIR}/w4_controls_3.log" 2>&1 &
C3=$!

wait ${C0} ${C1} ${C2} ${C3}
echo "[$(date)] Wave 4B COMPLETED 100% OF ALL TASKS!"
