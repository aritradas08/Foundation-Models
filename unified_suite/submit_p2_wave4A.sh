#!/bin/bash
#SBATCH --account=m4958_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --job-name=fm_p2_w4A
#SBATCH --output=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/w4A_%j.out
#SBATCH --error=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/w4A_%j.err
#SBATCH --qos=debug
#SBATCH --time=00:29:30

set -euo pipefail

PY="/global/homes/d/danieltm/.conda/envs/collider-env/bin/python"
SUITE="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite"
LOGDIR="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2"

echo "[$(date)] Starting Wave 4A on $(hostname) (Job ${SLURM_JOB_ID})"
nvidia-smi -L

# GPU 0 (~12m): Remaining probes half 1 -> Remaining 2 highstats models (pca_summary_d8, ssl_tokens_d64)
(
  echo "[$(date)] [W4A:GPU0] Running remaining probes half 1..."
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step probes \
    --models supcon_d64 supcon_d32 supcon_d8 ssl_tokens_legacy_d64 \
    --gpu 0 > "${LOGDIR}/w4_probes_1.log" 2>&1
  echo "[$(date)] [W4A:GPU0] Running remaining 2 highstats models..."
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats \
    --models pca_summary_d8 ssl_tokens_d64 \
    --gpu 0 > "${LOGDIR}/w4_highstats_rem.log" 2>&1
  echo "[$(date)] [W4A:GPU0] ALL DONE!"
) &
P0=$!

# GPU 1 (~12m): Remaining probes half 2 -> Full all_shape pass (merges 25 cached + runs 9 remaining)
(
  echo "[$(date)] [W4A:GPU1] Running remaining probes half 2..."
  CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step probes \
    --models supervised_e2e_d64 vqvae_code_d8 vqvae_encoder_d128 vqvae_quant_d8 \
    --gpu 0 > "${LOGDIR}/w4_probes_2.log" 2>&1
  echo "[$(date)] [W4A:GPU1] Running full all_shape (reusing 25 cached + 9 remaining)..."
  CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_shape \
    --gpu 0 > "${LOGDIR}/w4_all_shape_full.log" 2>&1
  echo "[$(date)] [W4A:GPU1] ALL DONE!"
) &
P1=$!

# GPU 2 (~12m): Full all_additive (10 models) -> Geometry ZCA whitening part 1 (5 models)
(
  echo "[$(date)] [W4A:GPU2] Running all_additive..."
  CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_additive \
    --gpu 0 > "${LOGDIR}/w4_all_additive.log" 2>&1
  echo "[$(date)] [W4A:GPU2] Running geometry ZCA part 1..."
  CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step geometry \
    --models supcon_d64 supervised_e2e_d64 ssl_kinematic_d64 ssl_tokens_d64 ssl_tokens_d32 \
    --gpu 0 > "${LOGDIR}/w4_geometry_1.log" 2>&1
  echo "[$(date)] [W4A:GPU2] ALL DONE!"
) &
P2=$!

# GPU 3 (~8m): Geometry ZCA whitening part 2 (6 models)
(
  echo "[$(date)] [W4A:GPU3] Running geometry ZCA part 2..."
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step geometry \
    --models ae_transformer_d32 ae_mlp_d32 vqvae_code_d8 vqvae_quant_d8 raw_summary_d28 phys_handcrafted_d6 \
    --gpu 0 > "${LOGDIR}/w4_geometry_2.log" 2>&1
  echo "[$(date)] [W4A:GPU3] ALL DONE!"
) &
P3=$!

wait ${P0} ${P1} ${P2} ${P3}

# Final 5-second consolidation of all 34 cached probe JSONs + scratch curves
CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step probes --gpu 0 >> "${LOGDIR}/w4_probes_1.log" 2>&1
CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step scratch_curves --gpu 0 >> "${LOGDIR}/w4_probes_1.log" 2>&1

echo "[$(date)] Wave 4A COMPLETED 100% OF ALL TASKS!"
