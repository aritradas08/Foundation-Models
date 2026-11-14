#!/bin/bash
#SBATCH --account=m4958_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --job-name=fm_p2_w3B
#SBATCH --output=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/w3B_%j.out
#SBATCH --error=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/w3B_%j.err
#SBATCH --qos=debug
#SBATCH --time=00:29:30

set -euo pipefail

PY="/global/homes/d/danieltm/.conda/envs/collider-env/bin/python"
SUITE="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite"
LOGDIR="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2"

echo "[$(date)] Starting Wave 3B on $(hostname) (Job ${SLURM_JOB_ID})"
nvidia-smi -L

# GPU 0 (~20m): highstats B0 (reuses cached *_highstats.npz from Wave 2B!)
CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats \
  --models supcon_d64 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_d32 \
  --gpu 0 > "${LOGDIR}/w3_highstats_B0.log" 2>&1 &
B0=$!

# GPU 1 (~20m): highstats B1 (reuses cached *_highstats.npz from Wave 2B!)
CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats \
  --models ssl_tokens_d32 ssl_tokens_d8 ae_transformer_d32 ae_mlp_d32 \
  --gpu 0 > "${LOGDIR}/w3_highstats_B1.log" 2>&1 &
B1=$!

# GPU 2 (~22m): highstats B2 (reuses cached *_highstats.npz!) -> geometry ZCA ablation part 2
(
  CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats \
    --models vqvae_code_d8 vqvae_quant_d8 raw_summary_d28 pca_summary_d8 \
    --gpu 0 > "${LOGDIR}/w3_highstats_B2.log" 2>&1
  CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step geometry \
    --models ae_transformer_d32 vqvae_code_d8 vqvae_quant_d8 raw_summary_d28 phys_handcrafted_d6 \
    --gpu 0 > "${LOGDIR}/w3_geometry_2.log" 2>&1
) &
B2=$!

# GPU 3 (~22m): highstats B3 (reuses cached *_highstats.npz!) -> all_shape part 1b -> all_additive
(
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats \
    --models phys_handcrafted_d6 mult_only_d1 random_init_d64 \
    --gpu 0 > "${LOGDIR}/w3_highstats_B3.log" 2>&1
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_shape \
    --models ssl_tokens_d8 ssl_tokens_legacy_d64 ssl_legacy_d64 ssl_legacy_d32 ssl_legacy_d8 raw_summary_d28 pca_summary_d8 phys_handcrafted_d6 mult_only_d1 \
    --gpu 0 > "${LOGDIR}/w3_all_shape_1b.log" 2>&1
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_additive \
    --models supcon_d64 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_drop_d64 ae_transformer_d64 ae_mlp_d64 random_init_d64 raw_summary_d28 phys_handcrafted_d6 \
    --gpu 0 > "${LOGDIR}/w3_all_additive.log" 2>&1
) &
B3=$!

wait ${B0} ${B1} ${B2} ${B3}
echo "[$(date)] Wave 3B COMPLETED 100% OF ALL TASKS!"
