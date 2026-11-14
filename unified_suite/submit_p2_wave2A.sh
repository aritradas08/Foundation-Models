#!/bin/bash
#SBATCH --account=m4958_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --job-name=fm_p2_w2A
#SBATCH --output=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/w2A_%j.out
#SBATCH --error=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/w2A_%j.err
#SBATCH --qos=debug
#SBATCH --time=00:29:30

set -euo pipefail

PY="/global/homes/d/danieltm/.conda/envs/collider-env/bin/python"
SUITE="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite"
RES="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results"
LOGDIR="${RES}/logs_v2"
TOK64_NPZ="${RES}/embeddings_v2/ssl_tokens_d64.npz"

echo "[$(date)] Starting Wave 2A on $(hostname) (Job ${SLURM_JOB_ID})"
nvidia-smi -L

# Safety check: if manifest.json wasn't finished in Wave 1, finish extract_all first
if [ ! -f "${RES}/embeddings_v2/manifest.json" ]; then
  echo "[$(date)] Finishing extract_all first..."
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_prepare_and_extract.py" --step extract_all --gpu 0 >> "${LOGDIR}/extract_all.log" 2>&1
fi

# GPU 0: Resume & finish unified ssl_tokens_d64 -> then run probes + all_shape + all_additive + highstats on ssl_tokens_d64
(
  if [ ! -f "${TOK64_NPZ}" ]; then
    echo "[$(date)] [W2A:GPU0] Resuming train_ssl_tokens_d64 from checkpoint..."
    CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_prepare_and_extract.py" --step train_ssl_tokens_d64 --epochs 12 --gpu 0 >> "${LOGDIR}/train_ssl_tokens_d64.log" 2>&1
  fi
  echo "[$(date)] [W2A:GPU0] Running probes + shape + additive + highstats for ssl_tokens_d64..."
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step probes --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/w2_probes_tok64.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_shape --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/w2_shape_tok64.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_additive --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/w2_add_tok64.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/w2_highstats_tok64.log" 2>&1
  echo "[$(date)] [W2A:GPU0] ALL DONE!"
) &
P0=$!

# GPU 1: Ensure paired probes (33 ready models) + all_shape part 1 + all_additive are 100% complete
(
  N_PROBES=$(wc -l < "${RES}/probes_v2/probe_summary.csv" 2>/dev/null || echo 0)
  if [ "${N_PROBES}" -lt 34 ]; then
    echo "[$(date)] [W2A:GPU1] Running paired probes on all ready models..."
    CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step probes --gpu 0 > "${LOGDIR}/w2_probes_all.log" 2>&1
  fi
  echo "[$(date)] [W2A:GPU1] Running all_shape part 1..."
  CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_shape \
    --models supcon_d64 supcon_d32 supcon_d8 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_d32 ssl_kinematic_d8 ssl_kinematic_drop_d64 ssl_tokens_d32 ssl_tokens_d8 ssl_tokens_legacy_d64 ssl_legacy_d64 ssl_legacy_d32 ssl_legacy_d8 raw_summary_d28 pca_summary_d8 phys_handcrafted_d6 mult_only_d1 \
    --gpu 0 > "${LOGDIR}/w2_all_shape_part1.log" 2>&1
  echo "[$(date)] [W2A:GPU1] Running all_additive..."
  CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_additive \
    --models supcon_d64 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_drop_d64 ae_transformer_d64 ae_mlp_d64 random_init_d64 raw_summary_d28 phys_handcrafted_d6 \
    --gpu 0 > "${LOGDIR}/w2_all_additive.log" 2>&1
  echo "[$(date)] [W2A:GPU1] ALL DONE!"
) &
P1=$!

# GPU 2: Ensure scratch_curves + all_shape part 2 + controls_2x2 part 1 are 100% complete
(
  N_SCR=$(wc -l < "${RES}/probes_v2/supervised_scratch_curves.csv" 2>/dev/null || echo 0)
  if [ "${N_SCR}" -lt 61 ]; then
    echo "[$(date)] [W2A:GPU2] Finishing supervised_scratch_d64 curves..."
    CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step scratch_curves --gpu 0 > "${LOGDIR}/w2_scratch_curves.log" 2>&1
  fi
  echo "[$(date)] [W2A:GPU2] Running all_shape part 2..."
  CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_shape \
    --models ae_transformer_d64 ae_transformer_d32 ae_transformer_d8 ae_transformer_postln_d64 ae_transformer_postln_d32 ae_transformer_postln_d8 ae_mlp_d64 ae_mlp_d32 ae_mlp_d8 vqvae_encoder_d128 vqvae_code_d8 vqvae_quant_d8 random_init_d64 random_init_d32 random_init_d8 \
    --gpu 0 > "${LOGDIR}/w2_all_shape_part2.log" 2>&1
  echo "[$(date)] [W2A:GPU2] Running controls_2x2 part 1..."
  CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_controls_2x2_and_masking.py" \
    --models supcon_d64 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_d32 ssl_tokens_d32 ssl_tokens_d8 ae_transformer_d32 ae_mlp_d32 \
    --gpu 0 > "${LOGDIR}/w2_controls_part1.log" 2>&1
  echo "[$(date)] [W2A:GPU2] ALL DONE!"
) &
P2=$!

# GPU 3: Ensure controls_2x2 part 2 + geometry ZCA ablation + ssl_tokens_d64 controls/geometry are 100% complete
(
  echo "[$(date)] [W2A:GPU3] Running controls_2x2 part 2..."
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_controls_2x2_and_masking.py" \
    --models vqvae_encoder_d128 vqvae_code_d8 vqvae_quant_d8 raw_summary_d28 pca_summary_d8 phys_handcrafted_d6 mult_only_d1 random_init_d64 \
    --gpu 0 > "${LOGDIR}/w2_controls_part2.log" 2>&1
  echo "[$(date)] [W2A:GPU3] Running geometry ZCA ablation..."
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step geometry \
    --models supcon_d64 supervised_e2e_d64 ssl_kinematic_d64 ssl_tokens_d32 ae_transformer_d32 ae_mlp_d32 vqvae_code_d8 vqvae_quant_d8 raw_summary_d28 phys_handcrafted_d6 \
    --gpu 0 > "${LOGDIR}/w2_geometry.log" 2>&1
  echo "[$(date)] [W2A:GPU3] Waiting for ssl_tokens_d64.npz to run controls_2x2 + geometry on ssl_tokens_d64..."
  while [ ! -f "${TOK64_NPZ}" ]; do sleep 5; done
  sleep 3
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_controls_2x2_and_masking.py" --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/w2_controls_tok64.log" 2>&1
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step geometry --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/w2_geometry_tok64.log" 2>&1
  echo "[$(date)] [W2A:GPU3] ALL DONE!"
) &
P3=$!

wait ${P0} ${P1} ${P2} ${P3}
echo "[$(date)] Wave 2A COMPLETED 100% OF ALL TASKS!"
