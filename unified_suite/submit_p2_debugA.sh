#!/bin/bash
#SBATCH --account=m4958_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --job-name=fm_p2_dbgA
#SBATCH --output=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/dbgA_%j.out
#SBATCH --error=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/dbgA_%j.err
#SBATCH --qos=debug
#SBATCH --time=00:29:30

set -euo pipefail

PY="/global/homes/d/danieltm/.conda/envs/collider-env/bin/python"
SUITE="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite"
LOGDIR="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2"
MANIFEST="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/embeddings_v2/manifest.json"
TOK64_NPZ="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/embeddings_v2/ssl_tokens_d64.npz"
mkdir -p "${LOGDIR}"

echo "[$(date)] Starting Debug Job A on $(hostname) (Job ${SLURM_JOB_ID})"
nvidia-smi -L

# GPU 0 pipeline: extract_all -> probes (33 models) -> all_shape (part 1) -> all_additive
(
  echo "[$(date)] [A:GPU0] Starting extract_all..."
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_prepare_and_extract.py" --step extract_all --gpu 0 > "${LOGDIR}/extract_all.log" 2>&1
  echo "[$(date)] [A:GPU0] extract_all DONE! Starting paired probes..."
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step probes --gpu 0 > "${LOGDIR}/probes_33.log" 2>&1
  echo "[$(date)] [A:GPU0] probes DONE! Starting all_shape part 1..."
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_shape \
    --models supcon_d64 supcon_d32 supcon_d8 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_d32 ssl_kinematic_d8 ssl_kinematic_drop_d64 ssl_tokens_d32 ssl_tokens_d8 ssl_tokens_legacy_d64 ssl_legacy_d64 ssl_legacy_d32 ssl_legacy_d8 raw_summary_d28 pca_summary_d8 phys_handcrafted_d6 mult_only_d1 \
    --gpu 0 > "${LOGDIR}/nplm_all_shape_part1.log" 2>&1
  echo "[$(date)] [A:GPU0] all_shape part 1 DONE! Starting all_additive..."
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_additive \
    --models supcon_d64 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_drop_d64 ae_transformer_d64 ae_mlp_d64 random_init_d64 raw_summary_d28 phys_handcrafted_d6 \
    --gpu 0 > "${LOGDIR}/nplm_all_additive.log" 2>&1
  echo "[$(date)] [A:GPU0] ALL DONE!"
) &
P0=$!

# Wait for canonical 1,050,000-event /pscratch cache (~75s) before launching GPU 1 & GPU 2
while ! ls /pscratch/sd/d/danieltm/fm_cache/eval_downstream_1050k_mt256_lblall_*.npz >/dev/null 2>&1; do
  sleep 4
done
sleep 4

# GPU 1 pipeline: train unified ssl_tokens_d64 (12 epochs) + extract -> then ssl_tokens_d64 probes/NPLM if time remains
(
  echo "[$(date)] [A:GPU1] Starting train_ssl_tokens_d64..."
  CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_prepare_and_extract.py" --step train_ssl_tokens_d64 --epochs 12 --gpu 0 > "${LOGDIR}/train_ssl_tokens_d64.log" 2>&1
  echo "[$(date)] [A:GPU1] train_ssl_tokens_d64 DONE! Running ssl_tokens_d64 probes + shape/additive/highstats..."
  CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step probes --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/probes_tok64.log" 2>&1
  CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_shape --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/nplm_shape_tok64.log" 2>&1
  CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_additive --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/nplm_add_tok64.log" 2>&1
  CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/nplm_highstats_tok64.log" 2>&1
  echo "[$(date)] [A:GPU1] ALL DONE!"
) &
P1=$!

# GPU 2 pipeline: supervised_scratch_d64 low-label curves -> wait manifest -> all_shape part 2 -> controls_2x2 part 1
(
  echo "[$(date)] [A:GPU2] Starting supervised_scratch_d64 curves..."
  CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step scratch_curves --gpu 0 > "${LOGDIR}/scratch_curves.log" 2>&1
  echo "[$(date)] [A:GPU2] scratch_curves DONE! Waiting for manifest.json..."
  while [ ! -f "${MANIFEST}" ]; do sleep 4; done
  echo "[$(date)] [A:GPU2] Starting all_shape part 2..."
  CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_shape \
    --models ae_transformer_d64 ae_transformer_d32 ae_transformer_d8 ae_transformer_postln_d64 ae_transformer_postln_d32 ae_transformer_postln_d8 ae_mlp_d64 ae_mlp_d32 ae_mlp_d8 vqvae_encoder_d128 vqvae_code_d8 vqvae_quant_d8 random_init_d64 random_init_d32 random_init_d8 \
    --gpu 0 > "${LOGDIR}/nplm_all_shape_part2.log" 2>&1
  echo "[$(date)] [A:GPU2] all_shape part 2 DONE! Starting controls_2x2 part 1..."
  CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_controls_2x2_and_masking.py" \
    --models supcon_d64 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_d32 ssl_tokens_d32 ssl_tokens_d8 ae_transformer_d32 ae_mlp_d32 \
    --gpu 0 > "${LOGDIR}/controls_2x2_part1.log" 2>&1
  echo "[$(date)] [A:GPU2] ALL DONE!"
) &
P2=$!

# GPU 3 pipeline: wait for manifest.json -> controls_2x2 part 2 (also computes truth efficiency first!) -> geometry ablation
(
  echo "[$(date)] [A:GPU3] Waiting for manifest.json..."
  while [ ! -f "${MANIFEST}" ]; do sleep 4; done
  echo "[$(date)] [A:GPU3] Starting controls_2x2 part 2 (including truth tracking efficiency)..."
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_controls_2x2_and_masking.py" \
    --models vqvae_encoder_d128 vqvae_code_d8 vqvae_quant_d8 raw_summary_d28 pca_summary_d8 phys_handcrafted_d6 mult_only_d1 random_init_d64 \
    --gpu 0 > "${LOGDIR}/controls_2x2_part2.log" 2>&1
  echo "[$(date)] [A:GPU3] controls_2x2 part 2 DONE! Starting geometry ZCA ablation..."
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step geometry --gpu 0 > "${LOGDIR}/nplm_geometry.log" 2>&1
  echo "[$(date)] [A:GPU3] ALL DONE!"
) &
P3=$!

wait ${P0} ${P1} ${P2} ${P3}
echo "[$(date)] Debug Job A COMPLETED ALL WORKERS!"
