#!/bin/bash
#SBATCH --account=m4958_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --job-name=fm_p2_w3A
#SBATCH --output=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/w3A_%j.out
#SBATCH --error=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/w3A_%j.err
#SBATCH --qos=debug
#SBATCH --time=00:29:30

set -euo pipefail

PY="/global/homes/d/danieltm/.conda/envs/collider-env/bin/python"
SUITE="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite"
RES="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results"
LOGDIR="${RES}/logs_v2"
TOK64_NPZ="${RES}/embeddings_v2/ssl_tokens_d64.npz"

echo "[$(date)] Starting Wave 3A on $(hostname) (Job ${SLURM_JOB_ID})"
nvidia-smi -L

# GPU 0 (~20m): Finish final ~2 epochs of ssl_tokens_d64 -> extract -> probes + all_shape + all_additive + highstats on ssl_tokens_d64
(
  if [ ! -f "${TOK64_NPZ}" ]; then
    echo "[$(date)] [W3A:GPU0] Finishing final epochs of train_ssl_tokens_d64..."
    CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_prepare_and_extract.py" --step train_ssl_tokens_d64 --epochs 12 --gpu 0 >> "${LOGDIR}/train_ssl_tokens_d64.log" 2>&1
  fi
  echo "[$(date)] [W3A:GPU0] Running probes + shape + additive + highstats for ssl_tokens_d64..."
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step probes --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/w3_probes_tok64.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_shape --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/w3_shape_tok64.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_additive --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/w3_add_tok64.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/w3_highstats_tok64.log" 2>&1
  echo "[$(date)] [W3A:GPU0] ALL DONE!"
) &
P0=$!

# GPU 1 (~17m): Finalize paired probes CSV (reuses cached *_probe_detail.json!) -> all_shape part 1 (9 models + instant merge of part 2!)
(
  echo "[$(date)] [W3A:GPU1] Finalizing paired probes + curve_models..."
  CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step probes --gpu 0 > "${LOGDIR}/w3_probes_all.log" 2>&1
  echo "[$(date)] [W3A:GPU1] Running all_shape part 1a + cached part 2..."
  CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_shape \
    --models supcon_d64 supcon_d32 supcon_d8 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_d32 ssl_kinematic_d8 ssl_kinematic_drop_d64 ssl_tokens_d32 ae_transformer_d64 ae_transformer_d32 ae_transformer_d8 ae_transformer_postln_d64 ae_transformer_postln_d32 ae_transformer_postln_d8 ae_mlp_d64 ae_mlp_d32 ae_mlp_d8 vqvae_encoder_d128 vqvae_code_d8 vqvae_quant_d8 random_init_d64 random_init_d32 random_init_d8 \
    --gpu 0 > "${LOGDIR}/w3_all_shape_1a.log" 2>&1
  echo "[$(date)] [W3A:GPU1] ALL DONE!"
) &
P1=$!

# GPU 2 (~19m): Finish last 2 seeds of supervised_scratch_d64 (~7m) -> controls_2x2 part 1 (~10m) -> ssl_tokens_d64 controls (~1.5m)
(
  echo "[$(date)] [W3A:GPU2] Finishing remaining seeds of supervised_scratch_d64..."
  CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step scratch_curves --gpu 0 > "${LOGDIR}/w3_scratch_curves.log" 2>&1
  echo "[$(date)] [W3A:GPU2] Running controls_2x2 part 1..."
  CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_controls_2x2_and_masking.py" \
    --models supcon_d64 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_d32 ssl_tokens_d32 ssl_tokens_d8 ae_transformer_d32 ae_mlp_d32 \
    --gpu 0 > "${LOGDIR}/w3_controls_part1.log" 2>&1
  while [ ! -f "${TOK64_NPZ}" ]; do sleep 5; done
  sleep 2
  CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_controls_2x2_and_masking.py" --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/w3_controls_tok64.log" 2>&1
  echo "[$(date)] [W3A:GPU2] ALL DONE!"
) &
P2=$!

# GPU 3 (~21m): controls_2x2 part 2 (including truth tracking efficiency, ~12m) -> geometry part 1 (~9m)
(
  echo "[$(date)] [W3A:GPU3] Running controls_2x2 part 2 (with gpfs_lock)..."
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_controls_2x2_and_masking.py" \
    --models vqvae_encoder_d128 vqvae_code_d8 vqvae_quant_d8 raw_summary_d28 pca_summary_d8 phys_handcrafted_d6 mult_only_d1 random_init_d64 \
    --gpu 0 > "${LOGDIR}/w3_controls_part2.log" 2>&1
  echo "[$(date)] [W3A:GPU3] Running geometry ZCA ablation part 1..."
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step geometry \
    --models supcon_d64 supervised_e2e_d64 ssl_kinematic_d64 ae_mlp_d32 \
    --gpu 0 > "${LOGDIR}/w3_geometry_1.log" 2>&1
  while [ ! -f "${TOK64_NPZ}" ]; do sleep 5; done
  sleep 2
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step geometry --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/w3_geometry_tok64.log" 2>&1
  echo "[$(date)] [W3A:GPU3] ALL DONE!"
) &
P3=$!

wait ${P0} ${P1} ${P2} ${P3}
echo "[$(date)] Wave 3A COMPLETED 100% OF ALL TASKS!"
