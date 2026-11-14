#!/bin/bash
#SBATCH --account=m4958_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --job-name=fm_phase2_all4
#SBATCH --output=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/phase2_%j.out
#SBATCH --error=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/phase2_%j.err
#SBATCH --qos=preempt
#SBATCH --time=02:00:00

set -euo pipefail

PY="/global/homes/d/danieltm/.conda/envs/collider-env/bin/python"
SUITE="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite"
LOGDIR="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2"
mkdir -p "${LOGDIR}"

echo "[$(date)] Starting Phase 2 All-4-Workstreams GPU Suite on $(hostname) (Job ${SLURM_JOB_ID})"
nvidia-smi -L

# ---------------------------------------------------------------------------
# WAVE 1:
#   GPU 0: Build canonical eval_downstream_1050k.parquet + extract all 32 ready models + VQ tail fidelity
#   Once canonical parquet + cache exist (~90s), launch in parallel:
#   GPU 1: Train unified ssl_tokens_d64 (12 epochs) + extract ssl_tokens_d64.npz
#   GPU 2: Train supervised_scratch_d64 low-label curves (N_labeled in {100,1k,10k,100k} x 5 seeds)
# ---------------------------------------------------------------------------
echo "[$(date)] Launching GPU 0: extract_all..."
CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_prepare_and_extract.py" --step extract_all --gpu 0 > "${LOGDIR}/extract_all.log" 2>&1 &
PID_EXT=$!

# Wait until canonical eval_downstream_1050k.parquet AND its /pscratch npz cache exist
while ! ls /pscratch/sd/d/danieltm/fm_cache/eval_downstream_1050k_mt256_lblall_*.npz >/dev/null 2>&1; do
  sleep 5
done
sleep 5

echo "[$(date)] Launching GPU 1: train_ssl_tokens_d64 & GPU 2: scratch_curves..."
CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_prepare_and_extract.py" --step train_ssl_tokens_d64 --epochs 12 --gpu 0 > "${LOGDIR}/train_ssl_tokens_d64.log" 2>&1 &
PID_TOK64=$!

CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step scratch_curves --gpu 0 > "${LOGDIR}/scratch_curves.log" 2>&1 &
PID_SCRATCH=$!

# Wait for GPU 0 (extract_all) to finish (~7-9 min)
wait ${PID_EXT}
echo "[$(date)] GPU 0 extract_all finished!"

# ---------------------------------------------------------------------------
# WAVE 2 (while GPU 1 finishes ssl_tokens_d64):
#   GPU 0: Paired probes on all 32 extracted models -> then all_shape NPLM part 1 + all_additive + geometry
#   GPU 3: High-stats 10k-null NPLM calibration part 1 (8 models)
#   GPU 2 (as soon as scratch_curves finishes): 2x2 Factorial + (d0,z0)=0 + MultMatched controls
# ---------------------------------------------------------------------------
(
  echo "[$(date)] GPU 0 starting paired probes (32 models)..."
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step probes --gpu 0 > "${LOGDIR}/probes_32.log" 2>&1
  echo "[$(date)] GPU 0 starting all_shape NPLM (first half)..."
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_shape \
    --models supcon_d64 supcon_d32 supcon_d8 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_d32 ssl_kinematic_d8 ssl_kinematic_drop_d64 ssl_tokens_d32 ssl_tokens_d8 ssl_tokens_legacy_d64 ssl_legacy_d64 ssl_legacy_d32 ssl_legacy_d8 raw_summary_d28 pca_summary_d8 phys_handcrafted_d6 mult_only_d1 \
    --gpu 0 > "${LOGDIR}/nplm_all_shape_part1.log" 2>&1
  echo "[$(date)] GPU 0 starting all_additive NPLM..."
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_additive \
    --models supcon_d64 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_drop_d64 ae_transformer_d64 ae_mlp_d64 random_init_d64 raw_summary_d28 phys_handcrafted_d6 \
    --gpu 0 > "${LOGDIR}/nplm_all_additive.log" 2>&1
) &
PID_GPU0_W2=$!

(
  echo "[$(date)] GPU 3 starting all_shape NPLM (second half) + highstats part 1..."
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_shape \
    --models ae_transformer_d64 ae_transformer_d32 ae_transformer_d8 ae_transformer_postln_d64 ae_transformer_postln_d32 ae_transformer_postln_d8 ae_mlp_d64 ae_mlp_d32 ae_mlp_d8 vqvae_encoder_d128 vqvae_code_d8 vqvae_quant_d8 random_init_d64 random_init_d32 random_init_d8 \
    --gpu 0 > "${LOGDIR}/nplm_all_shape_part2.log" 2>&1
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats \
    --models supcon_d64 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_d32 ssl_tokens_d32 ssl_tokens_d8 ae_transformer_d32 ae_mlp_d32 \
    --gpu 0 > "${LOGDIR}/nplm_highstats_part1.log" 2>&1
) &
PID_GPU3_W2=$!

(
  wait ${PID_SCRATCH}
  echo "[$(date)] GPU 2 scratch_curves finished! Starting highstats part 2..."
  CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats \
    --models vqvae_code_d8 vqvae_quant_d8 raw_summary_d28 pca_summary_d8 phys_handcrafted_d6 mult_only_d1 random_init_d64 \
    --gpu 0 > "${LOGDIR}/nplm_highstats_part2.log" 2>&1
) &
PID_GPU2_W2=$!

# ---------------------------------------------------------------------------
# WAVE 3: Once unified ssl_tokens_d64 finishes on GPU 1, evaluate ssl_tokens_d64
# across probes, all_shape, all_additive, highstats, geometry, and 2x2 + masking controls!
# ---------------------------------------------------------------------------
wait ${PID_TOK64}
echo "[$(date)] GPU 1 unified ssl_tokens_d64 finished! Running final ssl_tokens_d64 evaluations + 2x2/masking + geometry..."

wait ${PID_GPU0_W2}
CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_probes_and_scratch.py" --step probes --models ssl_tokens_d64 --gpu 0 >> "${LOGDIR}/probes_32.log" 2>&1 &
P_PR64=$!

CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_controls_2x2_and_masking.py" --gpu 0 > "${LOGDIR}/controls_2x2_and_masking.log" 2>&1 &
P_CTRL=$!

wait ${P_PR64}
(
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_shape --models ssl_tokens_d64 --gpu 0 >> "${LOGDIR}/nplm_all_shape_part1.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step all_additive --models ssl_tokens_d64 --gpu 0 >> "${LOGDIR}/nplm_all_additive.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/nplm_highstats_tok64.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step geometry --gpu 0 > "${LOGDIR}/nplm_geometry.log" 2>&1
) &
P_FIN0=$!

wait ${PID_GPU2_W2} ${PID_GPU3_W2} ${P_CTRL} ${P_FIN0}
echo "[$(date)] ALL 4 WORKSTREAMS COMPLETED SUCCESSFULLY!"
