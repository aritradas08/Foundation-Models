#!/bin/bash
#SBATCH --account=m4958_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --job-name=fm_p2_w2B
#SBATCH --output=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/w2B_%j.out
#SBATCH --error=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/w2B_%j.err
#SBATCH --qos=debug
#SBATCH --time=00:29:30

set -euo pipefail

PY="/global/homes/d/danieltm/.conda/envs/collider-env/bin/python"
SUITE="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite"
RES="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results"
LOGDIR="${RES}/logs_v2"

echo "[$(date)] Starting Wave 2B (4xA100 High-Stats 10k-Null NPLM Suite) on $(hostname) (Job ${SLURM_JOB_ID})"
nvidia-smi -L

while [ ! -f "${RES}/embeddings_v2/manifest.json" ]; do
  sleep 5
done

# Helper: run only models whose {m}_highstats.npz isn't already saved, or re-merge cleanly
run_missing_highstats() {
  local gpu_id=$1
  local logfile=$2
  shift 2
  local todo=()
  for m in "$@"; do
    if [ ! -f "${RES}/nplm_v2/highstats_calibrated/${m}_highstats.npz" ]; then
      todo+=("${m}")
    fi
  done
  if [ ${#todo[@]} -gt 0 ]; then
    echo "[$(date)] [W2B:GPU${gpu_id}] Running highstats for: ${todo[*]}"
    CUDA_VISIBLE_DEVICES=${gpu_id} ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats --models "${todo[@]}" --gpu 0 > "${logfile}" 2>&1
  else
    echo "[$(date)] [W2B:GPU${gpu_id}] All assigned models already have _highstats.npz; verifying CSV rows..."
    CUDA_VISIBLE_DEVICES=${gpu_id} ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats --models "$@" --gpu 0 > "${logfile}" 2>&1
  fi
}

# Actually: running all 4 models per GPU takes only ~8.5 min out of 29.5 min and guarantees 100% CSV completeness
# (since Wave 1 Job B may be interrupted before post-loop CSV merge if <4 models finished)
CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats \
  --models supcon_d64 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_d32 \
  --gpu 0 > "${LOGDIR}/w2_highstats_B0.log" 2>&1 &
B0=$!

CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats \
  --models ssl_tokens_d32 ssl_tokens_d8 ae_transformer_d32 ae_mlp_d32 \
  --gpu 0 > "${LOGDIR}/w2_highstats_B1.log" 2>&1 &
B1=$!

CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats \
  --models vqvae_code_d8 vqvae_quant_d8 raw_summary_d28 pca_summary_d8 \
  --gpu 0 > "${LOGDIR}/w2_highstats_B2.log" 2>&1 &
B2=$!

CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats \
  --models phys_handcrafted_d6 mult_only_d1 random_init_d64 \
  --gpu 0 > "${LOGDIR}/w2_highstats_B3.log" 2>&1 &
B3=$!

wait ${B0} ${B1} ${B2} ${B3}
echo "[$(date)] Wave 2B COMPLETED ALL 15 HIGHSTATS MODELS!"
