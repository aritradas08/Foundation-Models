#!/bin/bash
#SBATCH --account=m4958_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --job-name=fm_p2_dbgB
#SBATCH --output=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/dbgB_%j.out
#SBATCH --error=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2/dbgB_%j.err
#SBATCH --qos=debug
#SBATCH --time=00:29:30

set -euo pipefail

PY="/global/homes/d/danieltm/.conda/envs/collider-env/bin/python"
SUITE="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite"
LOGDIR="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs_v2"
MANIFEST="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/embeddings_v2/manifest.json"
TOK64_NPZ="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/embeddings_v2/ssl_tokens_d64.npz"
mkdir -p "${LOGDIR}"

echo "[$(date)] Starting Debug Job B on $(hostname) (Job ${SLURM_JOB_ID}) - waiting for ${MANIFEST}..."
nvidia-smi -L

while [ ! -f "${MANIFEST}" ]; do
  sleep 5
done
sleep 2
echo "[$(date)] Manifest detected! Launching 4 parallel highstats 10k-null workers..."

CUDA_VISIBLE_DEVICES=0 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats \
  --models supcon_d64 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_d32 \
  --gpu 0 > "${LOGDIR}/highstats_B0.log" 2>&1 &
B0=$!

CUDA_VISIBLE_DEVICES=1 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats \
  --models ssl_tokens_d32 ssl_tokens_d8 ae_transformer_d32 ae_mlp_d32 \
  --gpu 0 > "${LOGDIR}/highstats_B1.log" 2>&1 &
B1=$!

CUDA_VISIBLE_DEVICES=2 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats \
  --models vqvae_code_d8 vqvae_quant_d8 raw_summary_d28 pca_summary_d8 \
  --gpu 0 > "${LOGDIR}/highstats_B2.log" 2>&1 &
B2=$!

(
  CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step highstats \
    --models phys_handcrafted_d6 mult_only_d1 random_init_d64 \
    --gpu 0 > "${LOGDIR}/highstats_B3.log" 2>&1
  echo "[$(date)] [B:GPU3] highstats_B3 DONE! Waiting up to 10 min for ssl_tokens_d64.npz to run 2x2/masking on ssl_tokens_d64..."
  for _ in $(seq 1 120); do
    if [ -f "${TOK64_NPZ}" ]; then
      sleep 3
      echo "[$(date)] [B:GPU3] ssl_tokens_d64.npz detected! Running controls_2x2 + geometry for ssl_tokens_d64..."
      CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_controls_2x2_and_masking.py" \
        --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/controls_2x2_tok64.log" 2>&1
      CUDA_VISIBLE_DEVICES=3 ${PY} "${SUITE}/phase2_nplm_calibrated.py" --step geometry \
        --models ssl_tokens_d64 --gpu 0 > "${LOGDIR}/nplm_geometry_tok64.log" 2>&1
      break
    fi
    sleep 5
  done
) &
B3=$!

wait ${B0} ${B1} ${B2} ${B3}
echo "[$(date)] Debug Job B COMPLETED ALL WORKERS!"
