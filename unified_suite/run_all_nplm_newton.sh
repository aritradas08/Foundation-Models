#!/bin/bash
set -euo pipefail

NPLM=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite/nplm_runner.py
PY=/global/homes/d/danieltm/.conda/envs/collider-env/bin/python
LOGDIR=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs
mkdir -p "${LOGDIR}"

echo "[$(date)] Starting Newton NPLM suite across 2 GPUs on $(hostname)..."

CUDA_VISIBLE_DEVICES=0 ${PY} ${NPLM} \
  --models supcon_d64 supervised_e2e_d64 ssl_kinematic_d64 ssl_kinematic_drop_d64 ssl_tokens_d64 ssl_legacy_d64 ae_transformer_d64 ae_transformer_postln_d64 ae_mlp_d64 random_init_d64 vqvae_encoder_d128 supcon_d32 ssl_kinematic_d32 ssl_legacy_d32 \
  --signals 3 4 5 --sample-mode shape --gpu 0 > "${LOGDIR}/nplm_newton_gpu0.log" 2>&1 &
P0=$!

(
  CUDA_VISIBLE_DEVICES=1 ${PY} ${NPLM} \
    --models ae_transformer_d32 ae_transformer_postln_d32 ae_mlp_d32 random_init_d32 supcon_d8 ssl_kinematic_d8 ssl_tokens_d8 ssl_legacy_d8 ae_transformer_d8 ae_transformer_postln_d8 ae_mlp_d8 random_init_d8 vqvae_code_d8 \
    --signals 3 4 5 --sample-mode shape --gpu 0 > "${LOGDIR}/nplm_newton_gpu1.log" 2>&1 && \
  CUDA_VISIBLE_DEVICES=1 ${PY} ${NPLM} \
    --models supcon_d64 ssl_kinematic_d64 ssl_kinematic_drop_d64 ssl_tokens_d64 ae_transformer_d64 ae_mlp_d64 random_init_d64 \
    --signals 3 4 --sample-mode additive --gpu 0 > "${LOGDIR}/nplm_newton_additive.log" 2>&1
) &
P1=$!

wait ${P0} ${P1}
echo "[$(date)] Completed Newton NPLM suite on $(hostname)!"
