#!/bin/bash
set -euo pipefail

PYTHON=/global/homes/d/danieltm/.conda/envs/collider-env/bin/python
SCRIPT=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite/train_and_extract.py
LOGDIR=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/logs
mkdir -p "${LOGDIR}"

echo "[$(date)] Starting Job 1 on $(hostname)"

CUDA_VISIBLE_DEVICES=0 ${PYTHON} ${SCRIPT} --task train_ssl_kinematic --dim 8 --epochs 12 --gpu 0 > "${LOGDIR}/ssl_kinematic_d8.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 ${PYTHON} ${SCRIPT} --task train_ssl_kinematic --dim 32 --epochs 12 --gpu 0 > "${LOGDIR}/ssl_kinematic_d32.log" 2>&1 &
PID1=$!

wait ${PID0}
CUDA_VISIBLE_DEVICES=0 ${PYTHON} ${SCRIPT} --task train_ssl_kinematic --dim 64 --epochs 12 --gpu 0 > "${LOGDIR}/ssl_kinematic_d64.log" 2>&1 &
PID2=$!

wait ${PID1}
CUDA_VISIBLE_DEVICES=1 ${PYTHON} ${SCRIPT} --task train_ssl_kinematic_drop --dim 64 --epochs 12 --gpu 0 > "${LOGDIR}/ssl_kinematic_drop_d64.log" 2>&1 &
PID3=$!

wait ${PID2} ${PID3}
echo "[$(date)] Completed pretraining on $(hostname), starting probes and NPLM..."

${PYTHON} /global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite/evaluate_probes.py --models ssl_kinematic_d64 ssl_kinematic_drop_d64 > "${LOGDIR}/probes_ssl64.log" 2>&1 &

NPLM=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/Foundation-Models/unified_suite/nplm_runner.py

CUDA_VISIBLE_DEVICES=0 ${PYTHON} ${NPLM} --models ssl_kinematic_d64 ssl_kinematic_drop_d64 supcon_d32 ae_transformer_d32 ae_transformer_postln_d32 ae_mlp_d32 ssl_legacy_d32 random_init_d32 --signals 3 4 5 --sample-mode shape --gpu 0 > "${LOGDIR}/nplm_gpu0_nid200324.log" 2>&1 &
NPID0=$!

CUDA_VISIBLE_DEVICES=1 ${PYTHON} ${NPLM} --models supcon_d8 ae_transformer_d8 ae_transformer_postln_d8 ae_mlp_d8 ssl_legacy_d8 random_init_d8 vqvae_code_d8 --signals 3 4 5 --sample-mode shape --gpu 0 > "${LOGDIR}/nplm_gpu1_nid200324.log" 2>&1 &
NPID1=$!

wait ${NPID1}
CUDA_VISIBLE_DEVICES=1 ${PYTHON} ${NPLM} --models supcon_d64 ssl_kinematic_d64 ssl_tokens_d64 ae_transformer_d64 ae_mlp_d64 random_init_d64 --signals 3 4 --sample-mode additive --gpu 0 > "${LOGDIR}/nplm_additive_nid200324.log" 2>&1 &
NPID2=$!

wait ${NPID0} ${NPID2}
echo "[$(date)] Completed Job 1 on $(hostname)"
