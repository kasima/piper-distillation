#!/usr/bin/env bash
# Resume Run C training to step=40000, redo Phase 5C+6C, restart services.

set -uo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_ID="$(python3 -c "import json,sys;print(json.load(open(\"${RUN}/run_config.json\"))[\"teacher\"][\"voice_id\"])")"
OUT="${RUN}/output/${VOICE_ID}"
VENV="${RUN}/.venv"
UNIT="piper-train-takashii-low"
CKPT_DIR="${OUT}/phase4-low/checkpoints"
LOG="${OUT}/logs/extend_c_40k.log"
TRAIN_META="${OUT}/phase4-low/train_metadata.csv"
AUDIO_DIR="${OUT}/phase3-low/audio"
CACHE_DIR="${OUT}/phase4-low/cache"
CFG_PATH="${OUT}/phase4-low/config.json"

mkdir -p "${OUT}/logs"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "${LOG}"; }
log "extend-C-40k starting (PID $$)"

# 1. Stop vllm-aeon to free GPU 1
log "step 1/7: stop vllm-aeon"
sudo -n systemctl stop vllm-aeon
sleep 5
USED=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$USED" -gt 2000 ]; then
  log "FAILED: GPU 1 has ${USED} MiB used after vllm-aeon stop"
  exit 1
fi
log "GPU 1 free"

# 2. Resume training from step=30000 with max_steps=40000
LATEST=$(ls "${CKPT_DIR}"/*step=30000.ckpt 2>/dev/null | head -1)
if [ -z "${LATEST}" ]; then
  log "FAILED: no step=30000 ckpt found"
  exit 2
fi
log "step 2/7: resuming from $(basename ${LATEST}) with max_steps=40000"

systemctl --user reset-failed "${UNIT}" 2>/dev/null || true
systemctl --user stop "${UNIT}" 2>/dev/null || true

systemd-run --user \
  --unit="${UNIT}" \
  --description="Piper VITS fine-tune (Takashii LOW, extending to 40k)" \
  --setenv=CUDA_VISIBLE_DEVICES=1 \
  --setenv=PYTHONUNBUFFERED=1 \
  --working-directory="${RUN}/piper1-gpl" \
  -- \
  "${VENV}/bin/python3" -m piper.train fit \
    --data.voice_name "en_US-takashii-low" \
    --data.csv_path "${TRAIN_META}" \
    --data.audio_dir "${AUDIO_DIR}" \
    --model.sample_rate 16000 \
    --data.espeak_voice "en-us" \
    --data.cache_dir "${CACHE_DIR}" \
    --data.config_path "${CFG_PATH}" \
    --data.batch_size 32 \
    --trainer.devices "[0]" \
    --trainer.max_steps 40000 \
    --trainer.val_check_interval 0.5 \
    --trainer.check_val_every_n_epoch 1 \
    --trainer.callbacks+=ModelCheckpoint \
    --trainer.callbacks.dirpath="${CKPT_DIR}" \
    --trainer.callbacks.save_top_k=-1 \
    --trainer.callbacks.every_n_train_steps=1000 \
    --ckpt_path "${LATEST}"

# 3. Wait for step=40000.ckpt
log "step 3/7: waiting for step=40000.ckpt"
QUIET_TICKS=0
while true; do
  if ls "${CKPT_DIR}"/*step=40000.ckpt 2>/dev/null | head -1 | grep -q .; then
    log "step=40000.ckpt detected"
    break
  fi
  STATE=$(systemctl --user is-active "${UNIT}" 2>/dev/null || echo unknown)
  if [ "${STATE}" = "active" ] || [ "${STATE}" = "activating" ]; then
    QUIET_TICKS=0; sleep 60; continue
  fi
  QUIET_TICKS=$((QUIET_TICKS + 1))
  log "unit not active (state=${STATE}, quiet=${QUIET_TICKS})"
  if [ "${QUIET_TICKS}" -ge 3 ]; then
    log "unit dead; investigate via journalctl"
    exit 3
  fi
  sleep 30
done

# 4. Stop unit
log "step 4/7: stop unit"
systemctl --user stop "${UNIT}" 2>/dev/null || true
sleep 5

# 5. Phase 5C eval on last 3 ckpts (38k, 39k, 40k)
log "step 5/7: Phase 5C eval"
rm -rf "${OUT}/state/phase5-low/"*
CKPTS=$(ls "${CKPT_DIR}"/*step=38000.ckpt "${CKPT_DIR}"/*step=39000.ckpt "${CKPT_DIR}"/*step=40000.ckpt 2>/dev/null)
EVAL_ARGS=""
for c in ${CKPTS}; do EVAL_ARGS="${EVAL_ARGS} --checkpoint ${c}"; done
log "ckpts: $(echo ${CKPTS} | xargs -n1 basename | tr '\n' ' ')"
cd "${RUN}"
PIPER_DISTILL_EVAL_VARIANT=low PYTHONUNBUFFERED=1 \
  "${VENV}/bin/python3" scripts/phase5_eval.py ${EVAL_ARGS} >> "${LOG}" 2>&1
if [ $? -ne 0 ]; then
  log "Phase 5C eval failed"
  exit 4
fi

# 6. Phase 6C install
log "step 6/7: Phase 6C install"
WINNER=$("${VENV}/bin/python3" -c "
import json
print(json.load(open('${OUT}/state/phase5-low/manifest.json'))['winner'])
")
METRICS="${OUT}/state/phase5-low/$(basename ${WINNER} .ckpt)/metrics.json"
log "winner: $(basename ${WINNER})"
"${VENV}/bin/python3" "${RUN}/scripts/phase6_install.py" \
  --checkpoint "${WINNER}" \
  --voice-name "en_US-takashii-low" \
  --training-config "${OUT}/phase4-low/config.json" \
  --dataset-label "takashii_full_11713_16khz_40k" \
  --metrics-json "${METRICS}" \
  --no-restart-service >> "${LOG}" 2>&1

# 7. Restart services
log "step 7/7: restart piper + vllm-aeon"
sudo -n systemctl restart piper
sleep 3
sudo -n systemctl start vllm-aeon
sleep 5

log "EXTEND-C-40K COMPLETE"
echo "$(date): Run C extended to step=40000 and redeployed" \
  >> "${OUT}/state/notifications.txt"
