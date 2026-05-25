#!/usr/bin/env bash
# One-shot: resume Run C training from latest ckpt, wait for exact step=30000,
# then run Phase 5C + Phase 6C, restart piper.service, restart vllm-aeon.
#
# Designed to run detached (nohup). Writes progress to logs/finish_c.log.

set -uo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${RUN}/.venv"
UNIT="piper-train-takashii-low"
CKPT_DIR="${RUN}/phase4-low/checkpoints"
LOG="${RUN}/logs/finish_c.log"

mkdir -p "${RUN}/logs"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "${LOG}"; }
log "finish-C starting (PID $$)"

# 1. Launch (resume) training
log "step 1/6: launching training via phase4c_relaunch.sh"
bash "${RUN}/scripts/phase4c_relaunch.sh" >> "${LOG}" 2>&1
if [ $? -ne 0 ]; then
  log "FAILED: relaunch returned non-zero"
  echo "$(date): Run C finish: relaunch failed" >> "${RUN}/state/notifications.txt"
  exit 1
fi

# 2. Wait for exact step=30000.ckpt to appear
log "step 2/6: waiting for ${CKPT_DIR}/*step=30000.ckpt"
RESTART_COUNT=0
MAX_RESTARTS=5
QUIET_TICKS=0
while true; do
  # Exact-step match: file ending in step=30000.ckpt
  if ls "${CKPT_DIR}"/*step=30000.ckpt 2>/dev/null | head -1 | grep -q .; then
    log "step=30000.ckpt detected"
    break
  fi
  STATE=$(systemctl --user is-active "${UNIT}" 2>/dev/null || echo unknown)
  if [ "${STATE}" = "active" ] || [ "${STATE}" = "activating" ]; then
    QUIET_TICKS=0; sleep 60; continue
  fi
  QUIET_TICKS=$((QUIET_TICKS + 1))
  log "unit not active (state=${STATE}, quiet=${QUIET_TICKS})"
  if [ "${QUIET_TICKS}" -lt 2 ]; then sleep 30; continue; fi
  if [ "${RESTART_COUNT}" -ge "${MAX_RESTARTS}" ]; then
    log "RESTART LIMIT hit; giving up"
    echo "$(date): Run C finish: restart limit hit" >> "${RUN}/state/notifications.txt"
    exit 2
  fi
  RESTART_COUNT=$((RESTART_COUNT + 1))
  log "RESTART #${RESTART_COUNT}"
  bash "${RUN}/scripts/phase4c_relaunch.sh" >> "${LOG}" 2>&1
  QUIET_TICKS=0; sleep 60
done

# 3. Stop the unit cleanly (may already be inactive after max_steps)
log "step 3/6: stopping unit"
systemctl --user stop "${UNIT}" 2>/dev/null || true
sleep 5

# 4. Run Phase 5C eval on last 3 checkpoints
log "step 4/6: Phase 5C eval"
rm -rf "${RUN}/state/phase5-low/"*
CKPTS=$(ls -t "${CKPT_DIR}"/*.ckpt | head -3 | tac)
EVAL_ARGS=""
for c in ${CKPTS}; do EVAL_ARGS="${EVAL_ARGS} --checkpoint ${c}"; done
log "ckpts: $(echo ${CKPTS} | xargs -n1 basename | tr '\n' ' ')"
cd "${RUN}"
PIPER_DISTILL_EVAL_VARIANT=low PYTHONUNBUFFERED=1 \
  "${VENV}/bin/python3" scripts/phase5_eval.py ${EVAL_ARGS} >> "${LOG}" 2>&1
if [ $? -ne 0 ]; then
  log "Phase 5C eval failed"
  echo "$(date): Run C finish: Phase 5C eval failed" >> "${RUN}/state/notifications.txt"
  exit 3
fi

# 5. Run Phase 6C install on Phase 5C winner
log "step 5/6: Phase 6C install"
WINNER=$("${VENV}/bin/python3" -c "
import json
m = json.load(open('${RUN}/state/phase5-low/manifest.json'))
print(m['winner'])
")
METRICS="${RUN}/state/phase5-low/$(basename ${WINNER} .ckpt)/metrics.json"
log "winner: $(basename ${WINNER})"
"${VENV}/bin/python3" "${RUN}/scripts/phase6_install.py" \
  --checkpoint "${WINNER}" \
  --voice-name "en_US-takashii-low" \
  --training-config "${RUN}/phase4-low/config.json" \
  --dataset-label "takashii_full_11713_16khz_30k" \
  --metrics-json "${METRICS}" \
  --no-restart-service >> "${LOG}" 2>&1
if [ $? -ne 0 ]; then
  log "Phase 6C install failed"
  exit 4
fi

# 6. Restart services
log "step 6/6: restart piper.service, restart vllm-aeon"
sudo -n systemctl restart piper
sleep 3
sudo -n systemctl start vllm-aeon
sleep 5
sudo -n systemctl is-active vllm-aeon | head -1 >> "${LOG}"

log "FINISH-C COMPLETE"
echo "$(date): Run C properly finished at step 30000; all services restored" \
  >> "${RUN}/state/notifications.txt"
