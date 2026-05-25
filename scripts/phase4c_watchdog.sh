#!/usr/bin/env bash
# Run C watchdog — same pattern as phase4b_watchdog.sh but for the low-quality
# unit and 30000-step target.

set -uo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_ID="$(python3 -c "import json,sys;print(json.load(open(\"${RUN}/run_config.json\"))[\"teacher\"][\"voice_id\"])")"
OUT="${RUN}/output/${VOICE_ID}"
VOICE_LOWER="$(echo "${VOICE_ID}" | tr '[:upper:]' '[:lower:]')"
LOG="${OUT}/logs/watchdog_c.log"
UNIT="piper-train-${VOICE_LOWER}-low"
CHECKPOINTS="${OUT}/phase4-low/checkpoints"
TARGET="${CHECKPOINTS}/*step=30000*.ckpt"

mkdir -p "${OUT}/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "${LOG}"; }
log "watchdog-c starting (PID $$)"

RESTART_COUNT=0
MAX_RESTARTS=10
QUIET_TICKS=0

while true; do
  if ls ${TARGET} 1>/dev/null 2>&1; then
    log "30000-step checkpoint detected; exiting (restarts=${RESTART_COUNT})"
    exit 0
  fi
  STATE=$(systemctl --user is-active "${UNIT}" 2>/dev/null || echo "unknown")
  if [ "${STATE}" = "active" ] || [ "${STATE}" = "activating" ]; then
    QUIET_TICKS=0
    sleep 60
    continue
  fi
  QUIET_TICKS=$((QUIET_TICKS + 1))
  log "unit not active (state=${STATE}, quiet_ticks=${QUIET_TICKS})"
  if [ "${QUIET_TICKS}" -lt 2 ]; then sleep 30; continue; fi
  if [ "${RESTART_COUNT}" -ge "${MAX_RESTARTS}" ]; then
    log "RESTART LIMIT (${MAX_RESTARTS}) hit; giving up"
    echo "$(date): Run C watchdog gave up after ${MAX_RESTARTS} restarts" \
      >> "${OUT}/state/notifications.txt"
    exit 2
  fi
  RESTART_COUNT=$((RESTART_COUNT + 1))
  log "RESTART #${RESTART_COUNT}"
  echo "$(date): Run C watchdog: restart #${RESTART_COUNT} (state=${STATE})" \
    >> "${OUT}/state/notifications.txt"
  bash "${RUN}/scripts/phase4c_relaunch.sh" >> "${LOG}" 2>&1
  QUIET_TICKS=0
  sleep 60
done
