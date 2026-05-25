#!/usr/bin/env bash
# Run B watchdog — keeps the piper-train-takashii-full systemd unit running
# until the 40000-step checkpoint exists. Resumes from latest ckpt on each
# restart so progress is not lost.

set -uo pipefail  # no -e: we want to keep going on transient errors

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_ID="$(python3 -c "import json,sys;print(json.load(open(\"${RUN}/run_config.json\"))[\"teacher\"][\"voice_id\"])")"
OUT="${RUN}/output/${VOICE_ID}"
LOG="${OUT}/logs/watchdog.log"
UNIT="piper-train-takashii-full"
CHECKPOINTS="${OUT}/phase4-full/checkpoints"
TARGET="${CHECKPOINTS}/*step=40000*.ckpt"

mkdir -p "${OUT}/logs"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "${LOG}"
}

log "watchdog starting (PID $$)"

RESTART_COUNT=0
MAX_RESTARTS=10
QUIET_TICKS=0   # consecutive ticks where unit was inactive (might be normal terminal state)

while true; do
  # Done condition: target checkpoint exists
  if ls ${TARGET} 1>/dev/null 2>&1; then
    log "40000-step checkpoint detected; exiting (restarts=${RESTART_COUNT})"
    # Spawn orchestrator so the A→B→C finalization chain runs without needing
    # an attended relaunch. Orchestrator's skip cascade lands on Phase 5A.
    log "spawning orchestrator to chain Phase 5A → 6A → 5B → 6B → C → finalize"
    cd "${RUN}"
    PYTHONUNBUFFERED=1 nohup .venv/bin/python3 -u scripts/orchestrate.py \
      >> "${OUT}/logs/orchestrate.log" 2>&1 < /dev/null &
    ORCH_PID=$!
    disown ${ORCH_PID} 2>/dev/null || true
    echo "${ORCH_PID}" > "${OUT}/state/orchestrate.pid"
    log "orchestrator spawned (PID ${ORCH_PID})"
    exit 0
  fi

  STATE=$(systemctl --user is-active "${UNIT}" 2>/dev/null || echo "unknown")
  if [ "${STATE}" = "active" ] || [ "${STATE}" = "activating" ]; then
    QUIET_TICKS=0
    sleep 60
    continue
  fi

  # Unit not active. Either it died early (restart) or it exited after 40k
  # (covered above). Could also be "inactive" because it cleanly finished
  # before max_steps somehow; require two consecutive non-active ticks before
  # declaring crash, to avoid racing with a clean exit log update.
  QUIET_TICKS=$((QUIET_TICKS + 1))
  log "unit not active (state=${STATE}, quiet_ticks=${QUIET_TICKS})"
  if [ "${QUIET_TICKS}" -lt 2 ]; then
    sleep 30
    continue
  fi

  # Restart needed
  if [ "${RESTART_COUNT}" -ge "${MAX_RESTARTS}" ]; then
    log "RESTART LIMIT (${MAX_RESTARTS}) hit; giving up"
    echo "$(date): Run B watchdog gave up after ${MAX_RESTARTS} restarts" \
      >> "${OUT}/state/notifications.txt"
    exit 2
  fi
  RESTART_COUNT=$((RESTART_COUNT + 1))
  log "RESTART #${RESTART_COUNT}"
  echo "$(date): Run B watchdog: restart #${RESTART_COUNT} (state=${STATE})" \
    >> "${OUT}/state/notifications.txt"
  bash "${RUN}/scripts/phase4b_relaunch.sh" >> "${LOG}" 2>&1
  QUIET_TICKS=0
  sleep 60  # let the new unit settle before next health check
done
