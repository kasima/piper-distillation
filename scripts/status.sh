#!/usr/bin/env bash
# Quick status dump for the autonomous Takashii distillation run.

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_ID="$(python3 -c "import json,sys;print(json.load(open(\"${RUN}/run_config.json\"))[\"teacher\"][\"voice_id\"])")"
OUT="${RUN}/output/${VOICE_ID}"
VOICE_LOWER="$(echo "${VOICE_ID}" | tr '[:upper:]' '[:lower:]')"

cat() { /bin/cat "$@" 2>/dev/null; }
echo "===== orchestrate_status ====="
cat "${OUT}/state/orchestrate_status.json"
echo
echo "===== phase 2 progress ====="
if [ -f "${OUT}/state/phase2/done.txt" ]; then
  N=$(wc -l < "${OUT}/state/phase2/done.txt")
  PCT=$(/usr/bin/python3 -c "print(f'{100*${N}/12000:.1f}')")
  echo "done=${N}/12000 (${PCT}%)"
fi
PID=$(cat "${OUT}/state/phase2/pid")
if [ -n "$PID" ] && /bin/kill -0 "$PID" 2>/dev/null; then
  ETIME=$(/bin/ps -p "$PID" -o etime= | tr -d ' ')
  echo "phase2 PID ${PID} alive, elapsed ${ETIME}"
else
  echo "phase2 process not running"
fi
echo "last 3 synth log entries:"
tail -3 "${OUT}/state/phase2/synthesis_log.jsonl"
echo
echo "===== orchestrator ====="
ORCH_PID=$(cat "${OUT}/state/orchestrate.pid")
if [ -n "$ORCH_PID" ] && /bin/kill -0 "$ORCH_PID" 2>/dev/null; then
  echo "orchestrator PID ${ORCH_PID} alive"
else
  echo "orchestrator NOT running"
fi
echo "orchestrator log (last 10):"
tail -10 "${OUT}/logs/orchestrate.log"
echo
echo "===== notifications ====="
tail -20 "${OUT}/state/notifications.txt"
echo
echo "===== GPUs ====="
/usr/bin/nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
