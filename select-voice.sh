#!/usr/bin/env bash
# select-voice.sh <name> — activate a versioned per-voice config.
#
# Copies configs/<name>.json -> run_config.json (the active config every phase
# script reads). This is the robust multi-voice mechanism: run state is keyed
# by teacher.voice_id, so switching voices never collides, and the choice is
# persisted on disk (survives shell/agent restarts) rather than relying on an
# env var threaded through nohup -> orchestrate.py -> systemd-run.
#
# Scripts ALSO honor PIPER_DISTILL_CONFIG=<path> as an override for ad-hoc
# single-phase runs, but the orchestrated end-to-end run uses run_config.json.
#
# Usage:
#   ./select-voice.sh computer      # activate configs/computer.json
#   ./select-voice.sh adjutant
#   ./select-voice.sh               # show current active voice
set -euo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
  active="$(python3 -c "import json;c=json.load(open('${RUN}/run_config.json'));print(c['teacher']['voice_id'], '->', c['voice_name'])" 2>/dev/null || echo '(unreadable)')"
  echo "active run_config.json: ${active}"
  echo "available configs:"
  for f in "${RUN}"/configs/*.json; do echo "  - $(basename "${f%.json}")"; done
  exit 0
fi

name="$1"
src="${RUN}/configs/${name}.json"
[ -f "${src}" ] || { echo "no such config: ${src}" >&2; exit 1; }

cp "${src}" "${RUN}/run_config.json"
python3 -c "import json;c=json.load(open('${RUN}/run_config.json'));print('activated', c['teacher']['voice_id'], '->', c['voice_name'], 'runs='+str(c.get('runs')))"
