#!/usr/bin/env bash
# Phase 4 — fine-tune Piper from the Lessac checkpoint.
#
# Pre: vllm-aeon stopped, Phase 3 dataset under phase3/audio/ + phase3/metadata.csv,
# Phase 4 split under phase4/train_metadata.csv + phase4/eval_metadata.csv.
#
# Launches as a transient user systemd unit so it survives shell exits.
# Monitor: journalctl --user -u piper-train-takashii -f
# Stop:    systemctl --user stop piper-train-takashii

set -euo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${RUN}/.venv"
CKPT="${RUN}/checkpoints/en_US-lessac-medium-clean.ckpt"
TRAIN_META="${RUN}/phase4/train_metadata.csv"
AUDIO_DIR="${RUN}/phase3/audio"
CACHE_DIR="${RUN}/phase4/cache"
CFG_PATH="${RUN}/phase4/config.json"
UNIT="piper-train-takashii"

# Pre-flight: vllm-aeon must be stopped (GPU 1 free)
USED=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$USED" -gt 2000 ]; then
  echo "GPU 1 has ${USED} MiB used — vllm-aeon likely still running. Stop it first:"
  echo "  sudo systemctl stop vllm-aeon"
  exit 1
fi
echo "GPU 1 free (used=${USED} MiB)"

# Probe batch size starting at 32; back off on OOM via the unit's restart policy.
BATCH=${PIPER_BATCH:-32}

mkdir -p "${CACHE_DIR}" "${RUN}/phase4/logs"

systemd-run --user \
  --unit="${UNIT}" \
  --description="Piper VITS fine-tune (Takashii voice clone)" \
  --setenv=CUDA_VISIBLE_DEVICES=1 \
  --setenv=PYTHONUNBUFFERED=1 \
  --working-directory="${RUN}/piper1-gpl" \
  -- \
  "${VENV}/bin/python3" -m piper.train fit \
    --data.voice_name "en_US-takashii-medium" \
    --data.csv_path "${TRAIN_META}" \
    --data.audio_dir "${AUDIO_DIR}" \
    --model.sample_rate 22050 \
    --data.espeak_voice "en-us" \
    --data.cache_dir "${CACHE_DIR}" \
    --data.config_path "${CFG_PATH}" \
    --data.batch_size "${BATCH}" \
    --trainer.devices "[0]" \
    --trainer.max_steps 40000 \
    --trainer.val_check_interval 0.5 \
    --trainer.check_val_every_n_epoch 1 \
    --trainer.callbacks+=ModelCheckpoint \
    --trainer.callbacks.dirpath="${RUN}/phase4/checkpoints" \
    --trainer.callbacks.save_top_k=-1 \
    --trainer.callbacks.every_n_train_steps=1000 \
    --ckpt_path "${CKPT}"

sleep 2
systemctl --user status "${UNIT}" --no-pager -l | head -25 || true
echo
echo "Tail logs: journalctl --user -u ${UNIT} -f"
