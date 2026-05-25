#!/usr/bin/env bash
# Smart relaunch for Run C — resume from latest ckpt or cold-start from Lessac low.

set -euo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${RUN}/.venv"
TRAIN_META="${RUN}/phase4-low/train_metadata.csv"
AUDIO_DIR="${RUN}/phase3-low/audio"
CACHE_DIR="${RUN}/phase4-low/cache"
CFG_PATH="${RUN}/phase4-low/config.json"
UNIT="piper-train-takashii-low"
LESSAC="${RUN}/checkpoints/en_US-lessac-low-clean.ckpt"

USED=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$USED" -gt 2000 ]; then
  echo "[relaunch-c] GPU 1 has ${USED} MiB used — refusing to launch"
  exit 1
fi

LATEST=$(ls -t "${RUN}/phase4-low/checkpoints"/*step=*.ckpt 2>/dev/null | head -1 || true)
if [ -n "${LATEST}" ]; then
  CKPT="${LATEST}"
  echo "[relaunch-c] resuming from ${LATEST##*/}"
else
  CKPT="${LESSAC}"
  echo "[relaunch-c] cold-starting from Lessac low"
fi

BATCH=${PIPER_BATCH:-32}
mkdir -p "${CACHE_DIR}" "${RUN}/phase4-low/logs"

systemctl --user reset-failed "${UNIT}" 2>/dev/null || true
systemctl --user stop "${UNIT}" 2>/dev/null || true

systemd-run --user \
  --unit="${UNIT}" \
  --description="Piper VITS fine-tune (Takashii — LOW, resumable)" \
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
    --data.batch_size "${BATCH}" \
    --trainer.devices "[0]" \
    --trainer.max_steps 30000 \
    --trainer.val_check_interval 0.5 \
    --trainer.check_val_every_n_epoch 1 \
    --trainer.callbacks+=ModelCheckpoint \
    --trainer.callbacks.dirpath="${RUN}/phase4-low/checkpoints" \
    --trainer.callbacks.save_top_k=-1 \
    --trainer.callbacks.every_n_train_steps=1000 \
    --ckpt_path "${CKPT}"

echo "[relaunch-c] unit started"
