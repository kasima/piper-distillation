#!/usr/bin/env bash
# Smart relaunch for Run B — resumes from the latest saved checkpoint if one
# exists, otherwise cold-starts from Lessac. Idempotent; safe to call after a
# crash.

set -euo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_ID="$(python3 -c "import json,sys;print(json.load(open(\"${RUN}/run_config.json\"))[\"teacher\"][\"voice_id\"])")"
OUT="${RUN}/output/${VOICE_ID}"
VENV="${RUN}/.venv"
TRAIN_META="${OUT}/phase4-full/train_metadata.csv"
AUDIO_DIR="${OUT}/phase3-full/audio"
CACHE_DIR="${OUT}/phase4-full/cache"
CFG_PATH="${OUT}/phase4-full/config.json"
UNIT="piper-train-takashii-full"
LESSAC="${RUN}/checkpoints/en_US-lessac-medium-clean.ckpt"

# Pre-flight: GPU 1 free
USED=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$USED" -gt 2000 ]; then
  echo "[relaunch] GPU 1 has ${USED} MiB used — refusing to launch"
  exit 1
fi

# Pick the latest checkpoint as resume point (Lightning's ckpt_path restores
# optimizer/epoch/step). Fall back to Lessac if none.
LATEST=$(ls -t "${OUT}/phase4-full/checkpoints"/*step=*.ckpt 2>/dev/null | head -1 || true)
if [ -n "${LATEST}" ]; then
  CKPT="${LATEST}"
  echo "[relaunch] resuming from ${LATEST##*/}"
else
  CKPT="${LESSAC}"
  echo "[relaunch] cold-starting from Lessac (no prior checkpoints)"
fi

BATCH=${PIPER_BATCH:-32}
mkdir -p "${CACHE_DIR}" "${OUT}/phase4-full/logs"

# Clear any prior failed transient unit
systemctl --user reset-failed "${UNIT}" 2>/dev/null || true
systemctl --user stop "${UNIT}" 2>/dev/null || true

systemd-run --user \
  --unit="${UNIT}" \
  --description="Piper VITS fine-tune (Takashii — FULL dataset, resumable)" \
  --setenv=CUDA_VISIBLE_DEVICES=1 \
  --setenv=PYTHONUNBUFFERED=1 \
  --working-directory="${RUN}/piper1-gpl" \
  -- \
  "${VENV}/bin/python3" -m piper.train fit \
    --data.voice_name "en_US-takashii-medium-full" \
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
    --trainer.callbacks.dirpath="${OUT}/phase4-full/checkpoints" \
    --trainer.callbacks.save_top_k=-1 \
    --trainer.callbacks.every_n_train_steps=1000 \
    --ckpt_path "${CKPT}"

echo "[relaunch] unit started"
