#!/usr/bin/env bash
# Run B — train Piper on the FULL dataset (curated + recovered WER drops).
# Same config as phase4_train.sh but reads from phase3-full/ and writes
# checkpoints to phase4-full/checkpoints. Use a distinct systemd unit name.

set -euo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_ID="$(python3 -c "import json,sys;print(json.load(open(\"${RUN}/run_config.json\"))[\"teacher\"][\"voice_id\"])")"
OUT="${RUN}/output/${VOICE_ID}"
VOICE_LOWER="$(echo "${VOICE_ID}" | tr '[:upper:]' '[:lower:]')"
VENV="${RUN}/.venv"
CKPT="${RUN}/checkpoints/en_US-lessac-medium-clean.ckpt"
TRAIN_META="${OUT}/phase4-full/train_metadata.csv"
AUDIO_DIR="${OUT}/phase3-full/audio"
CACHE_DIR="${OUT}/phase4-full/cache"
CFG_PATH="${OUT}/phase4-full/config.json"
UNIT="piper-train-${VOICE_LOWER}-full"

# Pre-flight: GPU 1 free
USED=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$USED" -gt 2000 ]; then
  echo "GPU 1 has ${USED} MiB used — Run A or vllm-aeon likely still on GPU 1. Stop first."
  exit 1
fi
echo "GPU 1 free (used=${USED} MiB)"

BATCH=${PIPER_BATCH:-32}
mkdir -p "${CACHE_DIR}" "${OUT}/phase4-full/logs" "${OUT}/phase4-full/checkpoints"

systemd-run --user \
  --unit="${UNIT}" \
  --description="Piper VITS fine-tune (Takashii — FULL dataset)" \
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

sleep 2
systemctl --user status "${UNIT}" --no-pager -l | head -25 || true
echo
echo "Tail logs: journalctl --user -u ${UNIT} -f"
