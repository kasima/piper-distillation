#!/usr/bin/env bash
# Phase 6 — export winning checkpoint to ONNX, smoke-test, restart vllm-aeon.

set -euo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_ID="$(python3 -c "import json,sys;print(json.load(open(\"${RUN}/run_config.json\"))[\"teacher\"][\"voice_id\"])")"
OUT="${RUN}/output/${VOICE_ID}"
VOICE_LOWER="$(echo "${VOICE_ID}" | tr '[:upper:]' '[:lower:]')"
VENV="${RUN}/.venv"
STATE="${OUT}/state/phase6"
mkdir -p "${STATE}"

WINNER=$(jq -r '.winner' "${OUT}/state/phase5/manifest.json")
if [ ! -f "${WINNER}" ]; then
  echo "winner checkpoint not found: ${WINNER}" >&2
  exit 1
fi

OUT_DIR="${OUT}/phase6/en_US-takashii-medium"
mkdir -p "${OUT_DIR}"

ONNX="${OUT_DIR}/en_US-takashii-medium.onnx"
ONNX_JSON="${ONNX}.json"

echo "=== exporting ONNX ==="
cd "${RUN}/piper1-gpl"
"${VENV}/bin/python3" -m piper.train.export_onnx \
  --checkpoint "${WINNER}" \
  --output-file "${ONNX}"

echo "=== smoke test (30 sentences) ==="
SMOKE_DIR="${OUT_DIR}/smoke_test"
mkdir -p "${SMOKE_DIR}"
i=0
head -30 "${OUT}/phase4/eval_metadata.csv" | while IFS='|' read -r sid sentence; do
  out="${SMOKE_DIR}/${sid}.wav"
  echo "$sentence" | "${VENV}/bin/python3" -m piper \
    --model "${ONNX}" \
    --output-file "${out}"
  if [ ! -s "${out}" ]; then
    echo "FAIL: empty output for ${sid}" >&2
    exit 1
  fi
  i=$((i+1))
done
echo "smoke test: ${i}/30 clips produced"

echo "=== restarting vllm-aeon ==="
sudo -n systemctl start vllm-aeon
sleep 5
sudo -n systemctl is-active vllm-aeon
echo "vllm-aeon restarted"

# Model card
cat > "${OUT_DIR}/model_card.md" <<EOF
# en_US-takashii-medium — Piper VITS

Distillation of the qwen-tts Takashii voice clone into a Piper VITS model.

## Teacher

- Service: qwen-tts (Qwen3-TTS 1.7B, variant=base, voice clone)
- Voice ID: Takashii
- Reference clip SHA-256: 4fb00f2d1fadb123a274aad25a329230cf31937e6d3d808a28d6950b37f95448
- Endpoint at training time: http://192.168.122.1:8880

## Training data

See state/phase3/manifest.json for filtered hours, drop histogram, and coverage.

## Pretrained checkpoint

en_US-lessac-medium.ckpt (Lessac voice, Piper medium quality VITS).

## Final metrics (Phase 5)

See state/phase5/manifest.json.

## Known limitations

- Distilled from a generative voice clone, not the original speaker; expect some
  loss of micro-prosodic detail vs the teacher's WavLM-0.97 self-similarity.
- Accent features (/r/-/l/, /θ/-/ð/, English vowels absent in JA) inherited from
  the teacher; deeper analysis deferred to follow-up if needed.

## Inference

\`\`\`
piper --model en_US-takashii-medium.onnx --output-file out.wav < input.txt
\`\`\`
EOF

cat > "${STATE}/manifest.json" <<EOF
{
  "phase": "phase6",
  "onnx_path": "${ONNX}",
  "onnx_json_path": "${ONNX_JSON}",
  "model_card_path": "${OUT_DIR}/model_card.md",
  "smoke_test_count": ${i},
  "vllm_aeon_restarted": true,
  "winner_checkpoint": "${WINNER}"
}
EOF

echo
echo "=== done ==="
echo "model: ${ONNX}"
echo "config: ${ONNX_JSON}"
echo "card: ${OUT_DIR}/model_card.md"
