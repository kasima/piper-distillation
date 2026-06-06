#!/usr/bin/env bash
# One-shot bootstrap: venv + piper1-gpl fork + warm-start checkpoints + sources.
# Idempotent: re-running upgrades only what's missing.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPER1_FORK="https://github.com/kasima/piper1-gpl.git"
PIPER1_BRANCH="kasima/torch-2.6-compat"

cd "${REPO}"

echo "=== [1/6] check system prerequisites ==="
missing=()
for bin in cmake ninja espeak-ng python3 wget; do
  command -v "$bin" >/dev/null || missing+=("$bin")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "missing: ${missing[*]}"
  echo "install:  sudo apt install -y cmake ninja-build espeak-ng wget"
  exit 1
fi

echo "=== [2/6] clone the patched piper1-gpl fork ==="
if [ ! -d piper1-gpl ]; then
  git clone --branch "${PIPER1_BRANCH}" --single-branch "${PIPER1_FORK}" piper1-gpl
else
  echo "piper1-gpl/ exists; skipping clone"
fi

echo "=== [3/6] create venv + install non-editable (compiles espeakbridge.so via CMake) ==="
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip wheel setuptools
fi
# Trick: scikit-build's CMake step only fires on non-editable installs.
# Install once non-editable so espeakbridge.so + espeak-ng-data get built and
# placed into the source tree alongside the source files. Then reinstall
# editable so we can hack scripts/piper1-gpl together as a workspace.
.venv/bin/pip install --quiet ./piper1-gpl

echo "=== [4/6] copy compiled C extension + espeak data into source, reinstall editable ==="
cp -f .venv/lib/python3.12/site-packages/piper/espeakbridge.so piper1-gpl/src/piper/
cp -rf .venv/lib/python3.12/site-packages/piper/espeak-ng-data piper1-gpl/src/piper/ 2>/dev/null || true
.venv/bin/pip uninstall -y piper-tts >/dev/null
.venv/bin/pip install --quiet -e './piper1-gpl[train]'

# Build monotonic_align Cython extension (separate Cython build, not scikit-build).
# build_monotonic_align.sh calls `cythonize`, which is NOT pulled in by
# piper1-gpl[train]; install Cython explicitly first or the build fails with
# "cythonize: command not found".
.venv/bin/pip install --quiet cython
(cd piper1-gpl && PATH="${REPO}/.venv/bin:$PATH" ./build_monotonic_align.sh)

echo "=== [5/6] install eval + analysis deps ==="
.venv/bin/pip install --quiet \
  torchaudio \
  openai-whisper \
  speechbrain \
  transformers \
  librosa \
  soundfile \
  datasets \
  onnxscript

echo "=== [6/6] download + clean Lessac warm-start checkpoints ==="
mkdir -p checkpoints
fetch_and_clean() {
  local tier="$1" url="$2" raw="$3" clean="$4"
  if [ -f "checkpoints/${clean}" ]; then
    echo "checkpoints/${clean} present; skipping"
    return
  fi
  if [ ! -f "checkpoints/${raw}" ]; then
    echo "downloading Lessac ${tier}..."
    wget -q --show-progress "${url}" -O "checkpoints/${raw}"
  fi
  echo "cleaning hparams (Lightning 1.9 → 2.x compat)..."
  .venv/bin/python3 - <<PY
import pathlib, torch, inspect, sys, os
sys.path.insert(0, 'piper1-gpl/src')
torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
ckpt = torch.load('checkpoints/${raw}', map_location='cpu', weights_only=False)
from piper.train.vits.lightning import VitsModel
sig = inspect.signature(VitsModel.__init__)
allowed = set(sig.parameters.keys()) - {'self', 'kwargs', 'args'}
hp = ckpt.get('hyper_parameters', {})
ckpt['hyper_parameters'] = {k: v for k, v in hp.items() if k in allowed}
torch.save(ckpt, 'checkpoints/${clean}')
print(f'wrote checkpoints/${clean} ({os.path.getsize("checkpoints/${clean}")//1024//1024} MB)')
PY
  rm -f "checkpoints/${raw}"
}
fetch_and_clean medium \
  "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt" \
  "en_US-lessac-medium.ckpt" \
  "en_US-lessac-medium-clean.ckpt"

# Low only fetched if requested (Run C is optional and the 'low' tier produces
# a noticeably different voice — see sysadmin/piper FOLLOW_UPS.md).
if [ "${SETUP_FETCH_LOW:-0}" = "1" ]; then
  fetch_and_clean low \
    "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/low/epoch%3D2307-step%3D558536.ckpt" \
    "en_US-lessac-low.ckpt" \
    "en_US-lessac-low-clean.ckpt"
fi

echo
echo "=== setup complete ==="
echo "Next step: see README.md > 'Running the pipeline for a new voice'"
echo "Venv:           ${REPO}/.venv"
echo "Piper1-gpl:     ${REPO}/piper1-gpl  (branch ${PIPER1_BRANCH})"
echo "Lessac medium:  ${REPO}/checkpoints/en_US-lessac-medium-clean.ckpt"
