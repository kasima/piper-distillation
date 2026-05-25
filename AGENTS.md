# piper-distillation — agent execution reference

Distill a custom voice from `qwen-tts` (or any OpenAI-compatible TTS that
supports voice cloning) into a deployable Piper VITS model.

**Audience: a coding agent executing the pipeline.** This file is the
control surface — not a tutorial. Read top-to-bottom once, then operate on
the scripts directly. The repo's [`README.md`](README.md) is the human-facing
introduction; it points here for execution details.

**Design intent: fully autonomous, end-to-end.** The pipeline is meant to
run unattended for ~18 hours from kickoff to a deployed voice. Don't pause
to ask the user for intermediate decisions, confirmation, or input. Pick
reasonable defaults; if something is genuinely ambiguous, write a line to
`output/<voice>/state/notifications.txt` and keep going. Halt **only** on
the documented halt conditions below (teacher consistency below floor,
retained audio below 2 h, NaN/OOM/divergence, etc.). Anything else —
phoneme coverage warnings, transient HTTP failures during synth, single-
checkpoint divergences in eval — should be logged and worked around, not
escalated.

## What this is

A 7-phase pipeline that takes:
- A teacher voice exposed via qwen-tts's `/v1/audio/speech` (variant=base,
  voice-cloning from a 3-10 s reference clip)

…and produces:
- One or more `en_US-<voice>-<quality>.onnx` + `.onnx.json` files
- Installed into a `wyoming-piper` data directory, served via Wyoming
  protocol

## Pipeline at a glance

```
Phase 0 — teacher characterization     60 probe sentences × 2 takes; ECAPA + WavLM self-similarity baseline; derive Phase 3 thresholds
Phase 1 — text corpus curation         greedy-select 12000 sentences from Project Gutenberg with phoneme/prosody balancing
Phase 2 — corpus-wide synthesis        12000 WAVs via qwen-tts at concurrency=2; resumable via state/phase2/done.txt; ~13 h
Phase 3 — QA filter                    drop teacher failures (WER, silence, clipping, dual-embedding speaker verify); resample to 22 kHz
Phase 4 — fine-tune from Lessac        VITS training on A6000 from a cleaned Lessac warm-start ckpt; resumable; ~3 h per run
Phase 5 — eval top-3 ckpts             ECAPA + WavLM + Whisper WER + composite score; winner = mean − ½·variance
Phase 6 — ONNX export + install        export ONNX, write .onnx.json, smoke-test 30 sentences, install to wyoming-piper data dir
```

Each phase writes a manifest to `state/phase<N>/manifest.json`. A failed
phase halts the pipeline; resume by re-running its script. Orchestration
script `scripts/orchestrate.py` chains them all and survives shell exits.

## Layout

```
scripts/                       all pipeline code (Python + bash)
scripts/probe_sentences.txt    60-sentence accent-diagnostic probe set (voice-agnostic, seeds Phase 0)
run_config.json                active per-voice config (teacher endpoint, voice_id, GPU pinning, quality bar)
setup.sh                       one-shot bootstrap (venv, fork, warm-start ckpts)
README.md                      human-facing intro
AGENTS.md                      this file (dense agent execution reference)

checkpoints/                   warm-start Lessac ckpts (shared across voices; downloaded by setup.sh)
piper1-gpl/                    cloned + patched piper trainer (cloned by setup.sh)
.venv/                         project Python venv

output/<voice_id>/             ALL per-run artifacts, keyed by run_config.json's teacher.voice_id
  phase1/sources/                Gutenberg source text
  phase1/corpus.txt              greedy-selected 12000 sentences
  phase2/audio_raw/              12000 teacher WAVs at 24 kHz
  phase3/audio/                  filtered + 22 kHz resampled (Run A input)
  phase3-full/audio/             full set (curated + recovered WER drops) (Run B input)
  phase3-low/audio/              full set resampled to 16 kHz (Run C input)
  phase4{,-full,-low}/checkpoints/  per-run training checkpoints
  phase4{,-full,-low}/{cache,logs,config.json,*_metadata.csv}  per-run state
  phase5{,-full,-low}/<step>/    per-checkpoint eval clips + metrics
  phase6{,-full,-low}/<voice>/   per-voice deliverable (onnx + json + smoke + model_card.md)
  state/phase{0..6}{,-full,-low}/manifest.json   per-phase manifests
  state/notifications.txt        timestamped alerts during a run
  logs/                          per-phase stdout/stderr
```

Scripts derive `OUT = repo_root / "output" / <voice_id>` at import time by
reading `run_config.json`. All phase outputs land under `OUT/`. Switching
voices means editing `run_config.json` (specifically `teacher.voice_id` +
`voice_name`); previous voices' artifacts remain in their own
`output/<previous_voice_id>/` dirs.

Everything under `output/`, `checkpoints/`, `piper1-gpl/`, and `.venv/`
is gitignored. Only `scripts/`, `run_config.json`, `setup.sh`, `README.md`,
`AGENTS.md`, `.gitignore` are versioned.

## Prerequisites

System packages (`apt`):
```
cmake ninja-build espeak-ng wget python3-venv
```

Network access for: HuggingFace (warm-start ckpts), Project Gutenberg
(corpus sources), pypi (deps), github.com/kasima/piper1-gpl (fork clone).

Hardware: at least one GPU with ≥45 GB VRAM for training (A6000-class) and
a separate ~12 GB-class GPU for the teacher TTS — or the same GPU if you
can stop the teacher's server during Phases 4-5. Two GPUs is the
recommended layout (no contention; the teacher stays up to serve other
clients during the 13-hour Phase 2).

A running teacher: a voice-cloning TTS exposing
`POST /v1/audio/speech` (qwen-tts works; any OpenAI-compatible TTS that
supports reference-based voice cloning should work with minor edits to
`scripts/phase2_synthesize.py`). The target voice must already be
registered in the teacher.

A target `wyoming-piper` data directory where Phase 6 installs the result.
Configure the install path in `scripts/phase6_install.py`
(`SERVICE_MODELS` constant) — by default it points at a host-specific
location and you'll want to change this for a new deployment.

## First-time setup

```
bash setup.sh
```

Idempotent. Roughly 10 min on first run (mostly the Lessac ckpt download +
CMake build of espeakbridge.so). Re-running upgrades pip deps and skips
work that's already done.

Set `SETUP_FETCH_LOW=1 bash setup.sh` to additionally download the Lessac
LOW warm-start checkpoint (only needed if you intend to train a "low"
quality variant — note that the Piper "low" quality is 16 kHz output and
produces noticeably different pronunciation than the medium variant on
the same data, not just a downsampled version; treat it as a separate
voice).

## Running the pipeline for a new voice

Steps assume the teacher voice is already registered in qwen-tts.

### 1. Edit `run_config.json`

Critical fields:
- `voice_name`: target deployed name, e.g. `en_US-akira-medium`
- `teacher.voice_id`: qwen-tts registry name (case-sensitive)
- `teacher.ref_path` + `teacher.ref_sha256`: pin the teacher's reference clip
  (sha256 detects any silent ref swap; compute via
  `sha256sum ~/.cache/qwen-tts/voices/<NAME>/ref.wav`)
- `host.training_gpu_index` / `teacher_gpu_index`: confirm the GPU
  indices match the actual host

The rest can stay as-is for typical English-voice runs. For a wildly
different voice (e.g., heavy accent, very different fundamental
frequency), expect to recalibrate Phase 3 thresholds after Phase 0
measures the teacher's natural variance.

### 2. Phase 0 — probe the teacher

```
.venv/bin/python3 scripts/phase0_synthesize.py       # ~10 min  (60 sentences × 2 takes)
.venv/bin/python3 scripts/phase0_consistency.py      # ~5 min   (ECAPA + WavLM centroids; derives Phase 3 thresholds)
```

Halt condition: `state/phase0/consistency.json` shows
`ecapa_self_similarity.mean < 0.75`. The teacher is too inconsistent to
clone reliably; pick a different teacher config (lower temperature if the
service exposes it, or change voice).

Audit: listen to a few clips in `state/phase0/clips/first/`. If the voice
doesn't sound like what's expected, the teacher voice registration is
wrong — fix qwen-tts before continuing.

### 3. Phase 1 — corpus

```
.venv/bin/python3 scripts/phase1_curate.py           # ~5 min
```

Deterministic given the Gutenberg sources in `phase1/sources/` (setup.sh
seeds 10 books). To use a different corpus, drop additional `*.txt` files
into `phase1/sources/` and re-run; the greedy selector merges them.

Manifest at `state/phase1/manifest.json`, coverage report at
`state/phase1/coverage_report.md`. Verify each accent-marker phoneme
appears ≥3000 times in the report (heuristic floor for VITS to learn the
phoneme reliably).

### 4. Phase 2 — synthesize the full corpus

```
nohup .venv/bin/python3 scripts/phase2_synthesize.py > logs/phase2.log 2>&1 < /dev/null &
disown
```

Long-running (~11-13 h). Resumable via `state/phase2/done.txt`. Monitor
with `tail -f logs/phase2.log` or `cat state/phase2/synthesis_log.jsonl |
tail -1`.

Halt conditions: rolling latency increases >50% (thermal/memory issue) or
>20% of clips fail. Auto-retries individual failures up to 5x.

### 5. Phase 3 — QA filter

**Stop the vllm-aeon or whatever holds GPU 1 first.** Whisper-medium needs
~1.4 GB free on the GPU alongside ECAPA + WavLM.

```
sudo systemctl stop vllm-aeon          # or whatever holds GPU 1
.venv/bin/python3 scripts/phase3_filter.py     # ~1.5 h
```

Halt conditions: `state/phase3/manifest.json` shows `retained_hours < 2`.
Dataset too small — go back and add more sources to Phase 1 or check what
filter dominated (`drop_histogram`).

### 6. Phase 4 — fine-tune (Run A: curated)

```
.venv/bin/python3 scripts/phase4_prepare.py        # train/eval split
bash scripts/phase4_train.sh                        # launches systemd-run --user --unit=piper-train-<voice>
# monitor:  journalctl --user -u piper-train-<voice> -f
```

Default cap: 40000 steps (~3 h on A6000). Checkpoints every 1000 steps to
`phase4/checkpoints/`. Resumable: re-run `scripts/phase4b_relaunch.sh`
which picks up the latest checkpoint. Optional: spawn
`scripts/phase4b_watchdog.sh` as a sibling for auto-restart on crash.

Halt conditions: NaN loss, OOM, generator/discriminator ratio > 10× or
< 0.1× for 5 consecutive checkpoints, GPU temp sustained > 85°C.

### 7. Optional Run B — full dataset with WER-recovered clips

The Phase 3 WER filter biases against the most accented clips (the
distinctive signal). Run B re-injects them:

```
.venv/bin/python3 scripts/phase3b_full_dataset.py    # ~10 sec; hardlinks + recovers WER drops
.venv/bin/python3 scripts/phase4b_prepare.py
bash scripts/phase4_train_full.sh                     # ~3 h
```

By ear, Run B is more accented than Run A. Worth doing if accent fidelity
is the goal.

### 8. Phase 5 — eval

Per run. `PIPER_DISTILL_EVAL_VARIANT` switches the path roots:

```
# Run A:
.venv/bin/python3 scripts/phase5_eval.py \
  --checkpoint phase4/checkpoints/<step38k>.ckpt \
  --checkpoint phase4/checkpoints/<step39k>.ckpt \
  --checkpoint phase4/checkpoints/<step40k>.ckpt

# Run B:
PIPER_DISTILL_EVAL_VARIANT=full .venv/bin/python3 scripts/phase5_eval.py \
  --checkpoint phase4-full/checkpoints/<...>
```

Winner picked by `composite_mean - 0.5 * composite_var` (rewards
consistency). Manifest at `state/phase5{,-full}/manifest.json`.

### 9. Phase 6 — install

```
.venv/bin/python3 scripts/phase6_install.py \
  --checkpoint "$(jq -r .winner state/phase5/manifest.json)" \
  --voice-name "<voice_name from run_config.json>" \
  --training-config phase4/config.json \
  --metrics-json "state/phase5/$(basename $(jq -r .winner state/phase5/manifest.json) .ckpt)/metrics.json" \
  --dataset-label "<label for model card>"
sudo systemctl restart piper
```

The script writes `<voice>.onnx` + `<voice>.onnx.json` to the path in
`SERVICE_MODELS` (edit `scripts/phase6_install.py` if your wyoming-piper
data directory is elsewhere), runs a 30-clip smoke test, and writes a
model card.

The `.onnx.json` `dataset` field is set to the voice_name automatically.
**Critical gotcha**: `wyoming-piper`'s custom-voice discovery uses the
JSON `dataset` field as the voice name advertised to clients (e.g., Home
Assistant), but the synth handler then calls `find_voice(voice_name)`
which expects `{voice_name}.onnx` on disk. **The `dataset` field must
equal the .onnx filename stem** or every synth request fails with
`VoiceNotFoundError`. Don't override this; `phase6_install.py` enforces
it.

## Orchestration (chain it all)

```
PYTHONUNBUFFERED=1 nohup .venv/bin/python3 -u scripts/orchestrate.py >> logs/orchestrate.log 2>&1 < /dev/null &
disown
```

The orchestrator runs phase 2 wait → Phase 3 → 4 train (Run A) → Phase 5 →
Phase 6 → Run B prep → Phase 4 train (Run B) → Phase 5 (Run B) → Phase 6
(Run B) → restart piper.service. Skip-cascade: if a phase manifest exists,
it's skipped. Resumable after crashes.

Status: `bash scripts/status.sh`. Notifications: `state/notifications.txt`.

## Halt conditions (do not silently continue past these)

| Phase | Condition | Action |
|---|---|---|
| 0 | ECAPA self-sim < 0.75 | Teacher unfit; reconfigure or change voice |
| 1 | Any accent phoneme < 1000 occurrences in corpus | Add more sources, re-run greedy selection |
| 2 | Latency p50 > 2× baseline for 10 min | Thermal/memory issue; investigate |
| 3 | retained_hours < 2 | Dataset insufficient; revisit Phases 1-2 or WER threshold |
| 4 | NaN loss, OOM, GPU temp > 85°C sustained | Stop unit, investigate, possibly reduce batch |
| 5 | composite_mean < 0.6 on best checkpoint | Voice didn't converge to teacher; consider more steps or re-train |

## Known patches needed against upstream piper1-gpl

The `kasima/piper1-gpl` fork on branch `kasima/torch-2.6-compat` carries:

1. `src/piper/train/__main__.py` — `torch.serialization.add_safe_globals([PosixPath, WindowsPath])` for PyTorch 2.6's `weights_only=True` default
2. `src/piper/train/export_onnx.py` — `dynamo=False` on `torch.onnx.export` (new dynamo exporter rejects Piper's spline assert)

`setup.sh` clones this fork branch directly. If upstream merges the fixes,
update `PIPER1_BRANCH` in `setup.sh` to `main`.

## When something goes wrong

1. Check `state/notifications.txt` first — orchestrator writes here on
   restarts and halt conditions.
2. Per-phase log under `logs/`.
3. systemd unit log: `journalctl --user -u piper-train-<voice> --no-pager | tail -100`.
4. Manifests under `state/phase<N>/` — every successful phase writes one;
   their absence locates the failure.

Resumability: each phase is restartable. Phases 2, 3, 4 have explicit
`done.txt` / checkpoint-based resume. Phases 1, 5, 6 are cheap enough to
re-run end-to-end.

## Future-work toolkit extraction

The pieces are all here, but the scripts still carry a few host-specific
assumptions (install path in `phase6_install.py`, systemd unit naming,
GPU index assumptions). A future refactor would:

1. Parameterize all paths via CLI args + a single per-host config file
   (not embedded in `run_config.json` which is per-voice)
2. Replace systemd-based train launching with a portable backend
   (subprocess + PID file, or a thin wrapper that detects systemd vs
   non-systemd)
3. Auto-detect GPU layout via `nvidia-smi` rather than assume specific
   indices
4. Package as `pip install piper-distillation` with a single
   `piper-distill --ref-wav … --voice-name … --teacher-url …` entrypoint

None of this changes the pipeline's *semantics*; just makes it deployable
on more than the original host without surgery.
