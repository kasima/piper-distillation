# piper-distillation — agent execution reference

Distill a custom voice from `qwen-tts` (or any OpenAI-compatible TTS that
supports voice cloning) into a deployable Piper VITS model.

**Audience: a coding agent executing the pipeline.** This file is the
control surface — not a tutorial. Read top-to-bottom once, then operate on
the scripts directly. The repo's [`README.md`](README.md) is the human-facing
introduction; it points here for execution details.

## What this is

A 7-phase pipeline that takes:
- A teacher voice exposed via qwen-tts's `/v1/audio/speech` (variant=base,
  voice-cloning from a 3-10 s reference clip)

…and produces:
- One or more `en_US-<voice>-<quality>.onnx` + `.onnx.json` files
- Installed into a `wyoming-piper` data dir, served by the existing piper
  service over Wyoming protocol

Originally executed for the `Takashii` voice on bernard, producing three
deliverables: `en_US-takashii-medium`, `en_US-takashii-medium-full`,
`en_US-takashii-low`. See
[`sysadmin/piper/journal/2026-05-25-takashii-distillation.md`](../sysadmin/piper/journal/2026-05-25-takashii-distillation.md)
for the original run's metrics, dead-ends, and gotchas. Open issues and
deferred improvements live in
[`sysadmin/piper/FOLLOW_UPS.md`](../sysadmin/piper/FOLLOW_UPS.md).

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
run_config.json                voice-specific config (paths, teacher endpoint, GPU pinning, quality bar)
setup.sh                       one-shot bootstrap (venv, fork, warm-start ckpts)

checkpoints/                   warm-start Lessac ckpts (downloaded by setup.sh)
piper1-gpl/                    cloned + patched piper trainer (cloned by setup.sh from github.com/kasima/piper1-gpl)
.venv/                         project Python venv

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

Everything below the dashed line is gitignored. Only `scripts/`,
`run_config.json`, `setup.sh`, `README.md`, `.gitignore` are versioned.

## Prerequisites

System packages (`apt`):
```
cmake ninja-build espeak-ng wget python3-venv
```

Network access for: HuggingFace (warm-start ckpts), Project Gutenberg
(corpus sources), pypi (deps), github.com/kasima/piper1-gpl (fork clone).

Hardware: at least one GPU with ≥45 GB VRAM (training); a separate GPU or
the same one if the teacher's TTS server can be temporarily stopped. The
original Takashii run used an A6000 for training and a 4070 Ti for the
qwen-tts teacher — two GPUs, no contention.

A running teacher: a qwen-tts service (or any OpenAI-compatible TTS) with
the target voice registered. For qwen-tts on bernard, see
[`sysadmin/qwen-tts/README.md`](../sysadmin/qwen-tts/README.md) §"Adding a
voice to the base variant."

A target wyoming-piper service (or any process that scans an `.onnx` data
dir) where Phase 6 should install the result. For bernard, see
[`sysadmin/piper/README.md`](../sysadmin/piper/README.md).

## First-time setup

```
bash setup.sh
```

Idempotent. Roughly 10 min on first run (mostly the Lessac ckpt download +
CMake build of espeakbridge.so). Re-running upgrades pip deps and skips
work that's already done.

Set `SETUP_FETCH_LOW=1 bash setup.sh` to additionally download the Lessac
LOW warm-start checkpoint (only needed if you intend to train a "low"
quality variant — see [FOLLOW_UPS](../sysadmin/piper/FOLLOW_UPS.md) for
why "low" isn't always what you'd guess).

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

The rest can stay as-is for Takashii-style runs. For an English voice with
similar character, the existing thresholds work; for a wildly different
voice, expect to recalibrate after Phase 0.

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

The script writes `<voice>.onnx` + `<voice>.onnx.json` to
`~/src/sysadmin/piper/models/` (verify this matches your wyoming-piper
data dir), runs a 30-clip smoke test, and writes a model card.

The `.onnx.json` `dataset` field is set to the voice_name automatically.
Critical gotcha — see
[`reference_wyoming_piper_custom_voice_naming.md`](~/.claude/projects/-home-kasima-src-sysadmin/memory/reference_wyoming_piper_custom_voice_naming.md)
in long-term memory for why.

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

## Future-work toolkit-repo extraction

Open work — see
[`sysadmin/piper/FOLLOW_UPS.md`](../sysadmin/piper/FOLLOW_UPS.md). Goal:
factor `scripts/` into a standalone Python package that takes
`(sample_wav, transcript, voice_name)` and produces a deployed voice with
no manual editing required. The pieces are all here; what's missing is
parameterizing the path/voice/quality config into a single command-line
entrypoint.
