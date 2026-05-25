# piper-distillation

Take a custom voice from an OpenAI-compatible voice-cloning TTS server
(originally [qwen-tts](https://github.com/QwenLM/Qwen2-Audio) running
reference-clip voice cloning) and produce a self-contained
[Piper](https://github.com/OHF-Voice/piper1-gpl) VITS `.onnx` model you can
deploy to a `wyoming-piper` host or any ONNX TTS runtime that understands
Piper voice configs.

Useful when you want to keep using a particular voice but the original
serving stack is too heavy for real-time use. The teacher TTS is run
once over a curated text corpus; the resulting (text, audio) pairs train a
student Piper model that synthesizes the same voice at >40× faster than
real-time on CPU.

## What it produces

A ~30-60 MB `.onnx` file plus a small `.onnx.json` config, e.g.
`en_US-<yourvoice>-medium.onnx`. Drop the pair into a Piper data
directory and the voice is reachable via:

- **Wyoming protocol** (`wyoming-piper`, what Home Assistant talks to)
- **Piper CLI** (`piper --model en_US-<yourvoice>-medium.onnx < input.txt`)
- **Anything else that loads ONNX VITS models** (e.g.
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx))

## When you'd use this

You have a voice that exists only as a clone in a reference-based TTS
(qwen-tts, XTTS, similar), and you want:

- Lower per-request latency than the source TTS (real-time conversational use)
- A static, deployable artifact (no GPU server dependency at inference time)
- A voice servable by the broader Piper ecosystem

You'd **not** use this if the voice already has a Piper model in the
upstream catalog ([rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)),
if you can re-record a real speaker (cleaner training data), or if your
voice needs are met by a generic neutral voice.

## How the pipeline works (conceptually)

```
Phase 0:   characterize the teacher (does it sound consistent enough to clone?)
Phase 1:   pick a balanced 12k-sentence text corpus
Phase 2:   synthesize all 12k sentences from the teacher
Phase 3:   filter out the teacher's bad takes (silence, clipping, mispronunciations)
Phase 4:   fine-tune a Piper VITS model on the filtered (text, audio) pairs
Phase 5:   pick the best checkpoint by ECAPA + WavLM + Whisper-WER composite
Phase 6:   export to ONNX, smoke-test, install to a Piper data directory
```

Each phase writes a manifest under `state/phase<N>/manifest.json`. The
orchestrator chains them and skips phases whose manifest is already
present, so the whole thing is resumable after crashes.

## Timing breakdown

Approximate wall-clock for one voice (one training run). Numbers assume
two GPUs available — one for the teacher TTS (modest, ~12 GB class), one
for training + Whisper-medium QA (≥45 GB free, A6000-class). The two
biggest phases (2 and 4) are the dominant cost.

| Phase | What | Wall-clock | Where it runs |
|---|---|---|---|
| 0 | Probe teacher consistency | ~15 min | teacher GPU + embedding GPU |
| 1 | Greedy-select 12k-sentence corpus | ~5 min | CPU |
| 2 | Synthesize 12k clips from the teacher | **~13 h** | teacher GPU |
| 3 | QA filter (WER + speaker verify + audio sanity) | ~1.5 h | training GPU |
| 4 | Fine-tune Piper VITS (40k steps from Lessac) | **~3 h** | training GPU |
| 5 | Eval top-3 checkpoints, pick winner | ~30 min | training GPU |
| 6 | ONNX export + smoke test + install | ~5 min | CPU |
| | **Total per voice (single training run)** | **~18 h** | |
| | Per additional training variant (same data) | +3.5 h | training GPU |

Phase 2 (corpus synthesis) is the long pole and depends on the teacher
TTS's per-request latency; the figure above is for a ~1.7 B-parameter
voice-cloning TTS at concurrency 2. A faster teacher cuts this
proportionally.

Phase 4 cap of 40k steps is conservative; Piper VITS from a Lessac
warm-start often converges by 25-30k. Smaller architectures (`low`,
`x_low`) train faster but produce noticeably different audio — see
notes in `AGENTS.md` and the project follow-ups before assuming "low =
faster same voice".

## How to use it

This repo is meant to be operated by a coding agent. Your job as the
human is to satisfy the prerequisites, then hand the agent a prompt.

### Prerequisites

1. **A running OpenAI-compatible voice-cloning TTS** with your target
   voice already registered. The pipeline calls `POST /v1/audio/speech`
   and expects a `voice` parameter that picks the cloned voice.
   [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) is what was tested —
   see its docs for how to register a voice from a reference clip. Any
   other TTS that speaks the same API contract should work with minor
   edits to `scripts/phase2_synthesize.py`. Note the endpoint URL and
   the voice ID.

2. **A GPU with ≥16 GB VRAM.** VITS training at batch 32 peaks around
   14-15 GB; Phases 3 and 5 (Whisper-medium + ECAPA + WavLM) fit
   comfortably in the rest. The teacher TTS gets stopped (or
   time-sliced) during Phases 4-5 so it doesn't contend for memory.

3. **Disk:** ~50 GB free for the working directory (12k synthesized WAVs
   dominate; the rest is checkpoints, eval audio, logs).

4. **This repo cloned to a working directory.** Everything else
   (system packages, `setup.sh`, config edits, pipeline run) is the
   agent's job.

### Agent prompt

Once the prerequisites are in place, paste something like this to your
coding agent:

> Run the `piper-distillation` pipeline to distill the voice
> `<your-teacher-voice-id>` from the TTS server at
> `<http://your-teacher-host:port>`. The deployed voice name should be
> `en_US-<yourvoice>-medium`. Install the final ONNX into
> `<absolute-path-to-wyoming-piper-data-dir>` and restart `piper.service`
> after. You can find the voice's reference clip path by inspecting the
> teacher's voice registry (for Qwen3-TTS, that's
> `~/.cache/qwen-tts/voices/<voice-id>/ref.wav`).
>
> Read `AGENTS.md` first — it's the execution reference. Install any
> missing system packages (cmake, ninja-build, espeak-ng, wget,
> python3-venv); bootstrap with `bash setup.sh`; update `run_config.json`
> and `SERVICE_MODELS` in `scripts/phase6_install.py` for the values
> above; then run the pipeline end-to-end via `scripts/orchestrate.py`.
> Phase 2 takes ~13 hours; the orchestrator survives shell exits and is
> resumable, so run it detached.
>
> Halt and surface to me if you hit any of the halt conditions listed in
> AGENTS.md (Phase 0 teacher consistency below threshold, Phase 3
> retained audio below 2 hours, training NaN/OOM/divergence, etc.).
> Notify me when the final `.onnx` is installed and the smoke test
> passes.

The agent does the bootstrap, fills in the config, kicks off the run,
and watches for halt conditions. You wake up ~18 hours later to a
deployed voice. (Or, you check in periodically — the orchestrator
writes `state/notifications.txt` for any milestones or alerts.)

## Status

Works for the use case it was built for (a single voice-cloning teacher
→ Piper VITS). It's been run end-to-end exactly once. Generalization is
partial:

- ✅ Works end-to-end for English voices via espeak-ng phonemization
- ✅ Resumable; each phase survives crashes and re-runs
- ⚠️ `run_config.json` schema isn't fully parameterized — install paths,
  systemd unit names, and a few GPU index assumptions are still hardcoded
  to the original deployment's layout. A future refactor pulls these out
  into CLI args + per-host config.
- ⚠️ Tested only against qwen-tts as teacher. Other OpenAI-compatible
  TTS hosts (XTTS, Coqui, ElevenLabs, etc.) should work with minimal
  edits to `scripts/phase2_synthesize.py` but aren't tested.

## What's in the repo

| Path | What |
|---|---|
| `README.md` | This file — audience: human |
| `AGENTS.md` | Dense execution reference — audience: coding agent (or human in agent mode) |
| `setup.sh` | One-shot bootstrap |
| `run_config.json` | Per-voice configuration (worked-example values as a template) |
| `scripts/` | 25 pipeline scripts (phases 0-6, orchestrator, watchdogs, status helpers) |
| `.gitignore` | Excludes all runtime artifacts |

Everything generated during a run (`phase1/`, `phase2/audio_raw/`,
`phase4*/checkpoints/`, `state/`, `logs/`, the venv, the `piper1-gpl`
clone) is gitignored. The repo is the **pipeline**, not any specific
run's output.

## License

The pipeline scripts here are MIT-licensed (or whatever you want — set a
LICENSE file before publishing). The Piper trainer they patch
([`OHF-Voice/piper1-gpl`](https://github.com/OHF-Voice/piper1-gpl)) is
GPL-3.0-or-later — the fork at
[`kasima/piper1-gpl`](https://github.com/kasima/piper1-gpl) inherits that.
Generated voice models inherit the license of the teacher (e.g.
qwen-tts voice clones inherit Qwen3-TTS's license terms; verify before
redistributing).
