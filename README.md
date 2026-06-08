# piper-distillation

Distill a running OpenAI-compatible TTS server into a self-contained
[Piper](https://github.com/OHF-Voice/piper1-gpl) VITS `.onnx` model you can
deploy to a `wyoming-piper` host for Home Assistant or any ONNX TTS runtime 
that understands Piper voice configs.

The original motivating use case was a custom voice clone in
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — a heavy server-side
model — that we wanted to keep using but at Piper-speed and as a static
artifact. The pipeline is teacher-agnostic: anything that exposes
`POST /v1/audio/speech` works.

Useful when you want a particular voice that exists only on a heavy
serving stack and you'd rather have it as a small fast standalone model.
The teacher is run once over a curated text corpus; the resulting (text,
audio) pairs train a student Piper model that synthesizes at >40× faster
than real-time on CPU.

## What it produces

A ~30-60 MB `.onnx` file plus a small `.onnx.json` config, e.g.
`en_US-<yourvoice>-medium.onnx`. Drop the pair into a Piper data
directory and the voice is reachable via:

- **Wyoming protocol** (`wyoming-piper`, what Home Assistant talks to)
- **Piper CLI** (`piper --model en_US-<yourvoice>-medium.onnx < input.txt`)
- **Anything else that loads ONNX VITS models** (e.g.
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx))

## When you'd use this

You have a voice on a teacher TTS that's too heavy or stateful for your
real serving needs, and you want:

- Lower per-request latency than the teacher
- A static, deployable artifact (no GPU server dependency at inference)
- A voice servable by the broader Piper ecosystem

## How the pipeline works (conceptually)

```
Phase 0:   characterize the teacher (does it sound consistent enough to distill?)
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

Approximate wall-clock for one voice (one training run). The two biggest
phases (2 and 4) are the dominant cost.

| Phase | What | Wall-clock |
|---|---|---|
| 0 | Probe teacher consistency | ~15 min |
| 1 | Greedy-select 12k-sentence corpus | ~5 min |
| 2 | Synthesize 12k clips from the teacher | **~13 h** |
| 3 | QA filter (WER + speaker verify + audio sanity) | ~1.5 h |
| 4 | Fine-tune Piper VITS (40k steps from Lessac) | **~3 h** |
| 5 | Eval top-3 checkpoints, pick winner | ~30 min |
| 6 | ONNX export + smoke test + install | ~5 min |
| | **Total per voice (single training run)** | **~18 h** |
| | Per additional training variant (same data) | +3.5 h |

Phase 2 is the long pole and depends on the teacher's per-request
latency; the figure above is for a ~1.7 B-parameter TTS at concurrency 2.
A faster teacher cuts this proportionally.

Phase 4 cap of 40k steps is conservative; Piper VITS from a Lessac
warm-start often converges by 25-30k. Smaller architectures (`low`,
`x_low`) train faster but produce noticeably different audio — see
notes in `AGENTS.md` before assuming "low = faster same voice".

## How to use it

This repo is **meant to be operated fully autonomously by a coding agent**
end-to-end — from setup through training to deployed voice. Your job as
the human is to satisfy the prerequisites, then hand the agent a prompt.
The agent shouldn't need to ping you for intermediate decisions during
the ~18-hour run.

### Prerequisites

1. **A running OpenAI-compatible TTS** with your target voice
   accessible. The pipeline calls `POST /v1/audio/speech` and expects a
   `voice` parameter that picks the voice. Any teacher that speaks this
   contract should work with minor edits to
   `scripts/phase2_synthesize.py`. Note the endpoint URL and the voice
   ID.

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

Once the prerequisites are in place, paste this verbatim to your
coding agent:

> Run the `piper-distillation` pipeline end-to-end. Before doing anything,
> ask me for these three values up front and don't proceed until I
> answer:
>
> 1. **`voice_id`** — the voice ID registered with the teacher TTS (e.g.,
>    the `voice` parameter that the teacher's `/v1/audio/speech` endpoint
>    accepts)
> 2. **`teacher_url`** — the teacher TTS server URL (the base URL whose
>    `/v1/audio/speech` endpoint you'll call)
> 3. **`onnx_dest_dir`** — the absolute path to the `wyoming-piper` data
>    directory where Phase 6 should install the final `.onnx` + `.onnx.json`
>
> Once you have them: read `AGENTS.md` (the execution reference); install
> any missing system packages (cmake, ninja-build, espeak-ng, wget,
> python3-venv); bootstrap with `bash setup.sh`; update `run_config.json`
> with the `voice_id` and `teacher_url` and set `voice_name` to
> `en_US-<voice_id_lowercased>-medium` (unless I tell you a different
> name); set `SERVICE_MODELS` in `scripts/phase6_install.py` to
> `onnx_dest_dir`; then run the pipeline end-to-end via
> `scripts/orchestrate.py`. Phase 2 takes ~13 hours; the orchestrator
> survives shell exits and is resumable, so run it detached.
>
> Halt and surface to me **only** if you hit one of the halt conditions
> listed in AGENTS.md (Phase 0 teacher consistency below threshold,
> Phase 3 retained audio below 2 hours, training NaN/OOM/divergence,
> etc.). Everything else: log it to
> `output/<voice_id>/state/notifications.txt` and keep going. Notify me
> when the final `.onnx` is installed and the smoke test passes.

The agent does the bootstrap, fills in the config, kicks off the run,
and watches for halt conditions. You wake up ~18 hours later to a
deployed voice. (Or, you check in periodically — the orchestrator
writes `state/notifications.txt` for any milestones or alerts. Just ask
your agent.)

## Status

Used in production for three distilled voices (Takashii, Computer, Adjutant).

- ✅ Works end-to-end for English voices via espeak-ng phonemization
- ✅ Resumable; each phase survives crashes and re-runs
- ✅ **Multi-voice**: per-voice configs under `configs/<name>.json`, switched
  with `./select-voice.sh <name>` (or `PIPER_DISTILL_CONFIG=<path>`). Run state
  is keyed by `teacher.voice_id`, so voices never collide. `voice_name`, the
  training runs (`runs`), the Phase 3 WER threshold (`filters.wer_max`), and
  the install dir (`host.onnx_dest_dir`) are all config-driven — no per-voice
  code edits. See AGENTS.md "Multi-voice".
- ⚠️ A few host assumptions remain (GPU index assumptions, systemd-based train
  launch). Fine on the original two-GPU host; portability beyond it is the
  remaining future-work item (see AGENTS.md "Future-work toolkit extraction").
- ⚠️ Tested only against one teacher TTS (qwen-tts). Other OpenAI-compatible
  TTS hosts should work with minor edits to `scripts/phase2_synthesize.py`
  but aren't tested.

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
Generated voice models may inherit the license of the teacher's output;
verify before redistributing.
