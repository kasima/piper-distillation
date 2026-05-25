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

1. **Bootstrap.**
   ```
   bash setup.sh
   ```
   Creates a venv, clones a patched `piper1-gpl` fork
   ([`kasima/piper1-gpl`](https://github.com/kasima/piper1-gpl) branch
   `kasima/torch-2.6-compat`), downloads + cleans a Lessac warm-start
   checkpoint. ~10 minutes on first run.

2. **Have an OpenAI-compatible voice-cloning TTS running** with your
   target voice registered. The pipeline calls
   `POST /v1/audio/speech` for synthesis; configure the endpoint and
   voice ID in `run_config.json`. The committed values are from a worked
   example; replace `teacher.endpoint`, `teacher.voice_id`,
   `teacher.ref_path`, `teacher.ref_sha256`, and `voice_name` for your case.

3. **Run the pipeline.** Either chain it all via the orchestrator:
   ```
   nohup .venv/bin/python3 -u scripts/orchestrate.py >> logs/orchestrate.log 2>&1 < /dev/null &
   disown
   ```
   …or step through phases manually (each script is idempotent and
   resumable). Detailed phase-by-phase commands, halt conditions, and
   resume semantics are in [`AGENTS.md`](AGENTS.md).

4. **Read [`AGENTS.md`](AGENTS.md)** before doing anything non-trivial.
   It's the dense execution reference — phase commands, halt conditions,
   the Piper trainer patches that are required, the wyoming-piper voice-
   naming gotcha that bites if you skip Phase 6's `phase6_install.py`.
   Written for a coding agent (or a human in agent mode) operating the
   pipeline.

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
