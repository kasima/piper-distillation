# piper-distillation

Take a custom voice from an OpenAI-compatible voice-cloning TTS server
(originally [qwen-tts](https://github.com/QwenLM/Qwen2-Audio) doing
reference-clip voice cloning) and produce a self-contained
[Piper](https://github.com/OHF-Voice/piper1-gpl) VITS `.onnx` model you can
deploy to a `wyoming-piper` host (or any ONNX TTS runtime that understands
Piper voice configs).

Useful when you want to keep using a particular voice but the original
serving stack is too heavy for real-time use. The teacher TTS is run
once over a curated text corpus; the resulting (text, audio) pairs train a
student Piper model that synthesizes the same voice at >40× faster than
real-time on CPU.

## What it produces

A `~30-60 MB` `.onnx` file plus a small `.onnx.json` config, e.g.
`en_US-akira-medium.onnx`. Drop the pair into a Piper data dir and the
voice is reachable via:

- **Wyoming protocol** (`wyoming-piper`, what Home Assistant talks to)
- **Piper CLI** (`piper --model en_US-akira-medium.onnx < input.txt`)
- **Anything else that loads ONNX VITS models** (e.g.
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx))

## When you'd use this

You have a voice that exists only as a clone in a reference-based TTS
(qwen-tts, XTTS, similar), and you want:

- Lower per-request latency than the source TTS (real-time conversational
  use)
- A static, deployable artifact (no GPU server dependency)
- A voice servable by the broader Piper ecosystem

You'd **not** use this if the voice already has a Piper model in the
upstream catalog ([rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)),
if you can re-record a real speaker (cleaner training data), or if your
voice needs are met by a generic neutral voice.

## How long it takes

End-to-end for one voice, on a single A6000-class GPU + a 4070 Ti for the
teacher: about **18 hours wall-clock** (Phase 2 corpus synthesis is the
dominant cost). Plus ~3 hours per additional training variant (full
dataset, low quality, etc.).

A worked example with full timing breakdown, metric tables, and the
failed experiments along the way is in
[`sysadmin/piper/journal/2026-05-25-takashii-distillation.md`](../sysadmin/piper/journal/2026-05-25-takashii-distillation.md).

## How to use it

For a human reading this for the first time:

1. **Look at [`sysadmin/piper/`](../sysadmin/piper/)** in this monorepo for
   the deployed Piper service this repo's output gets installed into —
   especially the journal entry above for the worked example, and
   [`FOLLOW_UPS.md`](../sysadmin/piper/FOLLOW_UPS.md) for known
   improvements and open issues.

2. **Read [`AGENTS.md`](AGENTS.md)** in this directory. It's the dense
   reference for actually running the pipeline — phase-by-phase commands,
   resume semantics, halt conditions, what each artifact directory means,
   where ONNX files end up. The intended reader is a coding agent
   executing the pipeline (or a human in agent mode).

3. **Bootstrap with `bash setup.sh`.** Creates a venv, clones the
   patched `piper1-gpl` fork ([`kasima/piper1-gpl`](https://github.com/kasima/piper1-gpl)
   on branch `kasima/torch-2.6-compat`), downloads + cleans the Lessac
   warm-start checkpoint. About 10 minutes on first run.

4. **Edit `run_config.json`** for the voice you want to distill. The
   committed values are from the Takashii run as a working template.

5. **Run the orchestrator** (or step through phases manually — both work).
   Commands are in `AGENTS.md`.

## How the pipeline works (conceptually)

```
Phase 0:   characterize the teacher (does it sound consistent enough to clone?)
Phase 1:   pick a balanced 12k-sentence text corpus
Phase 2:   synthesize all 12k sentences from the teacher (≈12 hours)
Phase 3:   filter out the teacher's bad takes (silence, clipping, mispronunciations)
Phase 4:   fine-tune a Piper VITS model on the filtered (text, audio) pairs
Phase 5:   pick the best checkpoint by ECAPA + WavLM + Whisper-WER composite
Phase 6:   export to ONNX, smoke-test, install to the live Piper service
```

Each phase writes a manifest under `state/phase<N>/manifest.json`. The
orchestrator chains them and skips phases whose manifest is already
present, so the whole thing is resumable after crashes.

## Status

Production-ready for the use case it was designed for (a single
qwen-tts custom voice → Piper). Generalization is partial:

- ✅ Works end-to-end for English voices via espeak-ng phonemization
- ✅ Resumable; each phase survives `pkill -9` and re-runs
- ⚠️ The `run_config.json` schema isn't fully parameterized — paths and
  GPU indices are hardcoded to bernard's layout. A future refactor would
  pull these out into CLI args + per-host config. See
  [`sysadmin/piper/FOLLOW_UPS.md`](../sysadmin/piper/FOLLOW_UPS.md)
  "toolkit-repo extraction" for the plan.
- ⚠️ Tested only against qwen-tts as teacher. Other OpenAI-compatible
  TTS hosts (XTTS, Coqui, ElevenLabs, etc.) should work with minimal
  edits to `scripts/phase2_synthesize.py` but aren't tested.

## What's in the repo

| Path | What |
|---|---|
| `README.md` | This file (audience: human) |
| `AGENTS.md` | The dense execution reference (audience: coding agent or human in agent mode) |
| `setup.sh` | One-shot bootstrap |
| `run_config.json` | Per-voice configuration (Takashii values, as template) |
| `scripts/` | 25 pipeline scripts (phases 0-6, orchestrator, watchdogs, status) |
| `.gitignore` | Excludes all runtime artifacts |

Everything generated during a run (`phase1/`, `phase2/audio_raw/`,
`phase4*/checkpoints/`, `state/`, `logs/`, the venv, the piper1-gpl
clone) is gitignored. The repo is the **pipeline**, not any specific
run's output.

## License

The pipeline scripts are owned by you (kasima). The Piper trainer they
patch ([`OHF-Voice/piper1-gpl`](https://github.com/OHF-Voice/piper1-gpl))
is GPL-3.0-or-later — the fork at
[`kasima/piper1-gpl`](https://github.com/kasima/piper1-gpl) inherits that.
Generated voice models inherit the license of the teacher (qwen-tts
voice clones inherit Qwen3-TTS's license terms; check before distributing).
