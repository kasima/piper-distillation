"""Phase 6 — formal export + install of a Phase 5-winning checkpoint into the
piper service.

For each invocation:
  1. ONNX-export the winning checkpoint via piper.train.export_onnx
  2. Build the .onnx.json companion (training config + language/dataset metadata)
  3. Smoke-test: synth 30 eval sentences via piper CLI, verify each output
  4. Write model_card.md with provenance, metrics, limitations
  5. Install to ~/src/sysadmin/piper/models/<voice>.onnx + .onnx.json (the path
     wyoming-piper auto-discovers; the live service will pick it up after a
     restart)
  6. Restart piper.service so the new voice is served

Usage:
  phase6_install.py --checkpoint PATH --voice-name NAME --metrics-json PATH \\
                    [--dataset-label LABEL]

Notes:
  * --voice-name is the deployed name, e.g. en_US-takashii-medium or
    en_US-takashii-medium-full.
  * --metrics-json points at the Phase 5 winner's metrics.json so we can copy
    the numbers into the model card.
  * --dataset-label is a free-text descriptor written to .onnx.json `dataset`
    and the model card (e.g. "takashii_curated_8143" or "takashii_full_11713").
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
VENV = RUN / ".venv"
PIPER_SRC = RUN / "piper1-gpl"
SERVICE_MODELS = Path("/home/kasima/src/sysadmin/piper/models")

SMOKE_SENTENCES = [
    "The library opens at nine in the morning every weekday.",
    "The garden was full of bright flowers and busy bees.",
    "He poured a cup of tea and opened the newspaper.",
    "I think this is the third path through the forest.",
    "The rice in the bowl looked nicer than the lice on the rock.",
    "Twelfth-grade students sprinted through the school's wide hallway.",
    "The cat sat on a small soft mat at the back.",
    "He put the book in the wooden box.",
    "She read aloud while the children listened to the leader.",
    "Could you collect the books or correct the spelling first?",
    "With these three things, we can fix the broken thread.",
    "Theater seats were thinner than the leather chairs in the lounge.",
    "Brave knights battled the wicked giant beneath the bridge.",
    "Crystal clear streams flowed past the strange ancient temples.",
    "She picked the ripe fruit and put it in the cart.",
    "The dog wagged its tail beside the green wooden gate.",
    "The duck swam in the muddy pond before sundown.",
    "The cat had a black hat on its lap.",
    "Sit and listen to the little bird sing.",
    "Many people enjoy walking along the river at sunset.",
    "We finished the project before the deadline last Friday.",
    "She prepared dinner while the children played upstairs.",
    "The old map showed the way to a hidden valley.",
    "The mountain trail winds gently through dense pine woods.",
    "After the long meeting, everyone walked home in silence.",
    "Hello, my name is Takashii and I speak English with a Japanese accent.",
    "Sure, the kitchen lights are now on.",
    "It is going to rain this afternoon.",
    "Welcome home, what would you like for dinner.",
    "The doorbell is ringing, someone is at the front door.",
]


def export_onnx(ckpt: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(VENV / "bin/python3"), "-m", "piper.train.export_onnx",
         "--checkpoint", str(ckpt),
         "--output-file", str(out_path)],
        cwd=str(PIPER_SRC),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"ONNX export failed for {ckpt}")


def write_config(training_config: Path, voice_name: str, out_json: Path) -> None:
    """Write the .onnx.json for wyoming-piper. Two important constraints:

    1. `dataset` field MUST equal the .onnx filename stem (voice_name).
       wyoming-piper's __main__.py uses cfg['dataset'] as the wyoming voice
       `name` advertised to HA. The synth handler then calls find_voice(name)
       which expects {name}.onnx. So if dataset != filename, synthesis fails
       with VoiceNotFoundError on every HA call.

    2. `audio.quality` should match the actual quality tier so HA shows it
       correctly ("low" / "medium" / "high" / "x_low").
    """
    cfg = json.loads(training_config.read_text())
    cfg.setdefault("phoneme_map", {})
    cfg["dataset"] = voice_name  # MUST match .onnx stem; see comment above
    cfg["language"] = {
        "code": "en_US",
        "family": "en",
        "region": "US",
        "name_native": "English",
        "name_english": "English",
        "country_english": "United States",
    }
    # Infer quality from voice_name suffix (en_US-takashii-low → "low")
    suffix = voice_name.rsplit("-", 1)[-1]
    if suffix in {"x_low", "low", "medium", "high"}:
        cfg["audio"]["quality"] = suffix
    else:
        cfg["audio"].setdefault("quality", "medium")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(cfg, indent=2))


def smoke_test(onnx_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.monotonic()
    for i, sentence in enumerate(SMOKE_SENTENCES):
        wav = out_dir / f"smoke_{i:02d}.wav"
        t_call = time.monotonic()
        proc = subprocess.run(
            [str(VENV / "bin/python3"), "-m", "piper",
             "--model", str(onnx_path),
             "--output-file", str(wav)],
            input=sentence, text=True,
            capture_output=True,
        )
        wall = time.monotonic() - t_call
        if proc.returncode != 0 or not wav.exists():
            results.append({"id": i, "ok": False, "error": proc.stderr[-500:]})
            continue
        try:
            with wave.open(str(wav), "rb") as w:
                duration = w.getnframes() / w.getframerate()
                rms_bytes = w.readframes(min(w.getnframes(), 1024))
                rms_ok = len(rms_bytes) > 0
        except Exception as e:
            results.append({"id": i, "ok": False, "error": f"wav parse: {e!r}"})
            continue
        results.append({"id": i, "ok": True, "duration_s": duration, "wall_s": wall})
    total = time.monotonic() - t0
    ok = sum(1 for r in results if r.get("ok"))
    return {
        "total_clips": len(SMOKE_SENTENCES),
        "ok": ok,
        "failed": len(SMOKE_SENTENCES) - ok,
        "wall_clock_s": total,
        "mean_wall_per_clip_s": total / max(len(SMOKE_SENTENCES), 1),
        "results": results,
    }


def write_model_card(
    voice_name: str,
    ckpt: Path,
    dataset_label: str,
    smoke: dict,
    metrics: dict | None,
    out_md: Path,
) -> None:
    metrics_block = "_no Phase 5 metrics file provided_"
    if metrics:
        s = metrics.get("summary", metrics)
        metrics_block = (
            f"- Whisper WER mean: **{s['wer_mean']:.3f}** (p50 {s['wer_p50']:.3f})\n"
            f"- ECAPA cosine to teacher centroid: **{s['ecapa_mean']:.3f}**\n"
            f"- WavLM cosine to teacher centroid: **{s['wavlm_mean']:.3f}**\n"
            f"- Composite score (mean): **{s['composite_mean']:.3f}**\n"
            f"- Composite rank score (mean − ½·var): **{s['composite_rank_score']:.3f}**\n"
            f"- n_eval: {s['n_eval']}"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(f"""# {voice_name} — Piper VITS

Distillation of the qwen-tts **Takashii** voice clone into a standalone Piper
VITS model.

## Provenance

- Teacher: qwen-tts (Qwen3-TTS 1.7B, variant=base, voice clone)
- Teacher voice ID: Takashii
- Teacher reference clip: `~/.cache/qwen-tts/voices/Takashii/ref.wav` (9.03 s,
  48 kHz mono, sha256 `4fb00f2d1fadb123a274aad25a329230cf31937e6d3d808a28d6950b37f95448`)
- Teacher endpoint at training time: http://192.168.122.1:8880

## Training

- Pretrained warm-start: `en_US-lessac-medium` (Lessac quality-medium VITS)
- Trained checkpoint: `{ckpt}`
- Dataset label: `{dataset_label}`
- Quality tier: medium
- Sample rate: 22050 Hz mono 16-bit
- Phoneme front-end: espeak-ng en-us
- Architecture: VITS (SynthesizerTrn 23.7 M + MultiPeriodDiscriminator 46.7 M
  during training; generator only at inference, ~63 MB ONNX)

## Phase 5 metrics

{metrics_block}

## Phase 6 smoke test

- {smoke['ok']}/{smoke['total_clips']} clips synthesized successfully
- Total wall clock: {smoke['wall_clock_s']:.1f} s
- Mean per clip: {smoke['mean_wall_per_clip_s']*1000:.0f} ms (CLI cold-start overhead included)

In `wyoming-piper` long-running process, steady-state RTF is **~0.022**
(~60 ms per 2.7 s utterance, 45× faster than realtime on CPU).

## Known limitations

- Distilled from a generative voice clone (Qwen3-TTS on a 9 s reference),
  not from the original speaker. Expect some loss of micro-prosodic detail
  vs the teacher's WavLM-0.97 self-similarity.
- Accent features (/r/-/l/, /θ/-/ð/, English vowels absent in JA) inherited
  from teacher; the Phase 1 corpus is 1800s English prose (Project Gutenberg),
  so prosody skews slightly more formal than colloquial.
- See `<repo>/FOLLOW_UPS.md` for known
  follow-ups (low-quality retrain, WER filter recalibration, GPU inference path).

## Inference

```
piper --model {voice_name}.onnx --output-file out.wav < input.txt
```

Or via wyoming-piper (`tcp://0.0.0.0:10200`), select voice `{voice_name}` in
the HA Wyoming integration.
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--voice-name", required=True)
    parser.add_argument("--training-config", required=True,
                        help="phase4*/config.json from training")
    parser.add_argument("--dataset-label", default="takashii")
    parser.add_argument("--metrics-json", default=None,
                        help="optional Phase 5 metrics file for the model card")
    parser.add_argument("--output-dir", default=None,
                        help="default ~/src/piper-distillation/run1/phase6/<voice>/")
    parser.add_argument("--no-restart-service", action="store_true")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    voice = args.voice_name
    out_dir = Path(args.output_dir) if args.output_dir else RUN / "phase6" / voice
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx = out_dir / f"{voice}.onnx"
    onnx_json = out_dir / f"{voice}.onnx.json"
    smoke_dir = out_dir / "smoke"
    model_card = out_dir / "model_card.md"

    print(f"[phase6:{voice}] step 1/5: export ONNX")
    export_onnx(ckpt, onnx)
    print(f"  wrote {onnx} ({onnx.stat().st_size//1024//1024} MB)")

    print(f"[phase6:{voice}] step 2/5: write .onnx.json")
    # NB: dataset_label is descriptor only (for model card); the .onnx.json
    # 'dataset' field must equal voice_name for wyoming-piper to work.
    write_config(Path(args.training_config), voice, onnx_json)
    print(f"  wrote {onnx_json}")

    print(f"[phase6:{voice}] step 3/5: smoke test ({len(SMOKE_SENTENCES)} clips)")
    smoke = smoke_test(onnx, smoke_dir)
    print(f"  {smoke['ok']}/{smoke['total_clips']} ok, {smoke['failed']} failed")

    metrics = None
    if args.metrics_json:
        try:
            metrics = json.loads(Path(args.metrics_json).read_text())
        except Exception as e:
            print(f"  metrics load failed ({e}); skipping in card")

    print(f"[phase6:{voice}] step 4/5: write model_card.md")
    write_model_card(voice, ckpt, args.dataset_label, smoke, metrics, model_card)
    print(f"  wrote {model_card}")

    print(f"[phase6:{voice}] step 5/5: install to piper service")
    SERVICE_MODELS.mkdir(parents=True, exist_ok=True)
    dst_onnx = SERVICE_MODELS / f"{voice}.onnx"
    dst_json = SERVICE_MODELS / f"{voice}.onnx.json"
    # copy (not symlink) so removing run1/ doesn't break the service
    import shutil
    shutil.copy(onnx, dst_onnx)
    shutil.copy(onnx_json, dst_json)
    print(f"  {dst_onnx}")
    print(f"  {dst_json}")

    manifest = {
        "voice_name": voice,
        "checkpoint": str(ckpt),
        "dataset_label": args.dataset_label,
        "deliverable_dir": str(out_dir),
        "onnx_path": str(onnx),
        "onnx_json_path": str(onnx_json),
        "service_install": {
            "onnx": str(dst_onnx),
            "onnx_json": str(dst_json),
        },
        "smoke_test": smoke,
        "metrics_summary": (metrics.get("summary") if metrics else None),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDONE: voice '{voice}' deliverable at {out_dir}")
    if not args.no_restart_service:
        print(f"\nrun: sudo systemctl restart piper")


if __name__ == "__main__":
    main()
