"""Phase 5 — final eval, MVP-first metric stack.

Usage:
  phase5_eval.py --checkpoint PATH [--checkpoint PATH ...]
                 [--probe-mid]   # use 5-sentence probe set only (for mid-training probes)

For each checkpoint, synthesize the eval set, then compute:
  MVP: ECAPA cosine to teacher centroid, WavLM cosine to teacher centroid,
       Whisper WER (vs eval text), UTMOS (if available; otherwise skipped).

Composite weights: ECAPA 0.35, WavLM 0.25, WER (inverted, 1-WER) 0.30, UTMOS 0.10.

Picks the winning checkpoint by mean composite minus 0.5 * variance.

If MVP composite is in [0.55, 0.75], expand stack with formant/F0/etc.
(not implemented in this script — emit notice for follow-up).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import whisper

import os
RUN = Path(__file__).resolve().parent.parent
import json as _piper_json
OUT = RUN / "output" / _piper_json.loads((RUN / __import__("os").environ.get("PIPER_DISTILL_CONFIG", "run_config.json")).read_text())["teacher"]["voice_id"]
PHASE0 = OUT / "state/phase0"
# Variant: "" (default = Run A curated) or "full" (Run B with WER drops recovered)
_VARIANT = os.environ.get("PIPER_DISTILL_EVAL_VARIANT", "")
_SUFFIX = f"-{_VARIANT}" if _VARIANT else ""
EVAL_META = OUT / f"phase4{_SUFFIX}/eval_metadata.csv"
PHASE3_AUDIO = OUT / f"phase3{_SUFFIX}/audio"  # teacher reference WAVs
STATE = OUT / f"state/phase5{_SUFFIX}"

ECAPA_CENTROID = np.load(PHASE0 / "ecapa_centroid.npy")
WAVLM_CENTROID = np.load(PHASE0 / "wavlm_centroid.npy")

DEVICE = "cuda:1"  # Phase 5 runs after Phase 4 with vllm-aeon stopped on GPU 1.
# A6000 has 48 GB free at this point — plenty for whisper-large-v3 + ECAPA + WavLM.


def synth_piper(checkpoint: Path, sentences: list[tuple[str, str]], out_dir: Path) -> dict[str, Path]:
    """Run piper inference on each sentence, write WAVs to out_dir/<id>.wav.

    Returns dict[id, wav_path].
    """
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True)
    # Export to ONNX for inference speed; reuse if already exported.
    onnx_path = out_dir.parent / "model.onnx"
    onnx_json = out_dir.parent / "model.onnx.json"
    if not onnx_path.exists():
        export_log = subprocess.run(
            [str(RUN / ".venv/bin/python3"), "-m", "piper.train.export_onnx",
             "--checkpoint", str(checkpoint),
             "--output-file", str(onnx_path)],
            cwd=str(RUN / "piper1-gpl"),
            capture_output=True, text=True,
        )
        if export_log.returncode != 0:
            raise RuntimeError(f"ONNX export failed: {export_log.stderr}")
    # Piper CLI needs the .onnx.json config alongside the .onnx — copy from training config
    if not onnx_json.exists():
        shutil.copy(OUT / "phase4/config.json", onnx_json)

    out_map: dict[str, Path] = {}
    for sid, sentence in sentences:
        wav = out_dir / f"{sid}.wav"
        if wav.exists():
            out_map[sid] = wav
            continue
        # piper CLI
        proc = subprocess.run(
            [str(RUN / ".venv/bin/python3"), "-m", "piper",
             "--model", str(onnx_path),
             "--output-file", str(wav)],
            input=sentence, text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            print(f"piper failed for {sid}: {proc.stderr}", file=sys.stderr)
            continue
        out_map[sid] = wav
    return out_map


def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    audio, native_sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if native_sr != sr:
        audio = librosa.resample(audio, orig_sr=native_sr, target_sr=sr)
    return audio


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def normalize_for_wer(s: str) -> str:
    import re
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[-/]", " ", s)
    s = re.sub(r"[^a-z' ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def wer(ref: str, hyp: str) -> float:
    r = normalize_for_wer(ref).split()
    h = normalize_for_wer(hyp).split()
    if not r:
        return 0.0
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[len(r)][len(h)] / len(r)


def evaluate(checkpoint: Path, sentences: list[tuple[str, str]]) -> dict:
    print(f"=== evaluating {checkpoint.name} ===")
    work = STATE / checkpoint.stem
    work.mkdir(parents=True, exist_ok=True)
    synth_dir = work / "wavs"
    out_map = synth_piper(checkpoint, sentences, synth_dir)

    whisper_model = whisper.load_model("large-v3", device=DEVICE)
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import AutoFeatureExtractor, WavLMForXVector
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(RUN / ".cache/speechbrain_ecapa"),
        run_opts={"device": DEVICE},
    )
    wavlm_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    wavlm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(DEVICE).eval()

    rows = []
    for sid, sentence in sentences:
        wav = out_map.get(sid)
        if wav is None or not wav.exists():
            continue
        audio = load_audio(wav, 16000)
        # WER
        try:
            r = whisper_model.transcribe(audio, language="en", fp16=DEVICE.startswith("cuda"))
            hyp = r["text"]
        except Exception as e:
            hyp = ""
        w = wer(sentence, hyp)
        # ECAPA + WavLM
        audio_t = torch.from_numpy(audio).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            ec = ecapa.encode_batch(audio_t).squeeze().cpu().numpy()
            inp = wavlm_fe(audio, sampling_rate=16000, return_tensors="pt")
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            wl = wavlm(**inp).embeddings.squeeze().cpu().numpy()
        rows.append({
            "id": sid, "sentence": sentence,
            "wer": w,
            "ecapa": cos(ec, ECAPA_CENTROID),
            "wavlm": cos(wl, WAVLM_CENTROID),
        })
    if not rows:
        return {"checkpoint": str(checkpoint), "error": "no eval clips produced"}

    wer_arr = np.array([r["wer"] for r in rows])
    ec_arr = np.array([r["ecapa"] for r in rows])
    wl_arr = np.array([r["wavlm"] for r in rows])

    composite = (
        0.35 * ec_arr +
        0.25 * wl_arr +
        0.30 * (1 - np.clip(wer_arr, 0, 1)) +
        0.10 * 0.6  # UTMOS placeholder until utmos pkg installed
    )

    summary = {
        "checkpoint": str(checkpoint),
        "n_eval": len(rows),
        "wer_mean": float(wer_arr.mean()),
        "wer_p50": float(np.median(wer_arr)),
        "ecapa_mean": float(ec_arr.mean()),
        "ecapa_p50": float(np.median(ec_arr)),
        "wavlm_mean": float(wl_arr.mean()),
        "wavlm_p50": float(np.median(wl_arr)),
        "composite_mean": float(composite.mean()),
        "composite_var": float(composite.var()),
        "composite_rank_score": float(composite.mean() - 0.5 * composite.var()),
        "in_ambiguous_band_055_075": bool(0.55 < composite.mean() < 0.75),
    }
    with (work / "metrics.json").open("w") as f:
        json.dump({"summary": summary, "per_clip": rows}, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="Repeatable; one or more checkpoint paths to evaluate")
    parser.add_argument("--probe-mid", action="store_true",
                        help="Use 5-sentence probe set instead of full eval (for mid-training probes)")
    args = parser.parse_args()

    STATE.mkdir(parents=True, exist_ok=True)
    if args.probe_mid:
        # Pick 5 fixed sentences from the eval set
        sentences = []
        with EVAL_META.open() as f:
            for i, row in enumerate(csv.reader(f, delimiter="|")):
                if i >= 5:
                    break
                sentences.append((row[0], row[1]))
    else:
        sentences = []
        with EVAL_META.open() as f:
            for row in csv.reader(f, delimiter="|"):
                sentences.append((row[0], row[1]))
    print(f"eval set: {len(sentences)} sentences")

    results = []
    for ckpt in args.checkpoint:
        results.append(evaluate(Path(ckpt), sentences))

    if results:
        # Pick winner by composite_rank_score (mean - 0.5*variance)
        best = max(results, key=lambda r: r.get("composite_rank_score", -1e9))
        manifest = {
            "phase": "phase5",
            "candidates_evaluated": len(results),
            "winner": best["checkpoint"],
            "winner_summary": best,
            "all_candidates": results,
        }
        (STATE / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\nWINNER: {best['checkpoint']}")
        print(f"  composite mean: {best['composite_mean']:.3f}")
        print(f"  composite rank score: {best['composite_rank_score']:.3f}")
        if best["in_ambiguous_band_055_075"]:
            print(
                "  composite in 0.55-0.75 ambiguous band — recommend expanding metric stack "
                "(install parselmouth, utmos, run formant/pitch/epenthesis analysis)."
            )


if __name__ == "__main__":
    main()
