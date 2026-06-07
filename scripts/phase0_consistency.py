"""Phase 0.3 — measure teacher consistency.

For each of the 60 probe sentences, we have 2 takes. Compute:
  - ECAPA cosine self-similarity per pair (mean is teacher voice consistency)
  - WavLM cosine self-similarity per pair
  - Reference centroid (mean ECAPA embedding across all 120 clips) -> .npy
  - Reference centroid (mean WavLM embedding across all 120 clips) -> .npy
  - F0 mean+range per clip, mean speaking rate (chars / duration)

Writes:
  state/phase0/consistency.json
  state/phase0/ecapa_centroid.npy
  state/phase0/wavlm_centroid.npy
  state/phase0/per_clip_metrics.csv
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WavLMForXVector

RUN = Path(__file__).resolve().parent.parent
import json as _piper_json
OUT = RUN / "output" / _piper_json.loads((RUN / __import__("os").environ.get("PIPER_DISTILL_CONFIG", "run_config.json")).read_text())["teacher"]["voice_id"]
CLIPS = OUT / "state/phase0/clips"
OUT_DIR = OUT / "state/phase0"
PROBE = OUT / "state/phase0/probe_sentences.txt"

DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
# vllm-aeon holds 44 GB on GPU 1, leaving ~4 GB free. ECAPA+WavLM together are ~500 MB.
# If init OOMs, fall back to CPU.


def load_wav(path: Path, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def main() -> None:
    sentences = [s.strip() for s in PROBE.read_text().splitlines() if s.strip()]
    n = len(sentences)
    print(f"loading models on device={DEVICE}")

    # ECAPA via speechbrain
    from speechbrain.inference.speaker import EncoderClassifier

    sb_device = DEVICE if DEVICE.startswith("cuda") else "cpu"
    try:
        ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(RUN / ".cache/speechbrain_ecapa"),
            run_opts={"device": sb_device},
        )
    except Exception as e:
        print(f"ECAPA on {sb_device} failed ({e}); falling back to CPU")
        sb_device = "cpu"
        ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(RUN / ".cache/speechbrain_ecapa"),
            run_opts={"device": "cpu"},
        )

    # WavLM via transformers
    wavlm_id = "microsoft/wavlm-base-plus-sv"
    wavlm_fe = AutoFeatureExtractor.from_pretrained(wavlm_id)
    try:
        wavlm = WavLMForXVector.from_pretrained(wavlm_id).to(DEVICE).eval()
        wavlm_device = DEVICE
    except Exception as e:
        print(f"WavLM on {DEVICE} failed ({e}); CPU")
        wavlm = WavLMForXVector.from_pretrained(wavlm_id).to("cpu").eval()
        wavlm_device = "cpu"
    print(f"models loaded (ecapa={sb_device}, wavlm={wavlm_device})")

    ecapa_embeds: dict[tuple[int, int], np.ndarray] = {}
    wavlm_embeds: dict[tuple[int, int], np.ndarray] = {}
    per_clip = []

    t0 = time.monotonic()
    for sid in range(n):
        for take in (1, 2):
            wav_path = CLIPS / ("first" if take == 1 else "second") / f"{sid:03d}.wav"
            audio16 = load_wav(wav_path, 16000)

            audio_t = torch.from_numpy(audio16).unsqueeze(0)
            with torch.no_grad():
                ec = ecapa.encode_batch(audio_t.to(sb_device)).squeeze().cpu().numpy()
            ecapa_embeds[(sid, take)] = ec

            inputs = wavlm_fe(audio16, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(wavlm_device) for k, v in inputs.items()}
            with torch.no_grad():
                out = wavlm(**inputs)
                wl = out.embeddings.squeeze().cpu().numpy()
            wavlm_embeds[(sid, take)] = wl

            duration_s = len(audio16) / 16000
            chars = len(sentences[sid])
            per_clip.append({
                "id": sid,
                "take": take,
                "duration_s": duration_s,
                "chars": chars,
                "rate_chars_per_s": chars / duration_s if duration_s > 0 else 0.0,
            })
        if (sid + 1) % 10 == 0:
            print(f"  embedded {(sid+1)*2}/{n*2} in {time.monotonic()-t0:.1f}s")

    # Self-similarities per sentence
    ec_sims = []
    wl_sims = []
    for sid in range(n):
        a = ecapa_embeds[(sid, 1)]
        b = ecapa_embeds[(sid, 2)]
        ec_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        ec_sims.append(ec_sim)
        a = wavlm_embeds[(sid, 1)]
        b = wavlm_embeds[(sid, 2)]
        wl_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        wl_sims.append(wl_sim)

    ec_centroid = np.mean(
        np.stack([e for e in ecapa_embeds.values()]), axis=0
    )
    wl_centroid = np.mean(
        np.stack([e for e in wavlm_embeds.values()]), axis=0
    )
    np.save(OUT_DIR / "ecapa_centroid.npy", ec_centroid)
    np.save(OUT_DIR / "wavlm_centroid.npy", wl_centroid)

    rates = np.array([c["rate_chars_per_s"] for c in per_clip])
    durations = np.array([c["duration_s"] for c in per_clip])

    summary = {
        "n_pairs": n,
        "ecapa_self_similarity": {
            "mean": float(np.mean(ec_sims)),
            "p50": float(np.median(ec_sims)),
            "p10": float(np.percentile(ec_sims, 10)),
            "p90": float(np.percentile(ec_sims, 90)),
            "min": float(np.min(ec_sims)),
            "max": float(np.max(ec_sims)),
        },
        "wavlm_self_similarity": {
            "mean": float(np.mean(wl_sims)),
            "p50": float(np.median(wl_sims)),
            "p10": float(np.percentile(wl_sims, 10)),
            "p90": float(np.percentile(wl_sims, 90)),
            "min": float(np.min(wl_sims)),
            "max": float(np.max(wl_sims)),
        },
        "speaking_rate_chars_per_s": {
            "mean": float(np.mean(rates)),
            "std": float(np.std(rates)),
            "p10": float(np.percentile(rates, 10)),
            "p90": float(np.percentile(rates, 90)),
        },
        "duration_s": {
            "mean": float(np.mean(durations)),
            "p10": float(np.percentile(durations, 10)),
            "p90": float(np.percentile(durations, 90)),
        },
        # Phase 3.4 thresholds: 75% of measured self-similarity baseline (rounded)
        "derived_phase3_thresholds": {
            "ecapa_min": round(float(np.mean(ec_sims)) * 0.75, 3),
            "wavlm_min": round(float(np.mean(wl_sims)) * 0.75, 3),
        },
    }
    with (OUT_DIR / "consistency.json").open("w") as f:
        json.dump(summary, f, indent=2)
    # Phase 3 reads the Phase 0 manifest at state/phase0/manifest.json (canonical
    # per-phase manifest name) for derived_phase3_thresholds + the centroids.
    # Write it here too so the consistency output and the manifest never drift.
    with (OUT_DIR / "manifest.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (OUT_DIR / "per_clip_metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_clip[0].keys()), delimiter="|")
        writer.writeheader()
        writer.writerows(per_clip)
    # Also dump pair similarities for inspection
    with (OUT_DIR / "pair_similarities.csv").open("w") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["id", "sentence", "ecapa", "wavlm"])
        for sid in range(n):
            writer.writerow(
                [sid, sentences[sid], f"{ec_sims[sid]:.4f}", f"{wl_sims[sid]:.4f}"]
            )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
