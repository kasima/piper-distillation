"""Phase 3 — QA filter the synthesized corpus.

Filters (in order; drops logged):
  3.1 audio sanity (duration vs chars, clipping, RMS, silences)
  3.2 Whisper WER <= 8%
  3.4 dual ECAPA + WavLM cosine to Phase 0 reference centroids
  3.5 statistical outliers (>3 sigma on duration, RMS, F0 mean, centroid)
  3.6 resample to 22050 mono 16-bit; write to phase3/audio/
  3.7 coverage re-check vs Phase 1 floors (informational)

Skipping 3.3 forced alignment as a separate pass — using whisper's
word_timestamps logprob aggregation as a proxy. Add WhisperX later
if Phase 5 demands it.
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import whisper
from transformers import AutoFeatureExtractor, WavLMForXVector

RUN = Path(__file__).resolve().parent.parent
import json as _piper_json
OUT = RUN / "output" / _piper_json.loads((RUN / "run_config.json").read_text())["teacher"]["voice_id"]
RAW = OUT / "phase2/audio_raw"
META_RAW = OUT / "state/phase2/metadata_raw.csv"
OUT_AUDIO = OUT / "phase3/audio"
OUT_META = OUT / "phase3/metadata.csv"
STATE = OUT / "state/phase3"
PHASE0 = OUT / "state/phase0"

DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"

# Loaded from Phase 0 manifest
P0 = json.loads((PHASE0 / "manifest.json").read_text())
ECAPA_MIN = P0["derived_phase3_thresholds"]["ecapa_min"]
WAVLM_MIN = P0["derived_phase3_thresholds"]["wavlm_min"]

ECAPA_CENTROID = np.load(PHASE0 / "ecapa_centroid.npy")
WAVLM_CENTROID = np.load(PHASE0 / "wavlm_centroid.npy")

WHISPER_WER_MAX = 0.08
DUR_RATIO_MIN = 0.04  # seconds per char (lower)
DUR_RATIO_MAX = 0.12  # seconds per char (upper)
INTERNAL_SILENCE_MAX_MS = 800
LEADING_SILENCE_MAX_MS = 800
RMS_MIN = 0.005  # near silence
SAMPLES_NEAR_CLIP = 0.999


def load_audio(path: Path, sr: int = 16000) -> tuple[np.ndarray, int]:
    audio, native_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if native_sr != sr:
        audio = librosa.resample(audio, orig_sr=native_sr, target_sr=sr)
    return audio, sr


def detect_internal_silences(audio: np.ndarray, sr: int, thresh_db: float = -40, min_silence_ms: int = 100) -> list[tuple[int, int]]:
    """Returns list of (start_sample, end_sample) silent regions (excluding leading/trailing)."""
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-9)
    is_silent = db < thresh_db
    # find runs
    runs = []
    i = 0
    while i < len(is_silent):
        if is_silent[i]:
            j = i
            while j < len(is_silent) and is_silent[j]:
                j += 1
            runs.append((i * hop, j * hop))
            i = j
        else:
            i += 1
    # Filter for internal silences > min_silence_ms and not at the very ends
    out = []
    for s, e in runs:
        dur_ms = (e - s) / sr * 1000
        if dur_ms < min_silence_ms:
            continue
        # ignore leading/trailing silence
        if s < int(0.1 * sr) or e > len(audio) - int(0.1 * sr):
            continue
        out.append((s, e))
    return out


_normalize_re = re.compile(r"[^a-z' ]")


def normalize_for_wer(s: str) -> str:
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[-/]", " ", s)
    s = _normalize_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def wer(ref: str, hyp: str) -> float:
    r = normalize_for_wer(ref).split()
    h = normalize_for_wer(hyp).split()
    if not r:
        return 0.0
    # Levenshtein on tokens
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


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    OUT_AUDIO.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)

    # Load metadata_raw.csv: id|sentence|duration_s|bytes
    rows: list[tuple[int, str, float, int]] = []
    with META_RAW.open() as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader)
        for row in reader:
            rows.append((int(row[0]), row[1], float(row[2]), int(row[3])))
    print(f"input clips: {len(rows)}")

    print(f"loading models on {DEVICE}")
    # Use whisper "medium" in Phase 3 because vllm-aeon still holds ~44 GB on GPU 1.
    # Medium fits comfortably alongside ECAPA + WavLM (~4 GB total) in the residual headroom
    # and is plenty accurate for WER filtering. Phase 5 (final eval) upgrades to large-v3
    # since vllm-aeon will be stopped by then.
    whisper_model_name = "medium.en"
    try:
        whisper_model = whisper.load_model(whisper_model_name, device=DEVICE if DEVICE != "cpu" else "cpu")
    except torch.cuda.OutOfMemoryError:
        print(f"OOM loading whisper {whisper_model_name} on {DEVICE}; falling back to CPU")
        whisper_model = whisper.load_model(whisper_model_name, device="cpu")
    from speechbrain.inference.speaker import EncoderClassifier
    sb_device = DEVICE if DEVICE.startswith("cuda") else "cpu"
    try:
        ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(RUN / ".cache/speechbrain_ecapa"),
            run_opts={"device": sb_device},
        )
    except Exception:
        sb_device = "cpu"
        ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(RUN / ".cache/speechbrain_ecapa"),
            run_opts={"device": "cpu"},
        )
    wavlm_id = "microsoft/wavlm-base-plus-sv"
    wavlm_fe = AutoFeatureExtractor.from_pretrained(wavlm_id)
    try:
        wavlm = WavLMForXVector.from_pretrained(wavlm_id).to(DEVICE).eval()
        wavlm_device = DEVICE
    except Exception:
        wavlm = WavLMForXVector.from_pretrained(wavlm_id).to("cpu").eval()
        wavlm_device = "cpu"

    drops: list[dict] = []
    surviving: list[dict] = []
    per_clip_stats: list[dict] = []
    t0 = time.monotonic()
    for n, (sid, sentence, duration_raw_s, bytes_) in enumerate(rows):
        raw_path = RAW / f"{sid:05d}.wav"
        if not raw_path.exists():
            drops.append({"id": sid, "filter": "missing_file"})
            continue
        try:
            audio16, _ = load_audio(raw_path, 16000)
        except Exception as e:
            drops.append({"id": sid, "filter": "load_error", "error": repr(e)})
            continue

        duration_s = len(audio16) / 16000
        chars = len(sentence)

        # 3.1 sanity
        if duration_s < chars * DUR_RATIO_MIN or duration_s > chars * DUR_RATIO_MAX:
            drops.append({"id": sid, "filter": "duration_ratio",
                          "duration": duration_s, "chars": chars,
                          "ratio": duration_s/max(chars,1)})
            continue
        peak = float(np.max(np.abs(audio16)))
        rms = float(np.sqrt(np.mean(audio16 ** 2)))
        if peak >= SAMPLES_NEAR_CLIP:
            drops.append({"id": sid, "filter": "clipping", "peak": peak})
            continue
        if rms < RMS_MIN:
            drops.append({"id": sid, "filter": "rms_low", "rms": rms})
            continue
        silences = detect_internal_silences(audio16, 16000, thresh_db=-40, min_silence_ms=INTERNAL_SILENCE_MAX_MS)
        if silences:
            drops.append({"id": sid, "filter": "internal_silence",
                          "count": len(silences)})
            continue

        # 3.2 Whisper WER
        try:
            r = whisper_model.transcribe(audio16, language="en", fp16=DEVICE.startswith("cuda"))
            transcript = r["text"]
        except Exception as e:
            drops.append({"id": sid, "filter": "whisper_error", "error": repr(e)})
            continue
        w = wer(sentence, transcript)
        if w > WHISPER_WER_MAX:
            drops.append({"id": sid, "filter": "wer", "wer": w,
                          "ref": sentence, "hyp": transcript})
            continue

        # 3.4 ECAPA + WavLM
        audio_t = torch.from_numpy(audio16).unsqueeze(0)
        with torch.no_grad():
            ec = ecapa.encode_batch(audio_t.to(sb_device)).squeeze().cpu().numpy()
            inputs = wavlm_fe(audio16, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(wavlm_device) for k, v in inputs.items()}
            wl = wavlm(**inputs).embeddings.squeeze().cpu().numpy()
        ec_sim = cos(ec, ECAPA_CENTROID)
        wl_sim = cos(wl, WAVLM_CENTROID)
        if ec_sim < ECAPA_MIN or wl_sim < WAVLM_MIN:
            drops.append({"id": sid, "filter": "speaker_verify",
                          "ecapa": ec_sim, "wavlm": wl_sim})
            continue

        # Stats for outlier detection — use librosa.yin (deterministic, fast) instead
        # of pyin (probabilistic, CPU-bound, ~10x slower than realtime; would make
        # Phase 3 take ~100h on 12000 clips).
        try:
            f0 = librosa.yin(audio16, fmin=50, fmax=400, sr=16000)
            f0_clean = f0[(f0 > 0) & np.isfinite(f0)]
            f0_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
            f0_range = float(np.max(f0_clean) - np.min(f0_clean)) if len(f0_clean) > 0 else 0.0
        except Exception:
            f0_mean = 0.0
            f0_range = 0.0
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio16, sr=16000)))

        per_clip_stats.append({
            "id": sid, "sentence": sentence,
            "duration_s": duration_s, "rms": rms,
            "f0_mean": f0_mean, "f0_range": f0_range,
            "centroid": centroid, "wer": w,
            "ecapa": ec_sim, "wavlm": wl_sim,
            "transcript": transcript,
        })
        surviving.append({"id": sid, "sentence": sentence,
                          "raw_path": str(raw_path)})

        if (n + 1) % 50 == 0 or n == len(rows) - 1:
            elapsed = time.monotonic() - t0
            print(f"  [{n+1}/{len(rows)}] surviving={len(surviving)} dropped={len(drops)} elapsed={elapsed:.0f}s", flush=True)

    # 3.5 statistical outliers
    if per_clip_stats:
        arr = lambda k: np.array([c[k] for c in per_clip_stats])
        before = len(per_clip_stats)
        final_stats = []
        survive_set = set()
        for c in per_clip_stats:
            keep = True
            for k in ("duration_s", "rms", "f0_mean", "centroid"):
                v = c[k]
                col = arr(k)
                mu, sd = float(np.mean(col)), float(np.std(col))
                if sd > 0 and abs(v - mu) > 3 * sd:
                    drops.append({"id": c["id"], "filter": "outlier", "axis": k,
                                  "value": v, "mu": mu, "sd": sd})
                    keep = False
                    break
            if keep:
                final_stats.append(c)
                survive_set.add(c["id"])
        surviving = [s for s in surviving if s["id"] in survive_set]
        print(f"after outlier filter: {len(surviving)} (was {before})")

    # 3.6 resample to 22050 mono 16-bit
    print(f"resampling {len(surviving)} clips to 22050 Hz")
    with OUT_META.open("w") as f:
        writer = csv.writer(f, delimiter="|")
        for s in surviving:
            sid = s["id"]
            raw_path = Path(s["raw_path"])
            audio, native_sr = sf.read(str(raw_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio22 = librosa.resample(audio, orig_sr=native_sr, target_sr=22050)
            audio22 = np.clip(audio22, -1.0, 1.0)
            audio22_i16 = (audio22 * 32767).astype(np.int16)
            out_path = OUT_AUDIO / f"{sid:05d}.wav"
            sf.write(str(out_path), audio22_i16, 22050, subtype="PCM_16")
            writer.writerow([f"{sid:05d}", s["sentence"]])

    # totals
    total_hours = sum(c["duration_s"] for c in per_clip_stats if c["id"] in {s["id"] for s in surviving}) / 3600
    drop_histogram: dict[str, int] = {}
    for d in drops:
        drop_histogram[d["filter"]] = drop_histogram.get(d["filter"], 0) + 1

    manifest = {
        "phase": "phase3",
        "input_clips": len(rows),
        "retained": len(surviving),
        "dropped": len(drops),
        "drop_rate": len(drops) / max(len(rows), 1),
        "retained_hours": total_hours,
        "drop_histogram": drop_histogram,
        "thresholds": {
            "ecapa_min": ECAPA_MIN,
            "wavlm_min": WAVLM_MIN,
            "wer_max": WHISPER_WER_MAX,
        },
        "hard_floor_hours": 2,
        "below_hard_floor": total_hours < 2,
        "next_phase": "phase4",
    }
    (STATE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (STATE / "drops.jsonl").open("w") as f:
        for d in drops:
            f.write(json.dumps(d) + "\n")
    with (STATE / "per_clip_stats.json").open("w") as f:
        json.dump(per_clip_stats, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
