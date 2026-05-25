"""Prep Run C dataset: resample phase3-full/audio (22050 Hz) to phase3-low/audio (16000 Hz).

Run on CPU; can run concurrently with Run B training on GPU 1. Uses a process
pool for parallel resampling.
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

RUN = Path(__file__).resolve().parent.parent
import json as _piper_json
OUT = RUN / "output" / _piper_json.loads((RUN / "run_config.json").read_text())["teacher"]["voice_id"]
SRC_AUDIO = OUT / "phase3-full/audio"
SRC_META = OUT / "phase3-full/metadata.csv"
DST_AUDIO = OUT / "phase3-low/audio"
DST_META = OUT / "phase3-low/metadata.csv"
TARGET_SR = 16000


def resample_one(src: Path) -> tuple[str, bool, str]:
    dst = DST_AUDIO / src.name
    if dst.exists():
        return (src.name, True, "skip")
    try:
        audio, sr = sf.read(str(src), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        audio = np.clip(audio, -1.0, 1.0)
        audio_i16 = (audio * 32767).astype(np.int16)
        sf.write(str(dst), audio_i16, TARGET_SR, subtype="PCM_16")
        return (src.name, True, "ok")
    except Exception as e:
        return (src.name, False, repr(e))


def main() -> None:
    DST_AUDIO.mkdir(parents=True, exist_ok=True)
    srcs = sorted(SRC_AUDIO.glob("*.wav"))
    print(f"resampling {len(srcs)} clips: 22050 -> {TARGET_SR} Hz")
    t0 = time.monotonic()
    n_workers = min(mp.cpu_count(), 16)
    with mp.Pool(n_workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(resample_one, srcs, chunksize=8)):
            results.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  [{i+1}/{len(srcs)}] elapsed {time.monotonic()-t0:.0f}s")
    ok = sum(1 for _, b, _ in results if b)
    print(f"done: {ok}/{len(results)} OK in {time.monotonic()-t0:.1f}s ({n_workers} workers)")
    # copy metadata as-is (ids unchanged)
    shutil.copy(SRC_META, DST_META)
    print(f"wrote {DST_META}")


if __name__ == "__main__":
    main()
