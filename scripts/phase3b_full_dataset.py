"""Build Run B dataset: phase3 curated + the WER-dropped clips added back.

Keep dropped: clipping (truly bad audio), outlier (statistical anomaly),
              speaker_verify (voice drift), duration_ratio (truncation).
Add back:     wer (accent-induced Whisper struggle is the SIGNAL we want).

Inputs:
  phase3/metadata.csv       (8143 curated)
  phase3/audio/*.wav        (already 22 kHz)
  state/phase3/drops.jsonl  (drop reasons)
  phase2/audio_raw/*.wav    (24 kHz source)

Outputs:
  phase3-full/audio/*.wav   (8143 curated + ~3570 wer = ~11700)
  phase3-full/metadata.csv  (combined)
  state/phase3-full/manifest.json
"""
from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

RUN = Path(__file__).resolve().parent.parent
import json as _piper_json
OUT = RUN / "output" / _piper_json.loads((RUN / "run_config.json").read_text())["teacher"]["voice_id"]
SRC_META = OUT / "phase3/metadata.csv"
SRC_AUDIO = OUT / "phase3/audio"
DROPS = OUT / "state/phase3/drops.jsonl"
RAW = OUT / "phase2/audio_raw"
CORPUS = OUT / "phase1/corpus.txt"

DST_AUDIO = OUT / "phase3-full/audio"
DST_META = OUT / "phase3-full/metadata.csv"
STATE = OUT / "state/phase3-full"


def resample_22k(src: Path, dst: Path) -> float:
    audio, sr = sf.read(str(src), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 22050:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
    audio = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio * 32767).astype(np.int16)
    sf.write(str(dst), audio_i16, 22050, subtype="PCM_16")
    return len(audio_i16) / 22050


def main() -> None:
    DST_AUDIO.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)

    sentences = CORPUS.read_text().splitlines()

    # 1) Copy all curated clips + meta
    print("step 1: hardlinking curated clips + meta")
    curated = []
    with SRC_META.open() as f:
        for row in csv.reader(f, delimiter="|"):
            sid_str, sentence = row[0], row[1]
            curated.append((sid_str, sentence))
    t0 = time.monotonic()
    for sid_str, _ in curated:
        src = SRC_AUDIO / f"{sid_str}.wav"
        dst = DST_AUDIO / f"{sid_str}.wav"
        if not dst.exists():
            try:
                dst.hardlink_to(src)
            except (OSError, FileExistsError):
                shutil.copy(src, dst)
    print(f"  hardlinked {len(curated)} in {time.monotonic()-t0:.1f}s")

    # 2) Find WER drops and add their audio
    print("step 2: adding back WER drops")
    wer_drops = []
    with DROPS.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("filter") == "wer":
                wer_drops.append(d)
    print(f"  {len(wer_drops)} WER drops to recover")

    added: list[tuple[str, str]] = []
    skipped: list[int] = []
    t0 = time.monotonic()
    for i, d in enumerate(wer_drops):
        sid = d["id"]
        if sid >= len(sentences):
            skipped.append(sid)
            continue
        sentence = sentences[sid]
        src = RAW / f"{sid:05d}.wav"
        if not src.exists():
            skipped.append(sid)
            continue
        dst = DST_AUDIO / f"{sid:05d}.wav"
        if dst.exists():
            # already in (shouldn't happen but skip)
            continue
        try:
            resample_22k(src, dst)
            added.append((f"{sid:05d}", sentence))
        except Exception as e:
            print(f"  failed {sid}: {e}")
            skipped.append(sid)
        if (i + 1) % 500 == 0:
            print(f"    [{i+1}/{len(wer_drops)}] added={len(added)} elapsed={time.monotonic()-t0:.0f}s")
    print(f"  added {len(added)} (skipped {len(skipped)}) in {time.monotonic()-t0:.1f}s")

    # 3) Build full metadata
    all_rows = curated + added
    with DST_META.open("w") as f:
        w = csv.writer(f, delimiter="|")
        for sid_str, sentence in all_rows:
            w.writerow([sid_str, sentence])
    print(f"step 3: wrote {DST_META} with {len(all_rows)} rows")

    # 4) Manifest
    manifest = {
        "phase": "phase3-full",
        "based_on": "phase3 + recovered WER drops",
        "curated_count": len(curated),
        "wer_drops_recovered": len(added),
        "skipped": len(skipped),
        "total_clips": len(all_rows),
        "rationale": "WER filter biased against accented clips; for an accented voice the high-WER cases are the SIGNAL we want to train on, not noise to discard.",
    }
    (STATE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
