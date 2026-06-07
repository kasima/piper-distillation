"""Phase 0.3 — synthesize 60 probe sentences x 2 takes against qwen-tts Takashii.

Output:
- clips/first/<id>.wav   (first take, what kasima reviews)
- clips/second/<id>.wav  (second take, for self-similarity)
- synthesis_log.jsonl    (per-clip latency, bytes, errors)
- metadata.csv           (id|take|sentence|duration_s|bytes)
"""
import csv
import json
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

RUN = Path(__file__).resolve().parent.parent
import json as _piper_json
_CFG = _piper_json.loads((RUN / __import__("os").environ.get("PIPER_DISTILL_CONFIG", "run_config.json")).read_text())
OUT = RUN / "output" / _CFG["teacher"]["voice_id"]
PROBE_PATH = OUT / "state/phase0/probe_sentences.txt"
CLIPS_DIR = OUT / "state/phase0/clips"
LOG_PATH = OUT / "state/phase0/synthesis_log.jsonl"
META_PATH = OUT / "state/phase0/metadata.csv"

# Config-driven (was hardcoded to voice=Takashii — a bug that built the Phase 0
# centroid from the WRONG voice, so Phase 3 speaker-verify dropped every clip).
ENDPOINT = _CFG["teacher"].get("endpoint", "http://192.168.122.1:8880/v1/audio/speech")
PAYLOAD_BASE = {**_CFG["teacher"].get("params", {}), "voice": _CFG["teacher"]["voice_id"]}
CONCURRENCY = 2
TIMEOUT_SECONDS = 60


def synth(sentence_id: int, take: int, sentence: str) -> dict:
    out_dir = CLIPS_DIR / ("first" if take == 1 else "second")
    out_path = out_dir / f"{sentence_id:03d}.wav"
    payload = {**PAYLOAD_BASE, "input": sentence}
    t0 = time.monotonic()
    try:
        r = requests.post(
            ENDPOINT,
            json=payload,
            timeout=TIMEOUT_SECONDS,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        out_path.write_bytes(r.content)
        try:
            with wave.open(str(out_path), "rb") as w:
                duration_s = w.getnframes() / w.getframerate()
                sr = w.getframerate()
                ch = w.getnchannels()
        except Exception:
            duration_s = -1.0
            sr = -1
            ch = -1
        return {
            "id": sentence_id,
            "take": take,
            "ok": True,
            "latency_s": time.monotonic() - t0,
            "bytes": len(r.content),
            "duration_s": duration_s,
            "sample_rate": sr,
            "channels": ch,
            "path": str(out_path),
        }
    except Exception as e:
        return {
            "id": sentence_id,
            "take": take,
            "ok": False,
            "latency_s": time.monotonic() - t0,
            "error": repr(e),
        }


def main() -> None:
    CLIPS_DIR.joinpath("first").mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.joinpath("second").mkdir(parents=True, exist_ok=True)
    # Seed per-voice probe set from the committed default if missing.
    if not PROBE_PATH.exists():
        PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROBE_PATH.write_text((RUN / "scripts/probe_sentences.txt").read_text())
    sentences = [
        s.strip() for s in PROBE_PATH.read_text().splitlines() if s.strip()
    ]
    assert len(sentences) == 60, f"expected 60 probe sentences, got {len(sentences)}"

    jobs = []
    for idx, sentence in enumerate(sentences):
        jobs.append((idx, 1, sentence))
        jobs.append((idx, 2, sentence))

    print(f"submitting {len(jobs)} jobs at concurrency={CONCURRENCY}")
    results: list[dict] = []
    t_start = time.monotonic()
    with LOG_PATH.open("w") as logf, ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(synth, *j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            logf.write(json.dumps(res) + "\n")
            logf.flush()
            done += 1
            if done % 20 == 0 or done == len(jobs):
                rate = done / (time.monotonic() - t_start)
                print(f"  [{done}/{len(jobs)}] rate={rate:.2f} clips/s")
    total = time.monotonic() - t_start
    ok = sum(1 for r in results if r.get("ok"))
    print(f"done in {total:.1f}s, {ok}/{len(jobs)} ok")

    results.sort(key=lambda r: (r["id"], r["take"]))
    with META_PATH.open("w") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["id", "take", "sentence", "duration_s", "bytes"])
        for r in results:
            if r.get("ok"):
                writer.writerow(
                    [
                        r["id"],
                        r["take"],
                        sentences[r["id"]],
                        f"{r['duration_s']:.3f}",
                        r["bytes"],
                    ]
                )
    print(f"wrote {META_PATH}")


if __name__ == "__main__":
    main()
