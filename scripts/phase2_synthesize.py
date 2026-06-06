"""Phase 2 — corpus-wide synthesis against qwen-tts Takashii.

Resumable: re-reads done.txt at start; skips ids already present.
Concurrency from run_config.json (teacher.concurrency_max_safe).
"""
from __future__ import annotations

import csv
import json
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

RUN = Path(__file__).resolve().parent.parent
import json as _piper_json
OUT = RUN / "output" / _piper_json.loads((RUN / __import__("os").environ.get("PIPER_DISTILL_CONFIG", "run_config.json")).read_text())["teacher"]["voice_id"]
CFG = json.loads((RUN / __import__("os").environ.get("PIPER_DISTILL_CONFIG", "run_config.json")).read_text())
CORPUS = OUT / "phase1/corpus.txt"
OUT_DIR = OUT / "phase2/audio_raw"
META = OUT / "state/phase2/metadata_raw.csv"
DONE = OUT / "state/phase2/done.txt"
LOG = OUT / "state/phase2/synthesis_log.jsonl"
STATE = OUT / "state/phase2"

ENDPOINT = CFG["teacher"]["endpoint"]
PARAMS = CFG["teacher"]["params"]
VOICE = CFG["teacher"]["voice_id"]
CONCURRENCY = CFG["teacher"]["concurrency_max_safe"]
TIMEOUT_S = 90
RETRY_MAX = 5
RETRY_BACKOFF_BASE = 2.0


def already_done() -> set[int]:
    done: set[int] = set()
    if DONE.exists():
        for line in DONE.read_text().splitlines():
            line = line.strip()
            if line.isdigit():
                done.add(int(line))
    return done


def synth_one(sid: int, sentence: str) -> dict:
    payload = {**PARAMS, "voice": VOICE, "input": sentence}
    out_path = OUT_DIR / f"{sid:05d}.wav"
    last_err: Exception | None = None
    for attempt in range(1, RETRY_MAX + 1):
        t0 = time.monotonic()
        try:
            r = requests.post(
                ENDPOINT, json=payload, timeout=TIMEOUT_S,
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            out_path.write_bytes(r.content)
            with wave.open(str(out_path), "rb") as w:
                duration_s = w.getnframes() / w.getframerate()
                sr = w.getframerate()
                ch = w.getnchannels()
            return {
                "id": sid, "ok": True, "attempts": attempt,
                "latency_s": time.monotonic() - t0,
                "bytes": len(r.content),
                "duration_s": duration_s, "sample_rate": sr, "channels": ch,
            }
        except Exception as e:
            last_err = e
            time.sleep(RETRY_BACKOFF_BASE ** attempt)
    return {"id": sid, "ok": False, "attempts": RETRY_MAX,
            "error": repr(last_err)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)

    sentences = CORPUS.read_text().splitlines()
    total = len(sentences)
    print(f"corpus: {total} sentences, concurrency={CONCURRENCY}")

    done = already_done()
    todo_idx = [i for i in range(total) if i not in done]
    print(f"already done: {len(done)}, todo: {len(todo_idx)}")
    if not todo_idx:
        print("all done")
        return

    meta_mode = "a" if META.exists() else "w"
    done_mode = "a"
    log_mode = "a"
    META.parent.mkdir(parents=True, exist_ok=True)
    if not META.exists():
        with META.open("w") as f:
            csv.writer(f, delimiter="|").writerow(
                ["id", "sentence", "duration_s", "bytes"]
            )

    t_start = time.monotonic()
    latencies: list[float] = []
    failures = 0
    success = 0
    with (META.open("a") as metaf,
          DONE.open(done_mode) as donef,
          LOG.open(log_mode) as logf,
          ThreadPoolExecutor(max_workers=CONCURRENCY) as ex):
        meta_writer = csv.writer(metaf, delimiter="|")
        futs = {ex.submit(synth_one, i, sentences[i]): i for i in todo_idx}
        completed = 0
        for fut in as_completed(futs):
            res = fut.result()
            sid = res["id"]
            completed += 1
            logf.write(json.dumps(res) + "\n")
            logf.flush()
            if res.get("ok"):
                success += 1
                latencies.append(res["latency_s"])
                meta_writer.writerow([
                    sid,
                    sentences[sid],
                    f"{res['duration_s']:.3f}",
                    res["bytes"],
                ])
                metaf.flush()
                donef.write(f"{sid}\n")
                donef.flush()
            else:
                failures += 1
            if completed % 100 == 0 or completed == len(todo_idx):
                elapsed = time.monotonic() - t_start
                rate = completed / elapsed
                eta_h = (len(todo_idx) - completed) / rate / 3600 if rate > 0 else 0
                if latencies:
                    p50 = sorted(latencies)[len(latencies) // 2]
                    mean = sum(latencies) / len(latencies)
                else:
                    p50 = mean = 0.0
                print(
                    f"[{completed:>5}/{len(todo_idx)}] ok={success} fail={failures} "
                    f"elapsed={elapsed/3600:.2f}h rate={rate:.2f}/s eta={eta_h:.2f}h "
                    f"mean={mean:.2f}s p50={p50:.2f}s",
                    flush=True,
                )

    print(f"DONE total time {(time.monotonic()-t_start)/3600:.2f}h "
          f"success={success} failures={failures}")
    # write phase2 manifest stub
    manifest = {
        "phase": "phase2",
        "total_sentences": total,
        "succeeded": success,
        "failed": failures,
        "wall_clock_hours": (time.monotonic() - t_start) / 3600,
        "concurrency": CONCURRENCY,
        "next_phase": "phase3",
    }
    (STATE / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
