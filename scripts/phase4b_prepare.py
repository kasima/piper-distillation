"""Phase 4 prep for Run B: train/eval split from phase3-full/metadata.csv.

Same split logic as phase4_prepare.py but reads/writes phase3-full/ + phase4-full/.
Uses the same RNG seed so the eval set partially overlaps with Run A's eval
(any shared id has the same sentence); this makes A/B comparison cleaner.
"""
import csv
import json
import random
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
import json as _piper_json
OUT = RUN / "output" / _piper_json.loads((RUN / __import__("os").environ.get("PIPER_DISTILL_CONFIG", "run_config.json")).read_text())["teacher"]["voice_id"]
META = OUT / "phase3-full/metadata.csv"
AUDIO = OUT / "phase3-full/audio"
TRAIN_META = OUT / "phase4-full/train_metadata.csv"
EVAL_META = OUT / "phase4-full/eval_metadata.csv"
STATE = OUT / "state/phase4-full"

EVAL_SIZE = 250


def main() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    (OUT / "phase4-full").mkdir(parents=True, exist_ok=True)
    rows = []
    with META.open() as f:
        for row in csv.reader(f, delimiter="|"):
            if (AUDIO / f"{row[0]}.wav").exists():
                rows.append((row[0], row[1]))
    print(f"total clips: {len(rows)}")

    rng = random.Random(20260523)  # same seed as Run A
    rng.shuffle(rows)
    eval_rows = rows[:EVAL_SIZE]
    train_rows = rows[EVAL_SIZE:]
    print(f"train={len(train_rows)} eval={len(eval_rows)}")

    with TRAIN_META.open("w") as f:
        w = csv.writer(f, delimiter="|")
        for r in train_rows: w.writerow(r)
    with EVAL_META.open("w") as f:
        w = csv.writer(f, delimiter="|")
        for r in eval_rows: w.writerow(r)
    (STATE / "prep_manifest.json").write_text(json.dumps({
        "phase": "phase4_prep_full",
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
    }, indent=2))
    print(f"wrote train+eval splits")


if __name__ == "__main__":
    main()
