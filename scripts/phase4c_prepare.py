"""Phase 4 prep for Run C: train/eval split from phase3-low/metadata.csv.

Same eval IDs as Run B (same seed) so the A/B/C comparison can use overlapping
sentences when ids match.
"""
import csv
import json
import random
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
import json as _piper_json
OUT = RUN / "output" / _piper_json.loads((RUN / __import__("os").environ.get("PIPER_DISTILL_CONFIG", "run_config.json")).read_text())["teacher"]["voice_id"]
META = OUT / "phase3-low/metadata.csv"
AUDIO = OUT / "phase3-low/audio"
TRAIN_META = OUT / "phase4-low/train_metadata.csv"
EVAL_META = OUT / "phase4-low/eval_metadata.csv"
STATE = OUT / "state/phase4-low"

EVAL_SIZE = 250


def main() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    (OUT / "phase4-low").mkdir(parents=True, exist_ok=True)
    rows = []
    with META.open() as f:
        for row in csv.reader(f, delimiter="|"):
            if (AUDIO / f"{row[0]}.wav").exists():
                rows.append((row[0], row[1]))
    print(f"total clips: {len(rows)}")
    rng = random.Random(20260523)  # same seed as Runs A and B
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
        "phase": "phase4_prep_low",
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
    }, indent=2))
    print("wrote train+eval splits")


if __name__ == "__main__":
    main()
