"""Phase 1 — text corpus curation.

Pipeline:
  1. Strip Gutenberg boilerplate from books
  2. Sentence-split
  3. Normalize numbers + abbreviations
  4. Phonemize via espeak-ng (en-us, IPA)
  5. Filter (length, numbers, URLs, repeats, all-caps)
  6. Greedy select ~12k sentences scored by:
       - phoneme bigram coverage gain
       - accent-marker bonus (r/l, θ/ð, vowels absent in JA)
       - length-bucket balance
       - prosody balance (declarative / interrogative / exclamatory / complex)

Outputs:
  phase1/corpus.txt              — selected sentences, one per line
  phase1/corpus.tsv              — id | sentence | phonemes | category
  state/phase1/coverage_report.md
  state/phase1/manifest.json
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
import json as _piper_json
OUT = RUN / "output" / _piper_json.loads((RUN / "run_config.json").read_text())["teacher"]["voice_id"]
SOURCES = OUT / "phase1/sources"
OUT_CORPUS = OUT / "phase1/corpus.txt"
OUT_TSV = OUT / "phase1/corpus.tsv"
OUT_PHONE_CACHE = OUT / "phase1/phonemes_cache.tsv"
STATE = OUT / "state/phase1"

TARGET_SIZE = 12000
COVERAGE_BIGRAM_FLOOR_RARE = 20
COVERAGE_BIGRAM_FLOOR = 50

# Length buckets
SHORT = (4, 8)
MED = (8, 14)
LONG = (14, 20)
LENGTH_TARGET = {"short": 0.40, "med": 0.40, "long": 0.20}

# Prosody targets
PROSODY_TARGET = {"declarative": 0.60, "interrogative": 0.20, "exclamatory": 0.10, "complex": 0.10}

# Accent-marker phonemes (IPA from espeak-ng en-us)
ACCENT_MARKERS = {
    "r", "ɹ", "l", "ɫ",        # /r/-/l/
    "θ", "ð",                   # th
    "æ", "ɪ", "ʊ", "ɑ",         # vowels absent in JA
}


def strip_gutenberg(text: str) -> str:
    """Strip standard Project Gutenberg header/footer."""
    start_re = re.compile(r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG.*?\*\*\*", re.S | re.I)
    end_re = re.compile(r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG.*?\*\*\*", re.S | re.I)
    s = start_re.search(text)
    e = end_re.search(text)
    if s:
        text = text[s.end():]
    if e:
        text = text[:e.start() if not s else (e.start() - s.end())] if s else text[:e.start()]
    return text


def sentence_split(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("—", " — ")
    # Naive sentence segmentation on . ! ?
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
    return [p.strip() for p in parts if p.strip()]


SMALL_NUM = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty",
}


def normalize(sent: str) -> str:
    s = sent
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[‘’]", "'", s)
    s = s.replace("--", " — ")
    s = re.sub(r"\bMr\.", "Mister", s)
    s = re.sub(r"\bMrs\.", "Misses", s)
    s = re.sub(r"\bDr\.", "Doctor", s)
    s = re.sub(r"\bSt\.", "Saint", s)
    s = re.sub(r"\bNo\.", "Number", s)
    s = re.sub(r"\bvs\.", "versus", s)
    # Replace simple small numbers
    def num_repl(m: re.Match) -> str:
        n = m.group(0)
        return SMALL_NUM.get(n, n)
    s = re.sub(r"\b\d+\b", num_repl, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def category(words: int) -> str | None:
    if SHORT[0] <= words <= SHORT[1]:
        return "short"
    if MED[0] < words <= MED[1]:
        return "med"
    if LONG[0] < words <= LONG[1]:
        return "long"
    return None


def prosody(sent: str) -> str:
    s = sent.strip()
    if s.endswith("?"):
        return "interrogative"
    if s.endswith("!"):
        return "exclamatory"
    # Complex if has internal comma/semicolon/em-dash
    if re.search(r"[;,—:]", s):
        return "complex"
    return "declarative"


def accept(sent: str, words: int) -> bool:
    if not (4 <= words <= 20):
        return False
    if re.search(r"https?://|www\.", sent):
        return False
    digits = sum(1 for c in sent if c.isdigit())
    if digits > 0:  # post-normalization, any remaining digits are big numbers we don't want
        return False
    if re.search(r"\b[A-Z]{4,}\b.*\b[A-Z]{4,}\b", sent):
        return False
    # Repeated token like "the the"
    if re.search(r"\b(\w+)\s+\1\b", sent.lower()):
        return False
    # Reject non-ASCII printable
    if not all(ord(c) < 128 for c in sent):
        return False
    # Reject sentences that look like chapter headings or formal markers
    if re.match(r"^(CHAPTER|VOLUME|BOOK|PART|ACT|SCENE)\b", sent, re.I):
        return False
    # Reject very dialog-heavy with embedded multi quotes
    if sent.count('"') >= 4:
        return False
    return True


def phonemize_batch(sentences: list[str], batch_size: int = 200) -> list[str]:
    """Phonemize via espeak-ng. One sentence per input line, one IPA line out per."""
    out: list[str] = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        # espeak-ng reads stdin lines; -q quiet; --ipa=3 dashes; -v en-us
        # We use one process per chunk, feeding one sentence per line then a blank.
        # Trick: espeak-ng's `--phonout=-` writes phonemes to stdout but is awkward.
        # We use `--ipa=3` with `-q` and `--sep=_` to print phonemes inline.
        proc = subprocess.run(
            ["espeak-ng", "-v", "en-us", "-q", "--ipa=1"],
            input="\n".join(chunk),
            text=True,
            capture_output=True,
        )
        # espeak emits one line per sentence on stdout
        lines = proc.stdout.splitlines()
        # If counts don't line up, pad
        if len(lines) != len(chunk):
            # Fallback: phonemize one at a time
            lines = []
            for s in chunk:
                p = subprocess.run(
                    ["espeak-ng", "-v", "en-us", "-q", "--ipa=1"],
                    input=s, text=True, capture_output=True,
                )
                lines.append(p.stdout.strip().replace("\n", " "))
        out.extend(lines)
    return out


_STRESS_MARKS = "ˈˌ"


def _phoneme_tokens(phon: str) -> list[str]:
    """Split IPA=1 output into phoneme tokens.

    Format: words separated by space, phonemes within a word separated by '_'.
    Stress markers (ˈ, ˌ) precede the vowel; strip them.
    """
    toks: list[str] = []
    for word in phon.split():
        for p in word.split("_"):
            p = p.strip()
            if not p:
                continue
            # strip leading stress marks
            while p and p[0] in _STRESS_MARKS:
                p = p[1:]
            if p:
                toks.append(p)
    return toks


def bigrams(phon: str) -> list[str]:
    toks = _phoneme_tokens(phon)
    if len(toks) < 2:
        return []
    return [f"{toks[i]} {toks[i+1]}" for i in range(len(toks)-1)]


def has_accent_marker(phon: str) -> int:
    toks = _phoneme_tokens(phon)
    return sum(1 for t in toks if any(m in t for m in ACCENT_MARKERS))


def main() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    raw_sents: list[str] = []
    for p in sorted(SOURCES.glob("*.txt")):
        body = strip_gutenberg(p.read_text(encoding="utf-8", errors="ignore"))
        raw_sents.extend(sentence_split(body))
    print(f"raw sentences: {len(raw_sents)}")

    # Normalize + accept
    normed = []
    for s in raw_sents:
        n = normalize(s)
        w = len(n.split())
        if accept(n, w):
            normed.append(n)
    print(f"after normalize+filter: {len(normed)}")
    # Dedupe (case-insensitive)
    seen = set()
    unique = []
    for s in normed:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        unique.append(s)
    print(f"after dedupe: {len(unique)}")

    # Phonemize (slow — cache result on disk)
    if OUT_PHONE_CACHE.exists():
        # load cache; format: sentence \t phoneme
        cache = {}
        for line in OUT_PHONE_CACHE.read_text().splitlines():
            if "\t" in line:
                s, p = line.split("\t", 1)
                cache[s] = p
        phones = [cache.get(s, "") for s in unique]
        missing = [i for i, p in enumerate(phones) if not p]
        if missing:
            new = phonemize_batch([unique[i] for i in missing])
            for idx, p in zip(missing, new):
                phones[idx] = p
                cache[unique[idx]] = p
            with OUT_PHONE_CACHE.open("w") as f:
                for s, p in cache.items():
                    f.write(f"{s}\t{p}\n")
    else:
        t0 = time.monotonic()
        phones = phonemize_batch(unique)
        print(f"phonemized {len(unique)} in {time.monotonic()-t0:.1f}s")
        with OUT_PHONE_CACHE.open("w") as f:
            for s, p in zip(unique, phones):
                f.write(f"{s}\t{p}\n")

    # Score and greedy-select
    candidates = []  # (sentence, phon, words, cat, prosody, n_accent)
    for s, p in zip(unique, phones):
        words = len(s.split())
        cat = category(words)
        if cat is None:
            continue
        pro = prosody(s)
        n_acc = has_accent_marker(p)
        candidates.append((s, p, words, cat, pro, n_acc))
    print(f"candidates after length/category: {len(candidates)}")
    random.shuffle(candidates)

    coverage: Counter[str] = Counter()
    cat_count = Counter()
    pro_count = Counter()
    chosen: list[tuple] = []

    def score(item) -> float:
        _, phon, words, cat, pro, n_acc = item
        bgs = bigrams(phon)
        if not bgs:
            return -1e9
        gain = sum(1.0 / (1 + coverage[bg]) for bg in bgs)
        marker_bonus = 2.0 * n_acc
        # Length stratification penalty
        total = sum(cat_count.values()) or 1
        cat_share = cat_count[cat] / total
        cat_penalty = max(0, cat_share - LENGTH_TARGET[cat]) * 5
        # Prosody penalty
        ptotal = sum(pro_count.values()) or 1
        p_share = pro_count[pro] / ptotal
        p_penalty = max(0, p_share - PROSODY_TARGET[pro]) * 5
        return gain + marker_bonus - cat_penalty - p_penalty

    target = min(TARGET_SIZE, len(candidates))
    print(f"greedy-selecting {target}")
    t0 = time.monotonic()
    # For each pick, rescan all remaining candidates. With ~50k candidates and 12k picks,
    # naive is 600M scores — too slow. Use a coarse approach: pick by current best out of
    # a stochastic sample of size 200, rebuilt every step. Approximate but tractable.
    SAMPLE_SIZE = 256
    remaining = list(candidates)
    while len(chosen) < target and remaining:
        sample = random.sample(remaining, min(SAMPLE_SIZE, len(remaining)))
        best = max(sample, key=score)
        chosen.append(best)
        remaining.remove(best)
        s, phon, w, cat, pro, n_acc = best
        for bg in bigrams(phon):
            coverage[bg] += 1
        cat_count[cat] += 1
        pro_count[pro] += 1
        if len(chosen) % 1000 == 0:
            print(f"  picked {len(chosen)} in {time.monotonic()-t0:.1f}s, distinct bigrams={len(coverage)}")
    print(f"selected {len(chosen)} in {time.monotonic()-t0:.1f}s")

    # Write outputs
    OUT_CORPUS.write_text("\n".join(c[0] for c in chosen) + "\n")
    with OUT_TSV.open("w") as f:
        for i, (s, phon, w, cat, pro, n_acc) in enumerate(chosen):
            f.write(f"{i}\t{s}\t{phon}\t{cat}\t{pro}\t{n_acc}\n")

    # Coverage report
    bg_total = sum(coverage.values())
    below_floor = sum(1 for c in coverage.values() if c < COVERAGE_BIGRAM_FLOOR)
    below_rare = sum(1 for c in coverage.values() if c < COVERAGE_BIGRAM_FLOOR_RARE)
    accent_marker_counts = {m: 0 for m in ACCENT_MARKERS}
    for s, phon, w, cat, pro, n_acc in chosen:
        for tok in _phoneme_tokens(phon):
            for m in ACCENT_MARKERS:
                if m in tok:
                    accent_marker_counts[m] += 1
    length_dist = {k: cat_count[k] for k in ("short", "med", "long")}
    prosody_dist = dict(pro_count)

    report_lines = [
        "# Phase 1 coverage report",
        "",
        f"Selected sentences: {len(chosen)}",
        f"Distinct phoneme bigrams: {len(coverage)}",
        f"Total phoneme bigrams emitted: {bg_total}",
        f"Bigrams below rare floor ({COVERAGE_BIGRAM_FLOOR_RARE}): {below_rare}",
        f"Bigrams below standard floor ({COVERAGE_BIGRAM_FLOOR}): {below_floor}",
        "",
        "## Accent-marker phoneme coverage",
        ""
    ]
    for m in sorted(accent_marker_counts):
        report_lines.append(f"- `{m}` : {accent_marker_counts[m]}")
    report_lines.append("")
    report_lines.append("## Length distribution")
    report_lines.append("")
    total = sum(length_dist.values())
    for k, v in length_dist.items():
        report_lines.append(f"- {k}: {v} ({v/total:.0%}, target {LENGTH_TARGET[k]:.0%})")
    report_lines.append("")
    report_lines.append("## Prosody distribution")
    report_lines.append("")
    total = sum(prosody_dist.values())
    for k, v in prosody_dist.items():
        report_lines.append(f"- {k}: {v} ({v/total:.0%}, target {PROSODY_TARGET.get(k,0):.0%})")
    (STATE / "coverage_report.md").write_text("\n".join(report_lines))

    manifest = {
        "phase": "phase1",
        "sources_dir": str(SOURCES),
        "raw_sentence_count": len(raw_sents),
        "normalized_accepted_count": len(normed),
        "unique_count": len(unique),
        "candidate_count_after_length_cat": len(candidates),
        "selected_count": len(chosen),
        "distinct_bigrams": len(coverage),
        "bigrams_below_floor_50": below_floor,
        "bigrams_below_rare_floor_20": below_rare,
        "length_distribution": length_dist,
        "prosody_distribution": prosody_dist,
        "accent_marker_counts": accent_marker_counts,
        "corpus_path": str(OUT_CORPUS),
        "corpus_tsv_path": str(OUT_TSV),
        "next_phase": "phase2",
    }
    (STATE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest, corpus.txt ({len(chosen)} lines), coverage_report.md")


if __name__ == "__main__":
    main()
