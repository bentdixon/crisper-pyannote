"""Rank interviews by error rate, and name the failure mode on each.

An average tells you a system is worse; it does not tell you on which
recordings or in what way. This writes one row per interview with every
system's error rate side by side and the shape of the errors underneath it,
because the interviews at the top of the list fail for entirely different
reasons -- one system drops two thirds of the audio, another repeats itself,
another substitutes wrong words at the right times -- and those need different
fixes.

The dominant term is reported per system: substitutions, deletions and
insertions each divided by the reference length, largest wins. A deletion-heavy
failure is speech that was never transcribed; an insertion-heavy one is text
with no audio behind it; a substitution-heavy one is text present at roughly
the right length that does not match, which on this corpus has meant a
misaligned or mistranscribed recording rather than ordinary word errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import systems as registry


def shape(entry: dict) -> tuple[str, float]:
    """Which error term dominates, and its rate against the reference."""
    total = entry.get("total_words") or 0
    if not total:
        return "", 0.0
    terms = {
        "substitutions": entry.get("substitutions", 0) / total,
        "deletions": entry.get("deletions", 0) / total,
        "insertions": entry.get("insertions", 0) / total,
    }
    name = max(terms, key=terms.__getitem__)
    return name, terms[name]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jiwer", type=Path, help="jiwer_wer.json")
    parser.add_argument("--output", type=Path, default=None, help="CSV to write")
    parser.add_argument("--top", type=int, default=20, help="rows to print")
    parser.add_argument(
        "--rank-by", default="chirp3",
        help="system key to sort by; 'worst' sorts by the highest of any system",
    )
    args = parser.parse_args()

    data = json.loads(args.jiwer.read_text())
    keys = [k for k in registry.present_keys(data["aggregate"]) if not k.endswith("_llm")]
    rows = []
    for visit, entry in data["per_visit"].items():
        found = {k: registry.entry_of(entry, k) for k in keys}
        if any(v is None for v in found.values()):
            continue
        row = {"visit": visit, "reference_words": found[keys[0]].get("total_words")}
        for key in keys:
            term, rate = shape(found[key])
            row[f"{key}_wer"] = round(float(found[key]["wer"]), 4)
            row[f"{key}_shape"] = term
            row[f"{key}_shape_rate"] = round(rate, 4)
        rows.append(row)

    if args.rank_by == "worst":
        rows.sort(key=lambda r: -max(r[f"{k}_wer"] for k in keys))
    else:
        rows.sort(key=lambda r: -r[f"{args.rank_by}_wer"])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output} ({len(rows)} interviews)")

    width = max(len(v) for v in (r["visit"] for r in rows))
    header = "".join(f"{registry.PARTS[k][0][:14]:>16}" for k in keys)
    print(f"{'interview':{width}s}{header}   dominant error ({args.rank_by})")
    for row in rows[: args.top]:
        cells = "".join(f"{row[f'{k}_wer'] * 100:15.1f}%" for k in keys)
        ranked = keys[0] if args.rank_by == "worst" else args.rank_by
        print(
            f"{row['visit']:{width}s}{cells}   "
            f"{row[f'{ranked}_shape']} {row[f'{ranked}_shape_rate'] * 100:.0f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
