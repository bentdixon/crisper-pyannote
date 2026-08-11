"""Write the evaluation results as CSV tables alongside the JSON.

The JSON files are the record; these are for reading, citing and pasting into a
document. One row per system, columns named the way the report names them, and
systems named in full rather than by their internal keys.

Usage:
    uv run python scripts/export_tables.py --output-dir reports/latest/tables \
        --results results.json --taxonomy taxonomy.json \
        --redaction redaction.json --partner partner_wer.json \
        --mono results_mono.json --stereo results_stereo.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import systems as registry  # noqa: E402

# (column heading, key in the aggregate entry, format)
METRIC_COLUMNS = [
    ("Visits", "visits", "int"),
    ("WER", "WER", "pct"),
    ("WER substitutions", "WER_sub", "pct"),
    ("WER deletions", "WER_del", "pct"),
    ("WER insertions", "WER_ins", "pct"),
    ("WER excluding insertions", "WER_no_ins", "pct"),
    ("sWER", "sWER", "pct"),
    ("DER", "DER", "pct"),
    ("DER confusion", "DER_confusion", "pct"),
    ("QTP-F1", "QTP_F1", "pct"),
]

REDACTION_COLUMNS = [
    ("Visits", "visits", "int"),
    ("Gold spans", "gold_spans", "int"),
    ("Spans redacted", "predicted_spans", "int"),
    ("True positives", "true_positives", "int"),
    ("False positives", "false_positives", "int"),
    ("False negatives", "false_negatives", "int"),
    ("Recall (sensitivity)", "recall", "pct"),
    ("Precision", "precision", "pct"),
    ("F1", "f1", "pct"),
    ("Under-redaction", "under_rate", "pct"),
    ("Over-redaction", "over_rate", "pct"),
    ("Leak rate", "leak_rate", "pct"),
]

PARTNER_COLUMNS = [
    ("Visits", "visits", "int"),
    ("WER raw", "raw", "raw_pct"),
    ("WER normalized", "normalized", "raw_pct"),
    ("WER filler-normalized", "filler_normalized", "raw_pct"),
    ("Median visit", "filler_normalized_median", "raw_pct"),
    ("Words per reference word", "word_ratio", "ratio"),
]


def render(value, kind: str) -> str:
    if value is None:
        return ""
    if kind == "int":
        return str(value)
    if kind == "pct":
        return f"{value * 100:.2f}"
    if kind == "raw_pct":
        return f"{value:.2f}"
    return f"{value:.4f}"


def write_table(path: Path, columns, aggregate: dict, order: list[str] | None = None) -> int:
    keys = order or list(aggregate)
    rows = []
    for key in keys:
        entry = registry.entry_of(aggregate, key)
        if not entry:
            continue
        row = {"System": registry.label_of(key)}
        for heading, field, kind in columns:
            row[heading] = render(entry.get(field), kind)
        rows.append(row)
    if not rows:
        return 0
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["System"] + [c[0] for c in columns])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def load(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text()) if path else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results", default=None)
    parser.add_argument("--mono", default=None)
    parser.add_argument("--stereo", default=None)
    parser.add_argument("--taxonomy", default=None)
    parser.add_argument("--redaction", default=None)
    parser.add_argument("--partner", default=None)
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    order = [key for key, *_ in registry.SYSTEMS]
    written = []

    for label, path in [
        ("metrics-all-visits", args.results),
        ("metrics-mono-files", args.mono),
        ("metrics-stereo-container-files", args.stereo),
    ]:
        data = load(path)
        if data:
            count = write_table(
                out / f"{label}.csv", METRIC_COLUMNS, data.get("aggregate", {}), order,
            )
            written.append((f"{label}.csv", count))

    taxonomy = load(args.taxonomy)
    if taxonomy:
        categories = next(iter(taxonomy.values()))["rates"].keys()
        columns = [("WER reconstructed", "wer_reconstructed", "pct")]
        rows = []
        for key in order:
            entry = registry.entry_of(taxonomy, key)
            if not entry:
                continue
            row = {"System": registry.label_of(key)}
            row["WER"] = render(entry["wer_reconstructed"], "pct")
            for category in categories:
                row[category] = render(entry["rates"][category], "pct")
            rows.append(row)
        if rows:
            with open(out / "wer-error-types.csv", "w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["System", "WER"] + list(categories),
                )
                writer.writeheader()
                writer.writerows(rows)
            written.append(("wer-error-types.csv", len(rows)))
        del columns

    redaction = load(args.redaction)
    if redaction:
        count = write_table(
            out / "pii-redaction.csv", REDACTION_COLUMNS,
            redaction.get("aggregate", {}), order,
        )
        written.append(("pii-redaction.csv", count))

    partner = load(args.partner)
    if partner:
        count = write_table(
            out / "partner-wer.csv", PARTNER_COLUMNS, partner.get("aggregate", {}), order,
        )
        written.append(("partner-wer.csv", count))

    for name, count in written:
        print(f"  {name}  ({count} systems)")
    print(f"\n{len(written)} table(s) in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
