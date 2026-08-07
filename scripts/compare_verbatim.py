"""Diagnostic: how much verbatim content is Chirp-3 actually missing?

Verbatimize only pays off if the Chirp-3 transcript is meaningfully cleaner
than the audio. This measures that gap directly, per session, by counting
disfluency tokens in three transcripts of the same audio:

    chirp        the Chirp-3 transcript as stored in the bucket
    verbatimize  the output of verbatimize-session
    verbatim     a from-scratch CrisperWhisper verbatim transcription

If `verbatim` finds far more disfluencies than `verbatimize` produced, the
verbatimize prompt is under-inserting. If `chirp` is already close to
`verbatim`, the Chirp transcripts were never clean to begin with and there is
little to recover.

Only counts and rates are printed -- never transcript text -- so this is safe
to run against clinical audio and read off a terminal.

Usage:
    uv run python scripts/compare_verbatim.py --audio-dir testdata/audio \
        --chirp-dir testdata/chirp --verbatimize-dir testdata/out2
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from crisperwhisper.forced_align import default_normalize

from crisper_pipeline import asr, chirp
from crisper_pipeline.verbatimize_cli import normalize_stem

logger = logging.getLogger("compare_verbatim")

FILLERS = {
    "um", "uh", "mm", "hmm", "mhm", "mmhmm", "uhhuh", "er", "ah", "eh",
    "huh", "hm",
}
# CrisperWhisper marks vocal events with bracketed tokens, e.g. [UM], [laughter].
BRACKETED = re.compile(r"^\[.+\]$")


def disfluency_counts(words: list[str]) -> dict[str, int]:
    filler = sum(1 for w in words if default_normalize(w) in FILLERS)
    events = sum(1 for w in words if BRACKETED.match(w.strip()))
    return {"words": len(words), "fillers": filler, "events": events}


def rate(counts: dict[str, int]) -> float:
    return counts["fillers"] / counts["words"] if counts["words"] else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--chirp-dir", required=True)
    parser.add_argument("--verbatimize-dir", required=True, help="verbatimize-session output dir")
    parser.add_argument("--model", default="large")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--language", default="en")
    parser.add_argument("--output", default=None, help="write results JSON here")
    return parser


def index_by_session(directory: Path, pattern: str) -> dict[str, Path]:
    return {normalize_stem(p.stem): p for p in sorted(directory.rglob(pattern))}


def latest_verbatimize_outputs(directory: Path) -> dict[str, Path]:
    """Map session key -> newest transcript.json under a verbatimize output dir."""
    found: dict[str, Path] = {}
    for path in sorted(directory.glob("*/transcript.json")):
        key = normalize_stem(re.sub(r"_\d{8}-\d{6}(-\d+)?$", "", path.parent.name))
        found[key] = path  # sorted order means the newest run wins
    return found


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    chirp_index = index_by_session(Path(args.chirp_dir), "*.json")
    verb_index = latest_verbatimize_outputs(Path(args.verbatimize_dir))
    audio_files = sorted(Path(args.audio_dir).glob("*.wav"))
    logger.info("Comparing %d audio file(s)", len(audio_files))

    model = asr.load_model(
        args.model, draft_model=None,
        compute_type=args.compute_type, device_index=args.device_index,
    )

    results = []
    for audio in audio_files:
        key = normalize_stem(audio.stem)
        if key not in chirp_index or key not in verb_index:
            logger.warning("Missing chirp or verbatimize output for %s; skipping", audio.name)
            continue

        source = chirp.load_transcript(chirp_index[key])
        chirp_counts = disfluency_counts([w["word"] for w in source["words"]])

        produced = json.loads(verb_index[key].read_text())
        verbatimize_counts = disfluency_counts([w["word"] for w in produced["words"]])

        logger.info("Transcribing %s from scratch in verbatim mode", audio.name)
        scratch = asr.transcribe(model, audio, language=args.language)
        verbatim_counts = disfluency_counts([w["word"] for w in scratch["words"]])

        row = {
            "session": audio.stem,
            "duration": source["duration"],
            "chirp": chirp_counts,
            "verbatimize": verbatimize_counts,
            "verbatim": verbatim_counts,
        }
        results.append(row)
        logger.info(
            "%s | words c/v/s %d/%d/%d | fillers %d/%d/%d | rate %.3f/%.3f/%.3f",
            audio.stem[:28],
            chirp_counts["words"], verbatimize_counts["words"], verbatim_counts["words"],
            chirp_counts["fillers"], verbatimize_counts["fillers"], verbatim_counts["fillers"],
            rate(chirp_counts), rate(verbatimize_counts), rate(verbatim_counts),
        )

    if results:
        totals = {
            source: {
                field: sum(r[source][field] for r in results)
                for field in ("words", "fillers", "events")
            }
            for source in ("chirp", "verbatimize", "verbatim")
        }
        print("\n  source        words   fillers   events   filler rate")
        for source, counts in totals.items():
            print(
                f"  {source:12s} {counts['words']:7d} {counts['fillers']:9d} "
                f"{counts['events']:8d} {rate(counts):12.4f}"
            )
        if args.output:
            Path(args.output).write_text(
                json.dumps({"per_session": results, "totals": totals}, indent=2) + "\n"
            )
            logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
