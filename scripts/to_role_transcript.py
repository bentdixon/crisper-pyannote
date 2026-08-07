"""Convert our pipeline's output into the role-labelled form the LLM review expects.

The review prompt is written around INTERVIEWER and PARTICIPANT, and its
speaker_flags and role_mapping_check categories only make sense against those
labels. Our pipeline emits pyannote-style SPEAKER_00/SPEAKER_01, so before the
same review can be applied to it, each speaker has to be mapped to a role.

The mapping uses the other team's own deterministic question-rate heuristic
(baseline/transcription_core.heuristic_assign_roles), so the two systems get
labelled by identical logic and the LLM comparison isolates the LLM rather
than the labelling method.

Writes, per visit, the two files apply_llm_corrections.py consumes:
    {stem}_transcript.txt   "[start - end] ROLE: text" turns
    {stem}_words.json       words carrying roles instead of SPEAKER_ ids

Usage:
    uv run python scripts/to_role_transcript.py --inputs outputs/ours \
        --output-dir outputs/ours_roles
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

from transcription_core import apply_role_mapping, heuristic_assign_roles  # noqa: E402

from crisper_pipeline import merge  # noqa: E402

logger = logging.getLogger("to_role_transcript")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="run_cohort.py output tree")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    inputs = Path(args.inputs)
    output_root = Path(args.output_dir)

    transcripts = sorted(inputs.rglob("transcript.json"))
    if args.limit:
        transcripts = transcripts[: args.limit]
    logger.info("Converting %d transcript(s)", len(transcripts))

    converted = 0
    for path in transcripts:
        payload = json.loads(path.read_text())
        words = payload.get("words") or []
        if not words:
            continue

        # group_into_turns wants "word"/"start"/"end"/"speaker"; the role
        # heuristic wants turns whose "words" are text fragments.
        turns = merge.group_into_turns(words)
        heuristic_turns = [
            {"speaker": t["speaker"], "words": [" " + w["word"].strip() for w in t["words"]]}
            for t in turns
        ]
        mapping, _evidence = heuristic_assign_roles(heuristic_turns)
        apply_role_mapping(turns, words, mapping)

        stem = Path(payload.get("audio") or path.parent.name).stem
        destination = output_root / path.parent.relative_to(inputs)
        destination.mkdir(parents=True, exist_ok=True)

        lines = []
        for turn in turns:
            start = f"{turn['start']:.2f}" if turn["start"] is not None else "?"
            end = f"{turn['end']:.2f}" if turn["end"] is not None else "?"
            text = " ".join(w["word"].strip() for w in turn["words"]).strip()
            lines.append(f"[{start} - {end}] {turn['speaker']}: {text}")
        (destination / f"{stem}_transcript.txt").write_text("\n".join(lines))
        (destination / f"{stem}_words.json").write_text(
            json.dumps(
                [
                    {"word": w["word"], "start": w["start"], "end": w["end"], "speaker": w["speaker"]}
                    for w in words
                ],
                indent=2,
            ) + "\n"
        )
        converted += 1

    logger.info("Wrote %d role-labelled transcript(s) to %s", converted, output_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
