"""Score every transcription system against the human transcripts.

Implements the evaluation recipe from DIALOG-DeID (Oliveira et al., "Role and
Privacy Aware Transcription for Clinical Interviews Beyond WER"):

  WER      pooled transcript quality, computed after removing filled pauses
  sWER     speaker-attributed WER under permutation-invariant stream
           alignment -- reference and predicted streams are matched by a
           Hungarian assignment on their time-overlap matrix, then WER is
           averaged over reference streams:
               sWER = (1/|S|) sum_s WER(T_s, T_pi(s))
           This is the "word misattribution" number: a system can score a
           good pooled WER and still put the words on the wrong speaker.
  DER      diarization error rate over the same reference turns
  QTP-F1   Qualifier/Temporal Preservation -- per stream, whether the cue
           TYPES present in the reference (negation, modality/conviction,
           temporal anchors) survive into the matched predicted stream

On PSYCHS the streams are roles, matching the paper; here the reference
streams come from the human transcript's own speaker labels (S1/S2), and
predicted streams are whatever labels the system emits, so the Hungarian
step handles the naming difference.

Systems are supplied as adapters that turn a system's output for one visit
into a common word list: [{"word", "start", "end", "speaker"}, ...].

Usage:
    uv run python scripts/evaluate_systems.py --cohort /path/to/cohort \
        --system chirp3 --system ours=outputs/ours \
        --system verbatimize=outputs/verbatimize \
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))

import jiwer  # noqa: E402
import numpy as np  # noqa: E402
from prepare_data import load_timestamped_text, normalize_text  # noqa: E402
from pyannote.core import Annotation, Segment  # noqa: E402
from pyannote.metrics.diarization import DiarizationErrorRate  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from crisper_pipeline import chirp as chirp_reader  # noqa: E402

logger = logging.getLogger("evaluate_systems")

# --- text normalization -----------------------------------------------------
# The paper computes WER "after removing filled pauses"; CrisperWhisper writes
# them as bracketed events ([UM]) and Chirp/humans as bare words (um, uh).
FILLED_PAUSES = {
    "um", "uh", "umm", "uhh", "mm", "mmm", "hmm", "hm", "mhm", "mmhmm",
    "uhhuh", "er", "erm", "ah", "eh", "huh",
}
BRACKETED = re.compile(r"\[[^\]]*\]")


def normalize(text: str) -> str:
    text = BRACKETED.sub(" ", text).lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in FILLED_PAUSES]
    return " ".join(tokens)


# --- QTP-F1 trigger lists (paper, footnote 5) -------------------------------
NEGATION = {
    "no", "not", "never", "none", "nothing", "nowhere", "neither", "without",
    "cannot", "can't", "don't", "doesn't", "didn't", "isn't", "aren't",
    "wasn't", "weren't", "won't", "wouldn't", "shouldn't", "couldn't",
    "hasn't", "haven't", "hadn't",
}
MODALITY = {
    "maybe", "probably", "possibly", "definitely", "certainly", "unsure",
    "sure", "i think", "i guess", "i feel like", "kind of", "sort of",
    "almost", "likely", "unlikely", "seems", "seems like",
}
TEMPORAL = {
    "today", "yesterday", "tomorrow", "tonight", "now", "recently", "lately",
    "currently", "earlier", "later", "last week", "last month", "last year",
    "past week", "past month", "past year", "for months", "for years",
    "in weeks", "in months",
}
TRIGGERS = {"negation": NEGATION, "modality": MODALITY, "temporal": TEMPORAL}


def cue_types(text: str) -> set[str]:
    """Trigger types present in a stream. Multiword triggers match as phrases.

    Apostrophes are kept by `normalize`, so "don't" survives; filled pauses
    are already gone, which does not affect any trigger.
    """
    found: set[str] = set()
    padded = f" {text} "
    for family, triggers in TRIGGERS.items():
        for trigger in triggers:
            if " " in trigger:
                if f" {trigger} " in padded:
                    found.add(f"{family}:{trigger}")
            elif f" {trigger} " in padded:
                found.add(f"{family}:{trigger}")
    return found


def qtp_score(reference_text: str, hypothesis_text: str) -> float:
    reference_cues = cue_types(reference_text)
    hypothesis_cues = cue_types(hypothesis_text)
    if not reference_cues and not hypothesis_cues:
        return 1.0
    if not reference_cues and hypothesis_cues:
        return 0.0
    overlap = len(reference_cues & hypothesis_cues)
    return 2 * overlap / (len(reference_cues) + len(hypothesis_cues))


# --- stream construction ----------------------------------------------------

def reference_streams(turns: list[dict]) -> dict[str, dict]:
    """Human transcript -> {speaker: {"text", "spans"}}."""
    streams: dict[str, dict] = {}
    for turn in turns:
        entry = streams.setdefault(turn["speaker"], {"text": [], "spans": []})
        entry["text"].append(normalize_text(turn["text"]))
        end = turn["end"] if turn["end"] is not None else turn["start"]
        entry["spans"].append((turn["start"], max(end, turn["start"])))
    return {k: {"text": " ".join(v["text"]), "spans": v["spans"]} for k, v in streams.items()}


def predicted_streams(words: list[dict]) -> dict[str, dict]:
    """System word list -> {speaker: {"text", "spans"}}."""
    streams: dict[str, dict] = {}
    for word in words:
        if word.get("start") is None or word.get("end") is None:
            continue
        entry = streams.setdefault(word.get("speaker") or "UNKNOWN", {"text": [], "spans": []})
        entry["text"].append(word["word"])
        entry["spans"].append((float(word["start"]), max(float(word["end"]), float(word["start"]))))
    return {k: {"text": " ".join(v["text"]), "spans": v["spans"]} for k, v in streams.items()}


def diarization_spans(words: list[dict]) -> dict[str, dict]:
    """Word list -> speaker spans built by the reference's own tiling rule.

    DER compares a reference whose turn ends were synthesized from the next
    turn's start, so the reference tiles the recording and declares no
    non-speech at all. Scoring raw word spans against that reference charges
    every inter-word silence as missed detection: measured over six sessions,
    false alarm was exactly 0.000 (nothing can be a false alarm when the
    reference calls everything speech) and missed detection accounted for
    0.41-0.63 of a 0.54-0.76 DER, while real speaker confusion was 0.10-0.27.
    That DER ranked systems by how much of the timeline their segments covered,
    not by whether they attributed words to the right speaker.

    So the hypothesis is built the same way the reference was: consecutive
    same-speaker words group into a turn, and each turn extends to the start of
    the following word. Both sides then tile the timeline, and DER can only
    move on label disagreement.
    """
    usable = [
        w for w in words
        if w.get("start") is not None and w.get("end") is not None
    ]
    if not usable:
        return {}
    usable = sorted(usable, key=lambda w: float(w["start"]))

    streams: dict[str, dict] = {}
    index = 0
    while index < len(usable):
        speaker = usable[index].get("speaker") or "UNKNOWN"
        start = float(usable[index]["start"])
        end = max(float(usable[index]["end"]), start)
        while index + 1 < len(usable) and (usable[index + 1].get("speaker") or "UNKNOWN") == speaker:
            index += 1
            end = max(end, float(usable[index]["end"]))
        if index + 1 < len(usable):
            end = max(end, float(usable[index + 1]["start"]))
        streams.setdefault(speaker, {"text": "", "spans": []})["spans"].append((start, end))
        index += 1
    return streams


def overlap_seconds(a_spans, b_spans) -> float:
    """Total temporal overlap between two sets of intervals."""
    if not a_spans or not b_spans:
        return 0.0
    a = sorted(a_spans)
    b = sorted(b_spans)
    total = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def align_streams(reference: dict[str, dict], hypothesis: dict[str, dict]) -> dict[str, str | None]:
    """Hungarian assignment maximizing total reference/hypothesis overlap."""
    ref_keys = sorted(reference)
    hyp_keys = sorted(hypothesis)
    if not ref_keys or not hyp_keys:
        return {k: None for k in ref_keys}

    matrix = np.zeros((len(ref_keys), len(hyp_keys)))
    for i, r in enumerate(ref_keys):
        for j, h in enumerate(hyp_keys):
            matrix[i, j] = overlap_seconds(reference[r]["spans"], hypothesis[h]["spans"])

    rows, cols = linear_sum_assignment(-matrix)
    mapping: dict[str, str | None] = {k: None for k in ref_keys}
    for i, j in zip(rows, cols):
        mapping[ref_keys[i]] = hyp_keys[j]
    return mapping


# --- metrics ----------------------------------------------------------------

def to_annotation(spans_by_speaker: dict[str, dict]) -> Annotation:
    annotation = Annotation()
    for speaker, entry in spans_by_speaker.items():
        for start, end in entry["spans"]:
            if end > start:
                annotation[Segment(start, end)] = speaker
    return annotation


def score_visit(turns: list[dict], words: list[dict]) -> dict | None:
    reference = reference_streams(turns)
    hypothesis = predicted_streams(words)
    if not reference or not hypothesis:
        return None

    reference_text = normalize(" ".join(v["text"] for v in reference.values()))
    hypothesis_text = normalize(" ".join(w["word"] for w in words))
    if not reference_text or not hypothesis_text:
        return None

    pooled = jiwer.process_words(reference_text, hypothesis_text)

    mapping = align_streams(reference, hypothesis)
    stream_wers = []
    stream_qtps = []
    for speaker, matched in mapping.items():
        ref_text = normalize(reference[speaker]["text"])
        hyp_text = normalize(hypothesis[matched]["text"]) if matched else ""
        if not ref_text:
            continue
        # An unmatched reference stream is a total miss, not a skip.
        stream_wers.append(jiwer.process_words(ref_text, hyp_text or "*").wer if hyp_text else 1.0)
        stream_qtps.append(qtp_score(ref_text, hyp_text))

    # DER on granularity-matched spans (see diarization_spans), reported with
    # its confusion component broken out: confusion alone is the pure
    # speaker-attribution error, free of any speech/non-speech disagreement.
    tiled = diarization_spans(words)
    reference_annotation = to_annotation(reference)
    der = der_confusion = der_word_level = None
    if tiled:
        components = DiarizationErrorRate(collar=0.25, skip_overlap=False)(
            reference_annotation, to_annotation(tiled), detailed=True
        )
        total = components["total"]
        if total:
            der = float(components["diarization error rate"])
            der_confusion = float(components["confusion"] / total)
        # The old word-span number, kept so the change is auditable.
        der_word_level = float(
            DiarizationErrorRate(collar=0.25, skip_overlap=False)(
                reference_annotation, to_annotation(hypothesis)
            )
        )

    # WER split into its three error types, each as a rate over reference words
    # so they sum to WER. This is what makes the column readable here: the ASR
    # is verbatim and the reference semi-verbatim, so every system emits ~1.33
    # words per reference word and insertions carry most of the total. Without
    # the split, "transcribed a filler the human omitted" is indistinguishable
    # from "misheard the word".
    reference_length = len(reference_text.split()) or 1
    substitutions = pooled.substitutions / reference_length
    deletions = pooled.deletions / reference_length
    insertions = pooled.insertions / reference_length

    return {
        "wer": pooled.wer,
        "wer_sub": substitutions,
        "wer_del": deletions,
        "wer_ins": insertions,
        # WER with insertions removed: how much of the reference was actually
        # got wrong or missed, unaffected by the verbatim/semi-verbatim gap.
        "wer_no_ins": substitutions + deletions,
        "swer": float(np.mean(stream_wers)) if stream_wers else None,
        "der": der,
        "der_confusion": der_confusion,
        "der_word_level": der_word_level,
        "qtp_f1": float(np.mean(stream_qtps)) if stream_qtps else None,
        "ref_words": len(reference_text.split()),
        "hyp_words": len(hypothesis_text.split()),
        "ref_streams": len(reference),
        "hyp_streams": len(hypothesis),
    }


# --- system adapters --------------------------------------------------------

def load_chirp(visit: Path, _root: Path | None = None, _relative: Path | None = None) -> list[dict] | None:
    found = sorted((visit / "chirp").glob("*.json"))
    if not found:
        return None
    return chirp_reader.load_transcript(found[0])["words"]


def _pipeline_words(root: Path, relative: Path, pattern: str) -> list[dict] | None:
    """Newest run directory for a visit under a transcribe/verbatimize tree."""
    candidates = sorted((root / relative).glob(pattern))
    if not candidates:
        candidates = sorted(root.glob(f"{relative.as_posix()}/*/{pattern}"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text())["words"]


def make_run_adapter(pattern: str):
    def adapter(visit: Path, root: Path, relative: Path | None = None):
        return _pipeline_words(root, relative, pattern)
    return adapter


def make_file_adapter(suffix: str):
    def adapter(visit: Path, root: Path, relative: Path | None = None):
        found = sorted((root / relative).glob(f"*{suffix}"))
        if not found:
            return None
        return json.loads(found[-1].read_text())
    return adapter


ADAPTERS = {
    "chirp3": (load_chirp, None),
    "ours": (make_run_adapter("transcript.json"), "transcript.json"),
    "verbatimize": (make_run_adapter("transcript.json"), "transcript.json"),
    "baseline": (make_file_adapter("_words.json"), "_words.json"),
    "baseline_llm": (make_file_adapter("_words_corrected.json"), "_words_corrected.json"),
    "ours_llm": (make_file_adapter("_words_corrected.json"), "_words_corrected.json"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument(
        "--system", action="append", required=True, metavar="NAME[=DIR]",
        help=f"one of {sorted(ADAPTERS)}; all but chirp3 need an output directory",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cohort = Path(args.cohort)

    systems = []
    for spec in args.system:
        name, _, directory = spec.partition("=")
        if name not in ADAPTERS:
            logger.error("Unknown system %s; known: %s", name, sorted(ADAPTERS))
            return 1
        systems.append((name, ADAPTERS[name][0], Path(directory) if directory else None))

    visits = sorted(
        p for p in cohort.glob("Pronet*/*/*")
        if (p / "human").is_dir() and any((p / "human").glob("*.txt"))
        and any((p / "audio").glob("*.wav"))
    )
    if args.limit:
        visits = visits[: args.limit]
    logger.info("Scoring %d system(s) over %d visit(s)", len(systems), len(visits))

    per_visit: dict[str, dict] = {}
    scores: dict[str, list[dict]] = {name: [] for name, _, _ in systems}
    adapter_errors: dict[str, Counter] = {name: Counter() for name, _, _ in systems}
    missing: Counter = Counter()

    for index, visit in enumerate(visits, start=1):
        relative = visit.relative_to(cohort)
        human = sorted((visit / "human").glob("*.txt"))[0]
        audio = sorted((visit / "audio").glob("*.wav"))
        duration = 0.0
        if audio:
            try:
                import wave

                with wave.open(str(audio[0])) as handle:
                    duration = handle.getnframes() / handle.getframerate()
            except Exception:
                duration = 0.0
        try:
            turns = load_timestamped_text(human, duration)
        except Exception:
            logger.warning("  unparseable human transcript: %s", relative)
            continue

        for name, adapter, root in systems:
            # A bare "except -> no data" here once hid a TypeError in an
            # adapter signature, and the system silently scored zero visits
            # while looking merely absent. Failures are counted and reported.
            try:
                words = adapter(visit, root, relative)
            except Exception as error:
                adapter_errors[name][type(error).__name__ + f": {error}"] += 1
                words = None
            if not words:
                if words is None:
                    missing[name] += 1
                continue
            result = score_visit(turns, words)
            if result is None:
                continue
            result["visit"] = relative.as_posix()
            scores[name].append(result)
            per_visit.setdefault(relative.as_posix(), {})[name] = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in result.items() if k != "visit"
            }
        if index % 10 == 0 or index == len(visits):
            logger.info("  %d/%d visits", index, len(visits))

    for name in adapter_errors:
        if adapter_errors[name]:
            logger.error(
                "%s: adapter raised on %d visit(s): %s",
                name, sum(adapter_errors[name].values()),
                dict(adapter_errors[name].most_common(3)),
            )
        if not scores[name]:
            logger.error(
                "%s: scored 0 visits (%d with no output) -- not a result, a failure",
                name, missing[name],
            )

    def mean(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    aggregate = {}
    for name, results in scores.items():
        if not results:
            continue
        aggregate[name] = {
            "visits": len(results),
            "WER": mean([r["wer"] for r in results]),
            "WER_sub": mean([r.get("wer_sub") for r in results]),
            "WER_del": mean([r.get("wer_del") for r in results]),
            "WER_ins": mean([r.get("wer_ins") for r in results]),
            "WER_no_ins": mean([r.get("wer_no_ins") for r in results]),
            "sWER": mean([r["swer"] for r in results]),
            "DER": mean([r["der"] for r in results]),
            "DER_confusion": mean([r.get("der_confusion") for r in results]),
            "DER_word_level": mean([r.get("der_word_level") for r in results]),
            "no_timestamps": sum(1 for r in results if r["der"] is None),
            "QTP_F1": mean([r["qtp_f1"] for r in results]),
            "ref_words": sum(r["ref_words"] for r in results),
            "hyp_words": sum(r["hyp_words"] for r in results),
        }

    print(
        f"\n  {'system':16s} {'visits':>6} {'WER':>8} {'sub':>7} {'del':>7} {'ins':>7} "
        f"{'WERnoIns':>9} {'sWER':>8} {'DER':>8} {'DERconf':>8} {'QTP-F1':>8}"
    )
    for name, stats in aggregate.items():
        def show(key, spec=".4f"):
            return format(stats[key], spec) if stats.get(key) is not None else "-"
        print(
            f"  {name:16s} {stats['visits']:6d} {show('WER'):>8} {show('WER_sub'):>7} "
            f"{show('WER_del'):>7} {show('WER_ins'):>7} {show('WER_no_ins'):>9} "
            f"{show('sWER'):>8} {show('DER'):>8} {show('DER_confusion'):>8} "
            f"{show('QTP_F1'):>8}"
        )

    if args.output:
        Path(args.output).write_text(
            json.dumps({"aggregate": aggregate, "per_visit": per_visit}, indent=2) + "\n"
        )
        logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
