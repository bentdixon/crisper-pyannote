"""Render the five-system evaluation results as a standalone HTML page.

Reads the results.json written by evaluate_systems.py and emits a self-contained
page: DM Sans embedded as a data URI, hand-built SVG bar charts, no external
requests (the artifact host blocks them).

Two things this does beyond plotting the aggregate means:

  common subset   the sweeps cover different numbers of visits, so a
                  mean-vs-mean table compares systems on different audio.
                  Every metric is therefore also computed over the visits
                  where *all* systems produced output, which is the only
                  like-for-like comparison.
  per-visit IQR   whiskers show the p25-p75 spread across visits, so a gap
                  between two bars can be read against the variance behind it.

Usage:
    uv run python scripts/plot_results.py results.json --output report.html
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import systems as registry  # noqa: E402

# --- design tokens (from design_language.md, Palette 1) ---------------------
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
TEXT = "#0b0b0b"
MUTED = "#5a5a56"
MUTED_DARK = "#3a3a37"
WINNER = "#1baf7a"
# Ink for legend swatches that key an opacity ramp rather than a system.
# Near-black fades to a row of greys, which reads as "no colour here" and
# not as "these are the same colour at three strengths".
RAMP_INK = "#3f4a7a"

# The system registry lives in systems.py so a name is defined once and
# rendered the same way in every console table, JSON file, CSV and chart.
SYSTEMS = registry.SYSTEMS

# (title, per-visit key, aggregate key, direction, caption)
# Captions say what the number means in words, because every metric here is a
# rate between 0 and 1 rendered as a percentage and they are easy to confuse.
METRICS = [
    (
        "WER", "wer", "WER", "lower is better",
        "of every 100 reference words, this many errors -- including words the "
        "machine heard that the human transcript omits",
    ),
    (
        "WER excluding insertions", "wer_no_ins", "WER_no_ins", "lower is better",
        "reference words got wrong or missed; unaffected by the verbatim vs "
        "semi-verbatim mismatch, so this is the fair transcription-quality number",
    ),
    (
        "sWER", "swer", "sWER", "lower is better",
        "word error rate after matching speakers, so it counts words put on the "
        "wrong speaker as well as words got wrong",
    ),
    (
        "DER", "der", "DER", "lower is better",
        "share of speech time attributed to the wrong speaker, including "
        "speech/non-speech boundary disagreement",
    ),
    (
        "DER confusion", "der_confusion", "DER_confusion", "lower is better",
        "share of speech time on the wrong speaker, boundary disagreement "
        "excluded -- the cleanest speaker-attribution number here",
    ),
    (
        "QTP-F1", "qtp_f1", "QTP_F1", "higher is better",
        "how much negation, modality and temporal cue information survives "
        "into each speaker's transcript",
    ),
]

# The partner team's metric is appended only when their results file is passed,
# so the deck degrades to the six original charts without it.
PARTNER_METRIC = (
    "Partner WER", "filler_normalized", "PartnerWER", "lower is better",
    "the partner team's own WER code, run unmodified over the same visits -- "
    "an independent check on the ordering the charts above report",
)


# A third WER implementation, added the same way: jiwer with square-bracket
# content and punctuation stripped. It is the standard edit distance, unlike the
# partner metric's difflib upper bound, so it lands between ours and theirs.
JIWER_METRIC = (
    "jiwer WER", "jiwer_wer", "JiwerWER", "lower is better",
    "a third WER implementation -- jiwer, brackets and punctuation stripped -- "
    "run unmodified over the same visits",
)


def metrics_for(aggregate: dict) -> list[tuple[str, str, str, str, str]]:
    """METRICS, plus each optional metric any system carries."""
    metrics = list(METRICS)
    for metric in (PARTNER_METRIC, JIWER_METRIC):
        if any(entry.get(metric[2]) is not None for entry in aggregate.values()):
            metrics.append(metric)
    return metrics

# Full figure captions, keyed by aggregate key. These travel with the exported
# charts, so each one has to stand on its own: what the metric is, how it is
# computed here, and the caveat that decides how far it can be trusted.
CAPTIONS = {
    "WER": (
        "Substitutions plus deletions plus insertions, divided by the number of "
        "reference words, after filled pauses are removed from both sides. WER is "
        "not capped at 100%: a system that inserts more words than the reference "
        "contains scores above it. Scoring is restricted to the span each human "
        "transcript actually covers -- they stop early on a third of this cohort, and "
        "including the untranscribed remainder charged every system for minutes the "
        "reference never described, inflating WER from roughly 12% to roughly 42%. "
        "Within the covered span the machine now emits slightly fewer words than the "
        "reference, and deletions rather than insertions carry most of the total."
    ),
    "WER_no_ins": (
        "Substitutions plus deletions only: the share of reference words the machine "
        "got wrong or missed outright. Dropping insertions removes what remains of the "
        "verbatim versus semi-verbatim mismatch -- the machine keeps disfluencies and "
        "repetitions the transcriber deletes silently -- which is a convention "
        "difference rather than a transcription failure. This is the fairer measure of "
        "how well each system heard the words both sources agree were said."
    ),
    "sWER": (
        "Speaker-attributed word error rate. Reference and predicted speaker streams "
        "are matched by a Hungarian assignment on their time-overlap matrix, WER is "
        "computed within each reference stream, and the per-stream scores are averaged. "
        "Each stream is capped at 1.0 and reference streams under five words are "
        "discarded as transcript formatting artifacts. Both corrections matter: "
        "uncapped, a stray one-word speaker line in a human transcript matched against "
        "a large predicted stream scored 88.0 on a single visit and moved a system's "
        "corpus average by 32 points, so the metric was ranking systems by how many "
        "streams they emitted rather than by where they put the words."
    ),
    "DER": (
        "The share of speech time attributed to the wrong speaker, including "
        "disagreement about where speech starts and stops. Both sides are built the "
        "same way before scoring, because the human transcripts carry turn start "
        "times only and each turn's end is synthesized from the next turn's start. "
        "That synthesis charges boundary error to every system, so these values are "
        "not comparable with DER reported on properly annotated corpora -- only with "
        "each other."
    ),
    "DER_confusion": (
        "The confusion term of DER on its own: speech time given to the wrong "
        "speaker, with speech versus non-speech disagreement excluded. Since the "
        "reference's synthesized turn ends make the boundary term unreliable here, "
        "this is the cleanest speaker-attribution comparison on the page and the one "
        "to quote when asking which diarizer is better."
    ),
    "QTP_F1": (
        "Qualifier and Temporal Preservation F1. For each speaker stream the cue "
        "types present in the reference -- negation, modality or conviction, and "
        "temporal anchoring -- are compared with those surviving in the matched "
        "predicted stream, and the per-stream F1 scores are averaged. It asks whether "
        "clinically meaningful qualifiers survive transcription, which WER cannot "
        "distinguish from any other word. Higher is better, unlike every other "
        "chart here."
    ),
    "PartnerWER": (
        "The partner team's compareFiles.py, vendored unmodified and run over the same "
        "269 visits and the same system outputs. Their algorithm is not ours: alignment "
        "is difflib's SequenceMatcher rather than a minimum edit distance, so a replace "
        "block costs the longer of its two sides and the result is an upper bound on "
        "WER; and filled pauses are collapsed to a single token on both sides rather "
        "than deleted. It agrees with our ranking from independent code, which is what "
        "makes it worth showing. Note the unusually wide gap between the median dot and "
        "the mean marker: a tail of visits whose human transcript covers only part of "
        "the session scores above 100% and carries every mean on this chart."
    ),
    "JiwerWER": (
        "A third WER implementation, supplied by the study team and vendored unmodified: "
        "jiwer's edit distance after everything inside square brackets is deleted, all "
        "punctuation is stripped and both sides are lowercased. It is the standard "
        "minimum-edit-distance WER, so it reads below the partner team's difflib upper "
        "bound and close to ours, over the same covered spans and the same outputs. "
        "One asymmetry to know about: deleting bracketed content removes every "
        "CrisperWhisper filled pause, written [UM] and [UH], while Chirp-3's plain \"um\" "
        "survives, so the CrisperWhisper arms are charged deletions their transcripts "
        "did not earn. Removing filled pauses from both sides for every system widens "
        "their lead rather than narrowing it."
    ),
    "composition": (
        "The three error types that sum to WER, each as a rate over reference words. "
        "Insertions are words the machine heard that the human transcript omits, "
        "overwhelmingly disfluencies and repetitions; substitutions are words heard "
        "differently; deletions are reference words missed. The insertion share is "
        "why the WER figures sit near 85% for every system and why WER alone should "
        "not be read as transcription quality on this corpus."
    ),
}


def reading_line(name_values: dict[str, float | None], spreads: dict, direction: str) -> str:
    """A data-derived sentence naming the leader and whether the lead is safe.

    Generated rather than written so it cannot go stale against the numbers it
    sits beside, and so it states plainly when a visible gap is not supported
    by the per-visit spread.
    """
    ranked = [(n, v) for n, v in name_values.items() if v is not None]
    if len(ranked) < 2:
        return ""
    better_low = direction.startswith("lower")
    ranked.sort(key=lambda kv: kv[1], reverse=not better_low)
    best, runner = ranked[0], ranked[1]
    best_label = next((" + ".join(p) for n, p, _, _ in SYSTEMS if n == best[0]), best[0])
    runner_label = next((" + ".join(p) for n, p, _, _ in SYSTEMS if n == runner[0]), runner[0])

    best_spread, runner_spread = spreads.get(best[0]), spreads.get(runner[0])
    overlap = ""
    if best_spread and runner_spread:
        # Ranges overlap when neither sits entirely to one side of the other.
        if not (best_spread[2] < runner_spread[0] or runner_spread[2] < best_spread[0]):
            overlap = (
                " Their middle-50% ranges overlap, so this ordering is not "
                "established by these visits alone."
            )
    return (
        f"Best: {best_label} at {best[1] * 100:.1f}%, ahead of {runner_label} at "
        f"{runner[1] * 100:.1f}%.{overlap}"
    )


# Stacked composition of WER, which is the only figure here that is genuinely a
# part-of-a-whole and so the only one that earns a stacked bar.
WER_PARTS = [
    ("insertions", "WER_ins", "#eda100", "machine heard a word the human transcript omits"),
    ("substitutions", "WER_sub", "#2a78d6", "machine heard a different word"),
    ("deletions", "WER_del", "#e87ba4", "reference word the machine missed"),
]


def quartiles(values: list[float]) -> tuple[float, float, float] | None:
    """p25 / median / p75 of a per-visit metric, or None if too little data.

    The median matters here: several systems have means dragged well past their
    own p75 by a tail of catastrophic visits, so a bar showing only the mean
    would misrepresent the typical session.
    """
    clean = sorted(v for v in values if v is not None)
    if len(clean) < 4:
        return None
    quarters = statistics.quantiles(clean, n=4)
    return (quarters[0], quarters[1], quarters[2])


def collect(per_visit: dict, key: str) -> dict[str, list[float]]:
    """Per-system list of one metric's per-visit values."""
    out: dict[str, list[float]] = {name: [] for name, *_ in SYSTEMS}
    for visit in per_visit.values():
        for name in out:
            entry = visit.get(name)
            if entry and entry.get(key) is not None:
                out[name].append(entry[key])
    return out


def common_subset(per_visit: dict, present: list[str]) -> list[str]:
    """Visits where every scored system produced output."""
    return [
        visit for visit, entry in per_visit.items()
        if all(name in entry for name in present)
    ]


def subset_mean(per_visit: dict, visits: list[str], name: str, key: str) -> float | None:
    values = [
        per_visit[v][name][key]
        for v in visits
        if name in per_visit[v] and per_visit[v][name].get(key) is not None
    ]
    return sum(values) / len(values) if values else None


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load_results(path: str | None) -> dict | None:
    """Read a results file and key it by short identifier internally.

    Scorers write full system names now, and older files carry the short keys.
    Both are folded to the short key on the way in, so the rest of this module
    has one form to reason about and a scoring run from before the rename stays
    readable rather than silently rendering as an empty report.
    """
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    for section in ("aggregate", "per_visit"):
        block = data.get(section)
        if not isinstance(block, dict):
            continue
        if section == "aggregate":
            data[section] = {registry.key_of(k): v for k, v in block.items()}
        else:
            data[section] = {
                visit: {registry.key_of(k): v for k, v in entry.items()}
                for visit, entry in block.items()
            }
    return data


# A visit where every system emits more than this many words per reference word
# is a truncated human transcript, not a failing system: the systems disagree
# with each other far less than they all disagree with the reference.
TRUNCATION_RATIO = 2.0


def merge_partner(data: dict, partner: dict) -> dict:
    """Fold partner_wer.json into data's shape and summarise it.

    Their figures are percentages and every chart here multiplies by 100, so
    values are divided by 100 on the way in -- the alternative is a chart
    reading 4374%.
    """
    aggregate, per_visit = data.setdefault("aggregate", {}), data.setdefault("per_visit", {})
    p_agg, p_visit = partner.get("aggregate", {}), partner.get("per_visit", {})

    for name, entry in p_agg.items():
        if name in aggregate:
            aggregate[name]["PartnerWER"] = entry["filler_normalized"] / 100.0
    for visit, systems in p_visit.items():
        for name, entry in systems.items():
            target = per_visit.setdefault(visit, {}).setdefault(name, {})
            target["filler_normalized"] = entry["filler_normalized"] / 100.0

    # Which visits have a truncated reference: judged on the systems that
    # actually transcribe independently, so an LLM-corrected copy of another
    # system does not get a vote twice.
    independent = [n for n in ("ours", "baseline", "chirp3", "verbatimize") if n in p_agg]
    truncated = set()
    for visit, systems in p_visit.items():
        ratios = [
            systems[n]["hyp_words"] / max(systems[n]["ref_words"], 1)
            for n in independent if n in systems
        ]
        if ratios and len(ratios) == len(independent) and min(ratios) > TRUNCATION_RATIO:
            truncated.add(visit)

    summary = {"truncated": len(truncated), "total": len(p_visit), "systems": {}}
    for name, entry in p_agg.items():
        clean = [
            systems[name]["filler_normalized"]
            for visit, systems in p_visit.items()
            if name in systems and visit not in truncated
        ]
        summary["systems"][name] = {
            "raw": entry["raw"],
            "normalized": entry["normalized"],
            "filler": entry["filler_normalized"],
            "median": entry["filler_normalized_median"],
            "clean_mean": statistics.fmean(clean) if clean else None,
            "clean_median": statistics.median(clean) if clean else None,
        }
    return summary


def merge_jiwer(data: dict, jiwer_data: dict) -> dict:
    """Fold jiwer_wer.json into data's shape and summarise it.

    The per-visit key is `jiwer_wer`, not `wer`: our own per-visit WER already
    occupies `wer`, and overwriting it would silently redraw the headline chart
    with another implementation's numbers.
    """
    aggregate, per_visit = data.setdefault("aggregate", {}), data.setdefault("per_visit", {})
    j_agg, j_visit = jiwer_data.get("aggregate", {}), jiwer_data.get("per_visit", {})

    for name, entry in j_agg.items():
        if name in aggregate:
            # The mean of the per-visit rates, matching how every other metric
            # on this page aggregates -- the chart's hollow marker is labelled
            # "mean" in the legend. The pooled rate is in the section table.
            aggregate[name]["JiwerWER"] = entry["wer"]
    for visit, systems in j_visit.items():
        for name, entry in systems.items():
            target = per_visit.setdefault(visit, {}).setdefault(name, {})
            target["jiwer_wer"] = entry["wer"]

    summary = {"total": len(j_visit), "systems": {}}
    for name, entry in j_agg.items():
        summary["systems"][name] = {
            "pooled": entry["wer_micro"],
            "mean": entry["wer"],
            "median": entry["wer_median"],
            "sub": entry["substitutions_rate"],
            "delete": entry["deletions_rate"],
            "insert": entry["insertions_rate"],
            "no_fillers": entry["wer_micro_filler_symmetric"],
            "ratio": entry["word_ratio"],
        }
    return summary


# Systems whose speaker labels come from the frozen Chirp-3 bucket output and so
# could not be re-run under a forced-mono condition. Shown, but marked.
NOT_RERUN = {"chirp3", "verbatimize"}

# The two metrics the mono/stereo split is about. WER is left out on purpose:
# the condition changes diarization, and only incidentally the ASR input.
CONDITION_METRICS = [
    ("DER confusion", "DER_confusion", "lower is better"),
    ("sWER", "sWER", "lower is better"),
]


# The edit categories from error_taxonomy.py, ordered so the two bands read as
# blocks: what the machine got wrong, then what it said that the transcript does
# not record. Colours follow that split -- warm for real error, cool for the
# convention gap -- rather than giving seven unrelated hues.
TAXONOMY_PARTS = [
    ("different word", "#c0392b", "a genuine mishearing"),
    ("near-miss", "#e8834a", "same word, different spelling, inflection or number format"),
    ("deletion", "#eda100", "a reference word the machine did not produce"),
    ("repetition", "#7fb0e8", "the speaker said it twice; the transcriber wrote it once"),
    ("backchannel", "#5b8fd6", "yeah, okay, right -- dropped by the transcriber as noise"),
    ("discourse marker", "#2a78d6", "like, so, just, well, you know"),
    ("other insertion", "#1c4f8f", "speech present in the audio and absent from the transcript"),
]


def summary_section(aggregate: dict, redaction_data: dict | None,
                    taxonomy: dict | None) -> str:
    """The overview: who wins what, and the four readings that matter.

    Every figure is pulled from the data rather than written down, because a
    hand-typed summary is the first thing to go stale after a rescore -- and on
    this evaluation the numbers have moved by an order of magnitude twice.
    """
    if not aggregate:
        return ""

    def best(key: str, lower_is_better: bool = True) -> tuple[str, float] | None:
        scored = [
            (n, registry.entry_of(aggregate, n).get(key))
            for n, *_ in SYSTEMS
            if registry.entry_of(aggregate, n)
            and registry.entry_of(aggregate, n).get(key) is not None
        ]
        if not scored:
            return None
        return (min if lower_is_better else max)(scored, key=lambda kv: kv[1])

    cards = []
    for title, key, lower, note in [
        ("Word error rate", "WER", True, "best system, all 269 visits"),
        ("Speaker-attributed WER", "sWER", True, "words on the right speaker"),
        ("Speaker confusion", "DER_confusion", True, "share of speech time misattributed"),
        ("Cue preservation", "QTP_F1", False, "negation, modality, temporal"),
    ]:
        found = best(key, lower)
        if not found:
            continue
        name, value = found
        label = " + ".join(registry.PARTS.get(name, [name]))
        cards.append(
            f'<div class="stat"><span class="statlabel">{escape(title)}</span>'
            f'<span class="statvalue">{value * 100:.1f}%</span>'
            f'<span class="statnote">{escape(label)}</span>'
            f'<span class="statnote dim">{escape(note)}</span></div>'
        )

    findings = []
    ours = registry.entry_of(taxonomy or {}, "ours")
    if ours:
        rates = ours["rates"]
        misheard = (rates["different word"] + rates["near-miss"]) * 100
        missed = rates["deletion"] * 100
        findings.append(
            f"<b>Transcription accuracy is roughly {misheard:.0f}% of reference words "
            f"misheard and {missed:.0f}% missed.</b> Deletions, not insertions, are now "
            "the largest error term, so the improvement target is speech the systems "
            "are dropping rather than text they are inventing."
        )

    community = registry.entry_of(aggregate, "ours")
    pyannote = registry.entry_of(aggregate, "baseline")
    if community and pyannote:
        findings.append(
            "<b>The two diarizers are close, and split the honours.</b> "
            f"community-1 leads WER ({community['WER'] * 100:.1f}% against "
            f"{pyannote['WER'] * 100:.1f}%), sWER and cue preservation; pyannote 3.1 "
            f"leads speaker confusion ({pyannote['DER_confusion'] * 100:.1f}% against "
            f"{community['DER_confusion'] * 100:.1f}%). An earlier version of this page "
            "reported a 32-point sWER gap between them, which was a metric defect."
        )

    llm_pairs = [("ours", "ours_llm"), ("baseline", "baseline_llm")]
    if all(registry.entry_of(aggregate, a) and registry.entry_of(aggregate, b)
           for a, b in llm_pairs):
        findings.append(
            "<b>The LLM review never helps.</b> Applying the Qwen2.5-7B pass makes every "
            "metric equal or worse for both pipelines, on all 269 visits. It should be "
            "dropped."
        )

    if redaction_data:
        gemma = registry.entry_of(redaction_data.get("aggregate", {}), "ours_redacted")
        chirp = registry.entry_of(redaction_data.get("aggregate", {}), "chirp3")
        verb = registry.entry_of(redaction_data.get("aggregate", {}), "verbatimize")
        if gemma and chirp:
            findings.append(
                f"<b>Gemma 4 redaction beats Chirp-3's native redaction</b> on sensitivity "
                f"({gemma['recall'] * 100:.0f}% against {chirp['recall'] * 100:.0f}%), "
                f"precision and leak rate ({gemma['leak_rate'] * 100:.0f}% against "
                f"{chirp['leak_rate'] * 100:.0f}%), while redacting fewer spans."
            )
        if verb and chirp:
            findings.append(
                "<b>verbatimize is a privacy regression.</b> It consumes Chirp-3's "
                "de-identified transcript and re-derives the words from audio, restoring "
                f"identifiers Chirp had removed: the leak rate goes from "
                f"{chirp['leak_rate'] * 100:.0f}% to {verb['leak_rate'] * 100:.0f}%."
            )

    findings.append(
        "<b>Every number here is computed only over the span each human transcript "
        "covers.</b> The transcripts stop early on a third of the cohort, most of those "
        "at a hard 32-minute cutoff. Scoring whole sessions against them charges each "
        "system for minutes the reference never described and puts WER in the forties "
        "rather than the low teens."
    )

    items = "".join(f"<li>{f}</li>" for f in findings)
    return f"""
<section class="overview">
  <h2>Overview</h2>
  <div class="stats">{"".join(cards)}</div>
  <ul class="findings">{items}</ul>
</section>"""


def swatch_legend(items, width: int, y0: int, columns: int = 2) -> tuple[str, int]:
    """A colour key drawn inside the SVG rather than beside it in HTML.

    The stacked charts carry their legend in page markup, which is fine on the
    report but leaves an exported figure with unlabelled colours. Anything
    exported standalone needs its key in the same file.
    """
    if not items:
        return "", 0
    rows = -(-len(items) // columns)
    column_width = width / columns
    out = []
    for index, (label, colour, note) in enumerate(items):
        column, row = divmod(index, rows)
        x = column * column_width
        y = y0 + row * 17
        out.append(
            f'<rect x="{x:.1f}" y="{y - 9}" width="10" height="10" rx="2" fill="{colour}"/>'
        )
        # The dash is an entity, not a literal: these files are loaded from
        # disk by a headless browser, which without a charset declaration reads
        # UTF-8 as windows-1252 and renders every em dash as "a EUR quote".
        label_text = escape(label)
        note_text = f" &#8212; {escape(note)}" if note else ""
        out.append(
            f'<text x="{x + 16:.1f}" y="{y}" class="leg">{label_text}{note_text}</text>'
        )
    return "".join(out), rows * 17


def taxonomy_svg(data: dict | None, legend: bool = False) -> tuple[str, int, int]:
    """Stacked decomposition of WER into edit categories.

    The one chart on this page that answers "why is WER so high" rather than
    "how high is it". The segments sum to the reported WER exactly, which is
    the point: a decomposition that does not reconstruct the headline number is
    describing a different alignment.
    """
    if not data:
        return "", 0, 0
    rows = [
        (n, parts, registry.entry_of(data, n)) for n, parts, _, _ in SYSTEMS
        if registry.entry_of(data, n)
    ]
    if not rows:
        return "", 0, 0

    line_h = 14
    max_lines = max(len(p) for _, p, _ in rows)
    row_h = max(28, max_lines * line_h + 4)
    gap, pad_l, pad_r, pad_t, pad_b = 12, 250, 66, 16, 32
    plot_w = 330
    height = pad_t + len(rows) * (row_h + gap) + pad_b
    width = pad_l + plot_w + pad_r

    top = max(sum(r[2]["rates"].values()) for r in rows) * 1.08 or 1.0

    # The key is laid out first so its height is known before the header is
    # written; patching a viewBox after the fact is how charts end up cropped.
    key, key_height = ("", 0)
    if legend:
        # One column: these notes are long enough that two columns collide and
        # the right-hand one runs off the canvas.
        key, key_height = swatch_legend(
            list(TAXONOMY_PARTS), width, height + 8, columns=1,
        )
        key_height += 14
    total_height = height + key_height

    out = [
        f'<svg viewBox="0 0 {width} {total_height}" width="{width}" '
        f'height="{total_height}" role="img" '
        f'aria-label="What the word error rate is made of">'
    ]
    for i in range(6):
        value = top * i / 5
        gx = round(pad_l + (value / top) * plot_w, 1)
        out.append(
            f'<line x1="{gx}" y1="{pad_t}" x2="{gx}" y2="{height - pad_b}" '
            f'stroke="{GRIDLINE}" stroke-width="1"/>'
            f'<text x="{gx}" y="{height - pad_b + 15}" text-anchor="middle" '
            f'class="tick">{value * 100:.0f}%</text>'
        )

    for index, (_, label_parts, stats) in enumerate(rows):
        y = pad_t + index * (row_h + gap)
        block = len(label_parts) * line_h
        first = y + (row_h - block) / 2 + line_h - 3
        for line_index, component in enumerate(label_parts):
            suffix = " +" if line_index < len(label_parts) - 1 else ""
            out.append(
                f'<text x="{pad_l - 12}" y="{first + line_index * line_h:.1f}" '
                f'text-anchor="end" class="cat">{escape(component + suffix)}</text>'
            )
        cursor = pad_l
        total = 0.0
        for name, colour, _ in TAXONOMY_PARTS:
            value = stats["rates"].get(name) or 0.0
            total += value
            w = (value / top) * plot_w
            out.append(
                f'<rect x="{cursor:.1f}" y="{y}" width="{max(w, 0.4):.1f}" '
                f'height="{row_h}" fill="{colour}"/>'
            )
            if w > 24:
                out.append(
                    f'<text x="{cursor + w / 2:.1f}" y="{y + row_h / 2 + 4}" '
                    f'text-anchor="middle" class="inbar">{value * 100:.0f}</text>'
                )
            cursor += w
        out.append(
            f'<text x="{width - 8}" y="{y + row_h / 2 + 4}" text-anchor="end" '
            f'class="val">{total * 100:.1f}%</text>'
        )

    out.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    out.append(key)
    out.append("</svg>")
    return "".join(out), width, total_height


def taxonomy_panel(data: dict | None) -> str:
    """The decomposition with the reading that matters, for the report page."""
    svg, _, _ = taxonomy_svg(data)
    if not svg:
        return ""
    swatches = "".join(
        f'<span><span class="sw" style="background:{colour}"></span>'
        f'<b>{escape(name)}</b> &mdash; {escape(note)}</span>'
        for name, colour, note in TAXONOMY_PARTS
    )
    ours = registry.entry_of(data, "ours") or {}
    rates = ours.get("rates", {})
    heard_wrong = (rates.get("different word", 0) + rates.get("near-miss", 0)) * 100
    missed = rates.get("deletion", 0) * 100
    unrecorded = (
        rates.get("other insertion", 0) + rates.get("repetition", 0)
        + rates.get("backchannel", 0) + rates.get("discourse marker", 0)
    ) * 100
    return f"""
<section>
  <h2>What the word error rate is made of</h2>
  <p class="prose">Filled pauses are removed from <em>both</em> sides before scoring,
  so um and uh contribute nothing to any of these bars, and scoring is restricted to
  the span each human transcript covers. Splitting the remaining edits by type
  separates three things WER adds together: words the machine heard wrong, words it
  missed, and words it produced that the transcript does not record. The segments sum
  to the reported WER exactly. <b>Deletions now dominate</b> &mdash; the machine
  missing reference words, not inventing them.</p>
  {svg}
  <div class="legend parts">{swatches}</div>
  <p class="prose" style="margin-top:22px">For our pipeline that is
  <b>{heard_wrong:.1f}%</b> of reference words misheard, <b>{missed:.1f}%</b> missed,
  and <b>{unrecorded:.1f}%</b> spoken but unrecorded by the transcriber. An earlier
  version of this page put the last figure at 31% and the total in the forties: that
  was an artifact of scoring whole sessions against transcripts that stop early, most
  of them at a hard 32-minute cutoff. Restricted to the covered span, insertions
  collapse to about a point and <b>deletions are the largest term</b>. The number to
  quote for transcription accuracy is the misheard figure.</p>
</section>"""


# Identifier types with enough gold spans to carry a rate. Dates (4 spans) and
# ages (2) are reported in the footnote instead: a percentage over four spans
# moves 25 points per span and would read as a finding.
LEAK_TYPES = [("name", "#c0392b"), ("location", "#2a78d6")]
MIN_SPANS_FOR_RATE = 20


def leak_type_svg(data: dict | None, legend: bool = False) -> tuple[str, int, int]:
    """Leak rate per system, split by the kind of identifier that leaked.

    Grouped rather than stacked: these are independent rates over different
    denominators, not parts of a whole, so stacking them would invent a total
    that means nothing.
    """
    if not data:
        return "", 0, 0
    aggregate = data.get("aggregate", {})
    rows = []
    for name, parts, colour, _ in SYSTEMS:
        entry = registry.entry_of(aggregate, name)
        if not entry:
            continue
        # A system with no redaction step leaks every span by construction;
        # including it turns the chart into a comparison against nothing.
        if not entry.get("redactions"):
            continue
        by_type = entry.get("by_type", {})
        values = [
            (kind, by_type[kind]["leak_rate"], by_type[kind]["spans"], by_type[kind]["leaked"])
            for kind, _ in LEAK_TYPES
            if by_type.get(kind) and by_type[kind]["spans"] >= MIN_SPANS_FOR_RATE
        ]
        if values:
            rows.append((parts, values))
    if not rows:
        return "", 0, 0

    line_h = 14
    bar_h = 15
    max_lines = max(len(p) for p, _ in rows)
    row_h = max(len(LEAK_TYPES) * (bar_h + 4), max_lines * line_h + 6)
    gap, pad_l, pad_r, pad_t, pad_b = 16, 250, 66, 18, 32
    plot_w = 330
    height = pad_t + len(rows) * (row_h + gap) + pad_b

    key, key_height = ("", 0)
    if legend:
        items = []
        for kind, colour in LEAK_TYPES:
            spans = next(
                (v[2] for _, vals in rows for v in vals if v[0] == kind), 0,
            )
            items.append((kind, colour, f"{spans} gold spans with a surface form"))
        key, key_height = swatch_legend(items, plot_w + pad_l, height + 8, columns=1)
        key_height += 14
    total_height = height + key_height
    width = pad_l + plot_w + pad_r

    out = [
        f'<svg viewBox="0 0 {width} {total_height}" width="{width}" '
        f'height="{total_height}" role="img" aria-label="Leak rate by identifier type">'
    ]
    for i in range(6):
        value = i / 5
        gx = round(pad_l + value * plot_w, 1)
        out.append(
            f'<line x1="{gx}" y1="{pad_t}" x2="{gx}" y2="{height - pad_b}" '
            f'stroke="{GRIDLINE}" stroke-width="1"/>'
            f'<text x="{gx}" y="{height - pad_b + 15}" text-anchor="middle" '
            f'class="tick">{value * 100:.0f}%</text>'
        )

    for index, (label_parts, values) in enumerate(rows):
        y = pad_t + index * (row_h + gap)
        block = len(label_parts) * line_h
        first = y + (row_h - block) / 2 + line_h - 3
        for line_index, component in enumerate(label_parts):
            suffix = " +" if line_index < len(label_parts) - 1 else ""
            out.append(
                f'<text x="{pad_l - 12}" y="{first + line_index * line_h:.1f}" '
                f'text-anchor="end" class="cat">{escape(component + suffix)}</text>'
            )
        colours = dict(LEAK_TYPES)
        stack_top = y + (row_h - len(values) * (bar_h + 4)) / 2
        for slot, (kind, rate, spans, leaked) in enumerate(values):
            by = stack_top + slot * (bar_h + 4)
            w = max(rate * plot_w, 1.0)
            out.append(
                f'<rect x="{pad_l}" y="{by:.1f}" width="{w:.1f}" height="{bar_h}" '
                f'fill="{colours[kind]}" opacity="0.9"/>'
                f'<text x="{pad_l + w + 7:.1f}" y="{by + bar_h - 3:.1f}" class="val">'
                f'{rate * 100:.0f}% &#183; {leaked} of {spans}</text>'
            )

    out.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    out.append(key)
    out.append("</svg>")
    return "".join(out), width, total_height


def exposure_svg(data: dict | None, legend: bool = False) -> tuple[str, int, int]:
    """Per-transcript exposure: how many transcripts are clean, and how badly
    the rest are not.

    A distribution rather than a mean, because the question is operational --
    "can I release this transcript" is asked one transcript at a time, and an
    average of 27% exposed tells you nothing about how many files are safe.
    """
    if not data:
        return "", 0, 0
    aggregate = data.get("aggregate", {})
    per_visit = data.get("per_visit", {})
    rows = []
    for name, parts, colour, _ in SYSTEMS:
        entry = registry.entry_of(aggregate, name)
        if not entry:
            continue
        values = sorted(
            registry.entry_of(v, name)["exposed"]
            for v in per_visit.values() if registry.entry_of(v, name)
        )
        if not values:
            continue
        rows.append((parts, colour, entry, values))
    if not rows:
        return "", 0, 0

    line_h = 14
    band_h = 26
    max_lines = max(len(p) for p, _, _, _ in rows)
    row_h = max(band_h + 12, max_lines * line_h + 8)
    gap, pad_l, pad_r, pad_t, pad_b = 14, 250, 92, 20, 40
    plot_w = 320
    height = pad_t + len(rows) * (row_h + gap) + pad_b
    width = pad_l + plot_w + pad_r

    ceiling = max(v for _, _, _, values in rows for v in values) or 1
    # Buckets: clean, a handful, many. The first is the number people act on.
    buckets = [
        (0, 0, "no identifiers left open"),
        (1, 4, "1-4 left open"),
        (5, 19, "5-19 left open"),
        (20, ceiling, "20+ left open"),
    ]
    shades = ["#1baf7a", "#eda100", "#e8834a", "#c0392b"]

    key, key_height = ("", 0)
    if legend:
        key, key_height = swatch_legend(
            [
                (label, shades[i], "")
                for i, (_, _, label) in enumerate(buckets)
            ],
            plot_w + pad_l, height + 8, columns=2,
        )
        key_height += 14
    total_height = height + key_height

    out = [
        f'<svg viewBox="0 0 {width} {total_height}" width="{width}" '
        f'height="{total_height}" role="img" '
        f'aria-label="Per-transcript PII exposure by system">'
    ]
    for index, (label_parts, _, entry, values) in enumerate(rows):
        y = pad_t + index * (row_h + gap)
        block = len(label_parts) * line_h
        first = y + (row_h - block) / 2 + line_h - 3
        for line_index, component in enumerate(label_parts):
            suffix = " +" if line_index < len(label_parts) - 1 else ""
            out.append(
                f'<text x="{pad_l - 12}" y="{first + line_index * line_h:.1f}" '
                f'text-anchor="end" class="cat">{escape(component + suffix)}</text>'
            )
        counts = [
            sum(1 for v in values if low <= v <= high) for low, high, _ in buckets
        ]
        total = sum(counts) or 1
        cursor = pad_l
        by = y + (row_h - band_h) / 2
        for slot, count in enumerate(counts):
            w = count / total * plot_w
            if w <= 0:
                continue
            out.append(
                f'<rect x="{cursor:.1f}" y="{by:.1f}" width="{w:.1f}" '
                f'height="{band_h}" fill="{shades[slot]}"/>'
            )
            if w > 26:
                out.append(
                    f'<text x="{cursor + w / 2:.1f}" y="{by + band_h / 2 + 4:.1f}" '
                    f'text-anchor="middle" class="inbar">{count}</text>'
                )
            cursor += w
        out.append(
            f'<text x="{width - 8}" y="{by + band_h / 2 + 4:.1f}" text-anchor="end" '
            f'class="val">{counts[0]} of {total} clean</text>'
        )

    out.append(
        f'<text x="{pad_l}" y="{height - pad_b + 24}" class="tick">'
        f'each band is the share of transcripts in that exposure bucket'
        f'</text>'
    )
    out.append(key)
    out.append("</svg>")
    return "".join(out), width, total_height


def exposure_panel(data: dict | None) -> str:
    """Per-transcript exposure with the bounds the metric actually supports."""
    svg, _, _ = exposure_svg(data)
    if not svg:
        return ""
    aggregate = data.get("aggregate", {})
    rows = []
    for name, parts, colour, _ in SYSTEMS:
        entry = registry.entry_of(aggregate, name)
        if not entry:
            continue
        gold = (
            f'{entry["gold_exposed_rate"] * 100:.0f}%'
            if entry.get("gold_exposed_rate") is not None else "-"
        )
        rows.append(
            f'<tr><td class="sys"><span class="swatch" style="background:{colour}"></span>'
            f'{escape(" + ".join(parts))}</td>'
            f'<td class="num">{entry["transcripts"]}</td>'
            f'<td class="num">{entry["gold_exposed"]} / {entry["gold_locations"]} ({gold})</td>'
            f'<td class="num">{entry["exposed"]} / {entry["pii_locations"]} '
            f'({entry["exposed_rate"] * 100:.0f}%)</td>'
            f'<td class="num">{entry["clean_transcripts"]} '
            f'({entry["clean_transcript_rate"] * 100:.0f}%)</td></tr>'
        )
    return f"""
  <h3 class="condhead">Per-transcript exposure
    <span class="dir">how many identifier locations each transcript still leaves open</span></h3>
  <p class="prose">The leak rate above can only be measured where the transcriber left
  an identifier's surface form intact: 310 spans in 35 files. This asks the operational
  question over the whole cohort instead. A <em>PII location</em> is any position where
  the human marked a span &mdash; including the 577 already scrubbed to
  <code>{{redacted}}</code>, which no verbatim search can test &mdash; or where another
  system emitted a placeholder. A system is exposed at a location when it redacted
  nothing there.</p>
  {svg}
  <div class="tablewrap"><table>
    <thead><tr><th>System</th><th>Transcripts</th>
      <th>Exposed, human-marked spans</th><th>Exposed, all known locations</th>
      <th>Fully clean</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
  <p class="figcap">Two bounds, and the truth sits between them. The human-marked column
  counts only verified PII, so it is the floor. The all-locations column includes spans
  proposed by other systems, some of which are false positives that a system was right
  to skip, so it is the ceiling. Both are computed leave-one-out: a system is never
  judged against its own detections, because scoring a system on locations it proposed
  hands the win to whichever redactor is most aggressive &mdash; done that way, Chirp-3
  scored 34% against Gemma's 52%, reversing every other measure. Exposure is a lower
  bound on risk in one more sense: an identifier that no transcriber marked and no
  system caught is invisible here.</p>"""


def leak_type_panel(data: dict | None) -> str:
    """The by-type leak chart with the caveats its denominators require."""
    svg, _, _ = leak_type_svg(data)
    if not svg:
        return ""
    counts = data.get("spans_by_type", {})
    small = ", ".join(
        f"{kind} ({counts[kind]})" for kind in ("date", "age")
        if counts.get(kind)
    )
    swatches = "".join(
        f'<span><span class="sw" style="background:{colour}"></span>'
        f'<b>{escape(kind)}</b></span>'
        for kind, colour in LEAK_TYPES
    )
    return f"""
  <h3 class="condhead">Leak rate by identifier type
    <span class="dir">lower is better</span></h3>
  <p class="prose">The brace convention carries no label, so each gold span is typed
  by what a system called it when some system caught it &mdash; pooled across every
  system and visit, which types 91 of 102 distinct surface forms. The rest fall back
  to a lexical rule for dates and are otherwise left unclassified rather than guessed
  into a bucket.</p>
  {svg}
  <div class="legend parts">{swatches}</div>
  <p class="figcap">Only types with at least {MIN_SPANS_FOR_RATE} gold spans get a
  rate. {"Too few to report: " + small + "." if small else ""} 16 spans stayed
  unclassified because no system ever caught them; they leak at 62% under every
  system, which is what "nothing detects these" looks like. Denominators count only
  spans whose surface form the transcriber left intact &mdash; the 577 spans already
  scrubbed to {{redacted}} cannot leak and are excluded.</p>"""


def redaction_svg(data: dict | None) -> tuple[str, int, int]:
    """Over- and under-redaction as a diverging chart around the human gold.

    Centre is the human transcripts' own annotation. Everything left of it is
    identifying material the system left in the clear; everything right of it is
    material the system redacted that the transcribers did not mark. The two are
    drawn as one span per system rather than as a single net figure because they
    cancel: a redactor that misses ten names and invents ten places nets zero and
    is wrong twice over. The dot is that net, and its distance from the centre of
    its own bar is the tell.
    """
    if not data:
        return "", 0, 0
    aggregate = data.get("aggregate", {})
    rows = [
        (n, parts, colour, aggregate[n]) for n, parts, colour, _ in SYSTEMS
        if n in aggregate
    ]
    if not rows:
        return "", 0, 0

    line_h = 14
    max_lines = max(len(_label_lines(p)) for _, p, _, _ in rows)
    row_h = max(34, max_lines * line_h + 8)
    gap, pad_l, pad_r, pad_t, pad_b = 14, 250, 62, 20, 34
    plot_w = 340
    height = pad_t + len(rows) * (row_h + gap) + pad_b
    width = pad_l + plot_w + pad_r

    left = max((r[3]["under_rate"] for r in rows), default=0.5) or 0.5
    right = max((r[3]["over_rate"] for r in rows), default=0.5) or 0.5
    # A shared scale either side of zero would squash the under-redaction arm to
    # invisibility (over-redaction runs several times larger), so each side gets
    # its own scale and the axis is labelled to say so.
    left = max(left * 1.15, 0.05)
    right = max(right * 1.15, 0.05)
    zero = pad_l + plot_w * (left / (left + right))

    def x(value: float) -> float:
        if value < 0:
            return zero - (abs(value) / left) * (zero - pad_l)
        return zero + (value / right) * (pad_l + plot_w - zero)

    out = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Over and under redaction by system">'
    ]
    for fraction in (0.5, 1.0):
        for value in (-left * fraction, right * fraction):
            gx = round(x(value), 1)
            out.append(
                f'<line x1="{gx}" y1="{pad_t}" x2="{gx}" y2="{height - pad_b}" '
                f'stroke="{GRIDLINE}" stroke-width="1"/>'
                f'<text x="{gx}" y="{height - pad_b + 15}" text-anchor="middle" '
                f'class="tick">{abs(value) * 100:.0f}%</text>'
            )

    for index, (name, label_parts, colour, stats) in enumerate(rows):
        y = pad_t + index * (row_h + gap)
        block = len(label_parts) * line_h
        first = y + (row_h - block) / 2 + line_h - 3
        for line_index, component in enumerate(label_parts):
            suffix = " +" if line_index < len(label_parts) - 1 else ""
            out.append(
                f'<text x="{pad_l - 14}" y="{first + line_index * line_h:.1f}" '
                f'text-anchor="end" class="cat">{escape(component + suffix)}</text>'
            )
        cy = y + row_h / 2
        lo, hi = x(-stats["under_rate"]), x(stats["over_rate"])
        out.append(
            f'<line x1="{lo:.1f}" y1="{cy}" x2="{hi:.1f}" y2="{cy}" stroke="{colour}" '
            f'stroke-width="9" opacity="0.32" stroke-linecap="butt"/>'
        )
        out.append(
            f'<circle cx="{x(stats["net_rate"]):.1f}" cy="{cy}" r="5.5" fill="{colour}"/>'
        )
        out.append(
            f'<text x="{lo - 6:.1f}" y="{cy + 4}" text-anchor="end" class="val">'
            f'{stats["under_rate"] * 100:.0f}%</text>'
            f'<text x="{hi + 6:.1f}" y="{cy + 4}" class="val">'
            f'{stats["over_rate"] * 100:.0f}%</text>'
        )

    out.append(
        f'<line x1="{zero:.1f}" y1="{pad_t - 6}" x2="{zero:.1f}" y2="{height - pad_b + 4}" '
        f'stroke="{TEXT}" stroke-width="1.5"/>'
        f'<text x="{zero:.1f}" y="{pad_t - 10}" text-anchor="middle" class="tick">'
        f'matches the human annotation</text>'
        f'<text x="{pad_l}" y="{height - pad_b + 29}" class="tick">'
        f'&#8592; left in the clear</text>'
        f'<text x="{pad_l + plot_w}" y="{height - pad_b + 29}" text-anchor="end" '
        f'class="tick">redacted beyond the annotation &#8594;</text></svg>'
    )
    return "".join(out), width, height


def redaction_panel(data: dict | None, leaks: dict | None = None,
                    exposure: dict | None = None) -> str:
    """The diverging chart with its table and caveats, for the report page."""
    svg, _, _ = redaction_svg(data)
    if not svg:
        return ""
    aggregate = data.get("aggregate", {})
    rows = [
        (n, parts, colour, aggregate[n]) for n, parts, colour, _ in SYSTEMS
        if n in aggregate
    ]

    table_rows = []
    for name, label_parts, colour, stats in rows:
        leak = (
            f'{stats["leak_rate"] * 100:.1f}%' if stats.get("leak_rate") is not None else "-"
        )
        table_rows.append(
            f'<tr><td class="sys"><span class="swatch" style="background:{colour}"></span>'
            f'{escape(" + ".join(label_parts))}</td>'
            f'<td class="num">{stats["gold_spans"]}</td>'
            f'<td class="num">{stats["predicted_spans"]}</td>'
            f'<td class="num">{(stats["recall"] or 0) * 100:.1f}%</td>'
            f'<td class="num">{(stats["precision"] or 0) * 100:.1f}%</td>'
            f'<td class="num">{stats["f1"] * 100:.1f}%</td>'
            f'<td class="num">{leak}</td></tr>'
        )

    return f"""
<section>
  <h2>PII redaction</h2>
  <p class="prose">The human transcripts are the gold standard here: transcribers
  wrap identifying material in curly braces, and 887 such spans exist across 142 of
  the 269 sessions. Chirp-3 redacts natively into labelled placeholders; our
  pipelines transcribe verbatim and redact nothing, so Gemma 4 was run over their
  output in overlapping 400-word windows to give them a comparable capability. A
  system's redaction is matched to a gold span by aligning the two token sequences
  &mdash; "isaiah" against "[PERSON_NAME]" share no text, so they fall in the same
  difflib replace block, which is exactly the correspondence needed.</p>
  {svg}
  <div class="tablewrap"><table>
    <thead><tr><th>System</th><th>Gold spans</th><th>Redacted</th><th>Recall</th>
      <th>Precision</th><th>F1</th><th>Leak rate</th></tr></thead>
    <tbody>{"".join(table_rows)}</tbody>
  </table></div>
  {exposure_panel(exposure)}
  {leak_type_panel(leaks)}
  <div class="caveats" style="margin-top:26px">
    <div class="caveat">
      <h3>Recall is trustworthy; precision is a lower bound</h3>
      <p>A gold span is real PII, so failing to redact one is a real miss and recall
      means what it says. Precision does not: only 142 of 269 transcripts carry any
      annotation at all, and every genuine identifier a transcriber did not mark
      counts against a system that caught it. Read the right-hand arm as "redacted
      beyond what the humans marked", not as "wrong".</p>
    </div>
    <div class="caveat">
      <h3>Over-redaction is not free</h3>
      <p>Some of that right-hand arm is genuinely spurious: Chirp-3 turns "just some
      rooms" into "[LOCATION], just some rooms" on one session checked by hand. Each
      such substitution removes a real word from the transcript, which is why the
      redacted systems also carry a small WER penalty against a reference that keeps
      the original word.</p>
    </div>
    <div class="caveat">
      <h3>Leak rate is the number without an alignment step</h3>
      <p>Of the gold spans whose surface form the transcriber left intact, the leak
      rate is how many appear verbatim in the system's output. It needs no matching
      and no gold completeness assumption &mdash; the name is either still there or
      it is not &mdash; so it is the figure to quote when the question is privacy
      rather than detector quality.</p>
    </div>
  </div>
</section>"""


def comparison_chart(mono: dict, stereo: dict, title: str, key: str,
                     direction: str = "lower is better") -> str:
    """The dumbbell plus its heading, for the report page."""
    svg, _, _ = comparison_svg(mono, stereo, title, key)
    if not svg:
        return ""
    return (
        f'<h3 class="condhead">{escape(title)}'
        f'<span class="dir">{escape(direction)}; right-hand number is the shift '
        f'in points</span></h3>' + svg
    )


def comparison_svg(mono: dict, stereo: dict, title: str, key: str) -> tuple[str, int, int]:
    """One row per system, a line joining its mono and stereo-container value.

    A dumbbell rather than two separate charts: the quantity of interest is the
    shift between conditions, and putting the two values on one row makes the
    shift the thing you read, with its direction visible at a glance.
    """
    rows = []
    for name, parts, colour, _ in SYSTEMS:
        left = (mono.get("aggregate", {}).get(name) or {}).get(key)
        right = (stereo.get("aggregate", {}).get(name) or {}).get(key)
        if left is None or right is None:
            continue
        rows.append((name, parts, colour, left, right))
    if not rows:
        return "", 0, 0

    line_h = 14
    max_lines = max(len(p) for _, p, _, _, _ in rows)
    row_h = max(32, max_lines * line_h + 6)
    gap, pad_l, pad_r, pad_t, pad_b = 12, 250, 74, 22, 46
    plot_w = 320
    height = pad_t + len(rows) * (row_h + gap) + pad_b
    width = pad_l + plot_w + pad_r

    values = [v for _, _, _, a, b in rows for v in (a, b)]
    low, high = min(values), max(values)
    span = (high - low) or max(high, 0.01)
    low, high = max(low - span * 0.15, 0.0), high + span * 0.15

    def x(value: float) -> float:
        return pad_l + (value - low) / (high - low) * plot_w

    out = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{escape(title)}, mono against stereo container">'
    ]
    for i in range(5):
        value = low + (high - low) * i / 4
        gx = round(x(value), 1)
        out.append(
            f'<line x1="{gx}" y1="{pad_t}" x2="{gx}" y2="{height - pad_b}" '
            f'stroke="{GRIDLINE}" stroke-width="1"/>'
            f'<text x="{gx}" y="{height - pad_b + 15}" text-anchor="middle" '
            f'class="tick">{value * 100:.0f}%</text>'
        )

    for index, (name, parts, colour, left, right) in enumerate(rows):
        y = pad_t + index * (row_h + gap)
        block = len(parts) * line_h
        first = y + (row_h - block) / 2 + line_h - 3
        for line_index, component in enumerate(parts):
            suffix = " +" if line_index < len(parts) - 1 else ""
            out.append(
                f'<text x="{pad_l - 14}" y="{first + line_index * line_h:.1f}" '
                f'text-anchor="end" class="cat">{escape(component + suffix)}</text>'
            )
        cy = y + row_h / 2
        out.append(
            f'<line x1="{x(left):.1f}" y1="{cy}" x2="{x(right):.1f}" y2="{cy}" '
            f'stroke="{colour}" stroke-width="3" opacity="0.45"/>'
            f'<circle cx="{x(left):.1f}" cy="{cy}" r="5" fill="#ffffff" '
            f'stroke="{colour}" stroke-width="2"/>'
            f'<circle cx="{x(right):.1f}" cy="{cy}" r="5" fill="{colour}"/>'
            f'<text x="{width - 8}" y="{cy + 4}" text-anchor="end" class="val">'
            f'{(right - left) * 100:+.1f}</text>'
        )

    ly = height - 12
    out.append(
        f'<circle cx="{pad_l}" cy="{ly - 4}" r="5" fill="#ffffff" stroke="{MUTED}" '
        f'stroke-width="2"/><text x="{pad_l + 11}" y="{ly}" class="leg">'
        f'mono files (n={len(mono.get("per_visit", {}))})</text>'
        f'<circle cx="{pad_l + 150}" cy="{ly - 4}" r="5" fill="{MUTED}"/>'
        f'<text x="{pad_l + 161}" y="{ly}" class="leg">stereo-container files '
        f'(n={len(stereo.get("per_visit", {}))})</text></svg>'
    )
    return "".join(out), width, height


def condition_table(data: dict, caption: str) -> str:
    """One condition's systems scored on the metrics the condition is about.

    Every system in the file gets a row, so the stereo table -- which scores the
    other team's pipeline twice, once with channels and once forced to mono --
    reads as a paired comparison on identical visits rather than a comparison
    across two different visit sets.
    """
    aggregate = data.get("aggregate", {})
    heads = "".join(f"<th>{escape(t)}</th>" for t, _, _ in CONDITION_METRICS)
    rows = []
    for name, label_parts, colour, note in SYSTEMS:
        if name not in aggregate:
            continue
        flag = (
            '<span class="note">labels from the frozen Chirp-3 output; not re-run</span>'
            if name in NOT_RERUN else f'<span class="note">{escape(note)}</span>'
        )
        cells = []
        for _, key, _ in CONDITION_METRICS:
            value = aggregate[name].get(key)
            cells.append(
                f'<td class="num">{value * 100:.1f}%</td>' if value is not None
                else '<td class="num">-</td>'
            )
        rows.append(
            f'<tr><td class="sys"><span class="swatch" style="background:{colour}"></span>'
            f'{escape(" + ".join(label_parts))}{flag}</td>' + "".join(cells) + "</tr>"
        )
    if not rows:
        return ""
    return (
        f'<h3 class="condhead">{escape(caption)}'
        f'<span class="dir">lower is better, n={len(data.get("per_visit", {}))}</span></h3>'
        f'<div class="tablewrap"><table><thead><tr><th>System</th>{heads}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def conditions_section(mono: dict | None, stereo: dict | None) -> str:
    """Is the diarization gap a model gap or a channel gap?

    The other team's pipeline reads speakers straight off the stereo channels
    when the channels are genuinely separated, and only falls back to pyannote
    otherwise. On this corpus that is 66 of 269 sessions, so comparing it with
    our pipeline as shipped measures a channel trick as well as a model.
    """
    if not (mono and stereo):
        return ""
    return f"""
<section>
  <h2>Stereo-container files against mono files</h2>
  <p class="prose">66 of the 269 sessions are stereo, and the obvious question was
  whether the other team's pipeline was winning on speaker attribution by reading
  speakers off the channels rather than by diarizing. It was not. Their own
  channel-separation test &mdash; run here over every file &mdash; measures the average
  loudness difference between left and right at real speech moments and requires 3.0 dB
  before it trusts the channels. Across all 66 stereo files the measured separation runs
  from <b>0.000 to 0.082 dB</b>. Every one of them is a stereo container holding
  duplicated mono, so the channel path never fired on this corpus and all 269 files were
  diarized by pyannote from a downmix.</p>
  <p class="prose">What remains is still worth reading, but it is a comparison of
  recording provenance rather than of channel access: sessions delivered as
  stereo containers come from different sites and setups than the mono ones, and they
  score differently on every system. The shift is far larger than any gap between
  systems, which is a reminder that audio conditions dominate this corpus.</p>
  {comparison_chart(mono, stereo, "Transcription accuracy", "WER_no_ins")}
  {comparison_chart(mono, stereo, "Speaker confusion", "DER_confusion")}
  {condition_table(mono, "Mono files")}
  {condition_table(stereo, "Stereo-container files")}
</section>"""


def jiwer_section(summary: dict) -> str:
    """The third WER implementation, and the filler asymmetry it carries."""
    if not summary or not summary["systems"]:
        return ""
    rows = []
    for name, label_parts, colour, _ in SYSTEMS:
        stats = summary["systems"].get(name)
        if not stats:
            continue
        rows.append(
            f'<tr><td class="sys"><span class="swatch" style="background:{colour}"></span>'
            f'{escape(" + ".join(label_parts))}</td>'
            f'<td class="num">{stats["pooled"] * 100:.1f}%</td>'
            f'<td class="num">{stats["mean"] * 100:.1f}%</td>'
            f'<td class="num">{stats["median"] * 100:.1f}%</td>'
            f'<td class="num sub">{stats["sub"] * 100:.1f}%</td>'
            f'<td class="num sub">{stats["delete"] * 100:.1f}%</td>'
            f'<td class="num sub">{stats["insert"] * 100:.1f}%</td>'
            f'<td class="num">{stats["no_fillers"] * 100:.1f}%</td></tr>'
        )
    return f"""
<section>
  <h2>Cross-check: the study team's jiwer WER</h2>
  <p class="prose">A third WER implementation, supplied by the study team and vendored
  unmodified: jiwer's edit distance after everything inside square brackets is deleted,
  all punctuation is stripped and both sides are lowercased. It ran over the same
  {summary["total"]} visits, the same system outputs and the same covered spans as
  every other number on this page. Being a true minimum edit distance, it sits below
  the partner team's difflib upper bound; it reproduces the same ordering as both other
  metrics, which is now three independent implementations agreeing.</p>
  <div class="tablewrap">
  <table>
    <thead>
      <tr><th></th><th colspan="3">WER</th><th colspan="3">Error composition</th>
          <th>Fillers dropped<br>both sides</th></tr>
      <tr><th>System</th><th>Pooled</th><th>Mean visit</th><th>Median visit</th>
          <th>Sub</th><th>Del</th><th>Ins</th><th>Pooled</th></tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  </div>
  <div class="caveats" style="margin-top:26px">
    <div class="caveat">
      <h3>Deleting brackets is not neutral across systems</h3>
      <p>The rule removes the human transcripts' [inaudible] markup, which is the intent,
      but it also removes every CrisperWhisper filled pause &mdash; those are written
      [UM] and [UH] &mdash; while Chirp-3 writes "um" as a plain word that survives. The
      CrisperWhisper arms are therefore charged deletions for disfluencies they did
      transcribe. The last column removes filled pauses from both sides for every system
      alike, and the CrisperWhisper lead widens rather than narrows, so the asymmetry was
      working against the systems that win.</p>
    </div>
    <div class="caveat">
      <h3>Pooled, not averaged</h3>
      <p>The headline column pools errors and reference words across the corpus rather
      than averaging per-visit rates, so a two-hour session counts for more than a
      five-minute one. Mean and median are shown beside it; they sit close together here
      because the coverage window already removed the truncated-reference tail that
      distorts the partner metric's mean.</p>
    </div>
  </div>
</section>
"""


def partner_section(summary: dict) -> str:
    """The cross-check: their numbers, and the two things they added."""
    if not summary or not summary["systems"]:
        return ""
    rows = []
    for name, label_parts, colour, _ in SYSTEMS:
        stats = summary["systems"].get(name)
        if not stats:
            continue
        clean_mean = f'{stats["clean_mean"]:.1f}%' if stats["clean_mean"] is not None else "-"
        clean_median = f'{stats["clean_median"]:.1f}%' if stats["clean_median"] is not None else "-"
        rows.append(
            f'<tr><td class="sys"><span class="swatch" style="background:{colour}"></span>'
            f'{escape(" + ".join(label_parts))}</td>'
            f'<td class="num">{stats["raw"]:.1f}%</td>'
            f'<td class="num">{stats["normalized"]:.1f}%</td>'
            f'<td class="num">{stats["filler"]:.1f}%</td>'
            f'<td class="num">{stats["median"]:.1f}%</td>'
            f'<td class="num sub">{clean_mean}</td>'
            f'<td class="num sub">{clean_median}</td></tr>'
        )
    kept = summary["total"] - summary["truncated"]
    return f"""
<section>
  <h2>Cross-check: the partner team's WER</h2>
  <p class="prose">The partner team's own <em>compareFiles.py</em> was vendored
  unmodified and run over the same 269 visits, the same system outputs and the same
  human references. It is a different algorithm &mdash; difflib alignment rather than
  minimum edit distance, and filled pauses collapsed to a single token on both sides
  rather than deleted &mdash; so it reads a few points higher than ours throughout.
  It reproduces our ordering exactly, from code that shares nothing with ours, and it
  agrees on the LLM review being never better.</p>
  <div class="tablewrap">
  <table>
    <thead>
      <tr><th></th><th colspan="4">All {summary["total"]} visits</th>
          <th colspan="2">Excluding truncated references (n={kept})</th></tr>
      <tr><th>System</th><th>Raw</th><th>Normalized</th><th>Filler-normalized</th>
          <th>Median visit</th><th>Mean</th><th>Median</th></tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  </div>
  <div class="caveats" style="margin-top:26px">
    <div class="caveat">
      <h3>The means are set by broken references</h3>
      <p>On {summary["truncated"]} visits <em>every</em> system emits more than twice the
      reference's word count. Systems that share no code do not fail identically; those
      are human transcripts covering only part of the session, and they average around
      170% WER. Excluded, the right-hand block is the honest figure and the median is
      the number to quote.</p>
    </div>
    <div class="caveat">
      <h3>Chirp-3 wins the easy visits and loses the hard ones</h3>
      <p>Chirp-3 is the best system at the 10th and 25th percentile (4.7% and 8.7%,
      against 9.4% and 11.7% for ours) and the worst at the 90th (115% against 103%).
      Our pipeline beats it on only 124 of 269 visits and its median visit is 1.7 points
      worse: the aggregate win comes entirely from the failure tail, not from being
      better on a typical session.</p>
    </div>
  </div>
</section>"""


def chart_svg(
    title: str,
    values: dict[str, float | None],
    spreads: dict[str, tuple[float, float, float] | None],
    present: list[str],
    direction: str,
    caption: str | None = None,
    standalone: bool = False,
    footer: str = "",
) -> tuple[str, int, int]:
    """One horizontal-bar chart as inline SVG, with its pixel size.

    Horizontal because six system names do not fit legibly under vertical
    bars. Gridlines sit on the value axis only and are drawn before the bars
    so the data reads on top of them. The size is returned so a caller
    exporting a standalone image can fit the canvas to the chart exactly.
    """
    rows = [(n, parts, colour) for n, parts, colour, _ in SYSTEMS if n in present]
    if not rows:
        return "", 0, 0

    # Labels are multi-line (one component per line), so the row has to be tall
    # enough for the longest label and the label gutter wide enough for the
    # longest component -- otherwise names silently overlap or clip.
    line_h = 14
    max_lines = max(len(parts) for _, parts, _ in rows)
    row_h = max(30, max_lines * line_h + 6)
    gap, pad_l, pad_r, pad_b = 12, 250, 52, 30
    plot_w = 300
    # standalone puts the title block and legend inside the SVG, so an exported
    # image is a single element of known size. Composing them as HTML around
    # the SVG instead means the exporter has to predict the page's laid-out
    # height, and a wrong prediction silently crops the axis labels away.
    def wrap(text: str, limit: int) -> list[str]:
        lines: list[str] = []
        line = ""
        for word in text.split():
            if len(line) + len(word) + 1 > limit:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            lines.append(line)
        return lines

    # The subtitle wraps too. Unwrapped it ran off the right edge of the image,
    # silently truncating the sentence that explains the metric.
    caption_head = wrap(caption, 82) if (standalone and caption) else []
    head_h = (40 + len(caption_head) * 15) if standalone else 0
    # An exported PNG travels without the page around it, so the caption is
    # wrapped into the image itself. Height is computed from the wrapped line
    # count rather than assumed.
    caption_lines = wrap(footer, 96) if (standalone and footer) else []
    foot_h = (30 + 8 + len(caption_lines) * 14) if standalone else 0
    pad_t = 14 + head_h
    height = pad_t + len(rows) * (row_h + gap) + pad_b + foot_h
    width = pad_l + plot_w + pad_r

    numeric = [v for v in values.values() if v is not None]
    # s is (p25, median, p75): scale to p75, not the median, or a long upper
    # whisker runs past the plot area and strikes through the value label.
    spread_max = [s[2] for s in spreads.values() if s is not None]
    top = max(numeric + spread_max) if (numeric or spread_max) else 1.0
    top = max(top * 1.1, 0.05)

    better_low = direction.startswith("lower")
    ranked = [(n, v) for n, v in values.items() if v is not None]
    best = None
    if ranked:
        best = (min if better_low else max)(ranked, key=lambda kv: kv[1])[0]

    def x(value: float) -> float:
        return pad_l + (value / top) * plot_w

    # Explicit width/height as well as viewBox: an SVG with width="100%" and no
    # height has no intrinsic size, and a standalone export renders it short --
    # the axis labels and everything below them silently vanish from the image.
    # The combined report scales it back down responsively via CSS.
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{escape(title)} by system">'
    ]

    plot_bottom = height - pad_b - foot_h

    if standalone:
        parts.append(
            f'<text x="0" y="17" class="ttl">{escape(title)}</text>'
            f'<text x="0" y="34" class="sub">{escape(direction.upper())}</text>'
        )
        for i, line in enumerate(caption_head):
            parts.append(f'<text x="0" y="{50 + i * 15}" class="cap">{escape(line)}</text>')

    # gridlines + value-axis ticks, behind the data
    # Percentages, not bare decimals: 0.205 was read as "less than 1%" once, and
    # every metric on this page is a rate, so the unit belongs on the axis.
    ticks = 5
    for i in range(ticks + 1):
        value = top * i / ticks
        gx = round(x(value), 1)
        parts.append(
            f'<line x1="{gx}" y1="{pad_t}" x2="{gx}" y2="{plot_bottom}" '
            f'stroke="{GRIDLINE}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx}" y="{plot_bottom + 15}" text-anchor="middle" '
            f'class="tick">{value * 100:.0f}%</text>'
        )

    for index, (name, label_parts, colour) in enumerate(rows):
        y = pad_t + index * (row_h + gap)
        value = values.get(name)

        # One component per line, "+" kept at the end of the preceding line so
        # the composition reads as a chain rather than a list.
        block_h = len(label_parts) * line_h
        first_y = y + (row_h - block_h) / 2 + line_h - 3
        for line_index, component in enumerate(label_parts):
            suffix = " +" if line_index < len(label_parts) - 1 else ""
            parts.append(
                f'<text x="{pad_l - 24}" y="{first_y + line_index * line_h:.1f}" '
                f'text-anchor="end" class="cat">{escape(component + suffix)}</text>'
            )
        if value is None:
            parts.append(
                f'<text x="{pad_l + 6}" y="{y + row_h / 2 + 4}" class="tick">not scored</text>'
            )
            continue

        # A dot-and-range, not a bar. These per-visit distributions are heavily
        # right-skewed -- a few catastrophic visits drag the mean past the p75
        # (when this was chosen, chirp3's mean DER was 0.691 against a p75 of
        # 0.337) -- so a bar growing from zero both
        # overstates the typical visit and collides with the range drawn across
        # it, which reads as a strike-through rather than a spread. The median
        # dot carries "typical", the range carries spread, and the hollow mean
        # marker makes the skew legible as the gap between the two.
        cy = y + row_h / 2
        spread = spreads.get(name)
        if spread:
            low, median, high = spread
            lo, hi = x(low), x(high)
            parts.append(
                f'<line x1="{lo:.1f}" y1="{cy}" x2="{hi:.1f}" y2="{cy}" '
                f'stroke="{colour}" stroke-width="4" opacity="0.4" stroke-linecap="round"/>'
            )
            for cap in (lo, hi):
                parts.append(
                    f'<line x1="{cap:.1f}" y1="{cy - 6}" x2="{cap:.1f}" y2="{cy + 6}" '
                    f'stroke="{colour}" stroke-width="1.6" opacity="0.8"/>'
                )
            parts.append(f'<circle cx="{x(median):.1f}" cy="{cy}" r="5.5" fill="{colour}"/>')

        parts.append(
            f'<circle cx="{x(value):.1f}" cy="{cy}" r="4.5" fill="#ffffff" '
            f'stroke="{MUTED_DARK}" stroke-width="1.6"/>'
        )

        # Value in a fixed right-hand column rather than at the bar's end: the
        # p25-p75 whisker often extends past the bar, and a label placed by bar
        # length ends up struck through by the whisker cap.
        parts.append(
            f'<text x="{width - 8}" y="{y + row_h / 2 + 4}" text-anchor="end" '
            f'class="val">{value * 100:.1f}%</text>'
        )
        if name == best:
            parts.append(
                f'<circle cx="{pad_l - 12}" cy="{y + row_h / 2}" r="4" fill="{WINNER}"/>'
            )

    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{plot_bottom}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )

    if standalone and caption_lines:
        base = plot_bottom + 44
        for i, line in enumerate(caption_lines):
            parts.append(
                f'<text x="0" y="{base + i * 14}" class="figcap">{escape(line)}</text>'
            )

    if standalone:
        ly = height - 10
        parts.append(
            f'<line x1="0" y1="{height - 26}" x2="{width}" y2="{height - 26}" '
            f'stroke="{GRIDLINE}" stroke-width="1"/>'
            f'<circle cx="5" cy="{ly - 4}" r="5" fill="{MUTED}"/>'
            f'<text x="15" y="{ly}" class="leg">median visit</text>'
            f'<circle cx="98" cy="{ly - 4}" r="4.5" fill="#ffffff" '
            f'stroke="{MUTED_DARK}" stroke-width="1.6"/>'
            f'<text x="108" y="{ly}" class="leg">mean (value shown)</text>'
            f'<line x1="228" y1="{ly - 4}" x2="252" y2="{ly - 4}" '
            f'stroke="{MUTED}" stroke-width="4" opacity="0.4" stroke-linecap="round"/>'
            f'<text x="258" y="{ly}" class="leg">middle 50% of visits</text>'
            f'<circle cx="382" cy="{ly - 4}" r="4" fill="{WINNER}"/>'
            f'<text x="392" y="{ly}" class="leg">best</text>'
        )

    parts.append("</svg>")
    return "".join(parts), width, height


def composition_svg(aggregate: dict, present: list[str],
                    legend: bool = False) -> tuple[str, int, int]:
    """Stacked WER composition: insertions vs substitutions vs deletions.

    WER is the one number on this page that is a sum of parts, so it is the one
    place a stacked bar is the right mark. It exists to answer the question the
    single WER figure cannot: how much of the error is the machine hearing
    fillers the human transcript leaves out, rather than mishearing words.
    """
    rows = [(n, parts, colour) for n, parts, colour, _ in SYSTEMS if n in present]
    rows = [r for r in rows if aggregate.get(r[0], {}).get("WER_ins") is not None]
    if not rows:
        return "", 0, 0

    line_h = 14
    max_lines = max(len(p) for _, p, _ in rows)
    row_h = max(26, max_lines * line_h + 4)
    gap, pad_l, pad_r, pad_t, pad_b = 12, 250, 62, 14, 30
    plot_w = 290
    height = pad_t + len(rows) * (row_h + gap) + pad_b
    width = pad_l + plot_w + pad_r

    top = max(
        sum(aggregate[n].get(k) or 0.0 for _, k, _, _ in WER_PARTS) for n, _, _ in rows
    ) * 1.08 or 1.0

    key, key_height = ("", 0)
    if legend:
        key, key_height = swatch_legend(
            [(label, colour, note) for label, _, colour, note in WER_PARTS],
            width, height + 8, columns=1,
        )
        key_height += 14
    total_height = height + key_height

    out = [
        f'<svg viewBox="0 0 {width} {total_height}" width="{width}" '
        f'height="{total_height}" role="img" aria-label="WER composition by system">'
    ]
    for i in range(6):
        value = top * i / 5
        gx = round(pad_l + (value / top) * plot_w, 1)
        out.append(
            f'<line x1="{gx}" y1="{pad_t}" x2="{gx}" y2="{height - pad_b}" '
            f'stroke="{GRIDLINE}" stroke-width="1"/>'
            f'<text x="{gx}" y="{height - pad_b + 15}" text-anchor="middle" '
            f'class="tick">{value * 100:.0f}%</text>'
        )

    for index, (name, label_parts, _) in enumerate(rows):
        y = pad_t + index * (row_h + gap)
        block = len(label_parts) * line_h
        first = y + (row_h - block) / 2 + line_h - 3
        for line_index, component in enumerate(label_parts):
            suffix = " +" if line_index < len(label_parts) - 1 else ""
            out.append(
                f'<text x="{pad_l - 12}" y="{first + line_index * line_h:.1f}" '
                f'text-anchor="end" class="cat">{escape(component + suffix)}</text>'
            )
        cursor = pad_l
        total = 0.0
        for _, key, colour, _ in WER_PARTS:
            value = aggregate[name].get(key) or 0.0
            total += value
            w = (value / top) * plot_w
            out.append(
                f'<rect x="{cursor:.1f}" y="{y}" width="{max(w, 0.5):.1f}" '
                f'height="{row_h}" fill="{colour}"/>'
            )
            if w > 26:
                out.append(
                    f'<text x="{cursor + w / 2:.1f}" y="{y + row_h / 2 + 4}" '
                    f'text-anchor="middle" class="inbar">{value * 100:.0f}</text>'
                )
            cursor += w
        out.append(
            f'<text x="{width - 8}" y="{y + row_h / 2 + 4}" text-anchor="end" '
            f'class="val">{total * 100:.1f}%</text>'
        )

    out.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    out.append(key)
    out.append("</svg>")
    return "".join(out), width, total_height


def composition_panel(aggregate: dict, present: list[str]) -> str:
    """The stacked composition with its legend, for the report page."""
    svg, _, _ = composition_svg(aggregate, present)
    if not svg:
        return ""
    swatches = "".join(
        f'<span><span class="sw" style="background:{colour}"></span>'
        f'<b>{escape(label)}</b> &mdash; {escape(note)}</span>'
        for label, _, colour, note in WER_PARTS
    )
    return (
        '<figure class="panel wide">'
        '<figcaption><h3>What makes up the word error rate</h3>'
        '<span class="dir">lower is better</span>'
        '<p>The three error types sum to WER. Insertions dominate, but not because '
        'of disfluencies -- see the error-type breakdown above for what they are.</p>'
        '</figcaption>'
        + svg
        + f'<div class="legend parts">{swatches}</div>'
        + f'<p class="figcap">{escape(CAPTIONS["composition"])}</p></figure>'
    )


def bar_panel(
    title: str,
    direction: str,
    caption: str,
    values: dict[str, float | None],
    spreads: dict[str, tuple[float, float, float] | None],
    present: list[str],
    full_caption: str = "",
) -> str:
    """chart_svg wrapped in its heading and figure caption, for the report."""
    svg, _, _ = chart_svg(title, values, spreads, present, direction)
    if not svg:
        return ""
    reading = reading_line(values, spreads, direction)
    tail = ""
    if full_caption:
        tail += f'<p class="figcap">{escape(full_caption)}</p>'
    if reading:
        tail += f'<p class="reading">{escape(reading)}</p>'
    return (
        '<figure class="panel">'
        f'<figcaption><h3>{escape(title)}</h3>'
        f'<span class="dir">{escape(direction)}</span>'
        f'<p>{escape(caption)}</p></figcaption>'
        + svg
        + tail
        + "</figure>"
    )


def build_page(
    data: dict,
    font_b64: str,
    partner_summary: dict | None = None,
    jiwer_summary: dict | None = None,
    mono: dict | None = None,
    stereo: dict | None = None,
    redaction_data: dict | None = None,
    taxonomy: dict | None = None,
    leaks: dict | None = None,
    exposure: dict | None = None,
) -> str:
    aggregate = data.get("aggregate", {})
    per_visit = data.get("per_visit", {})
    present = [n for n, *_ in SYSTEMS if n in aggregate]
    subset = common_subset(per_visit, present)
    metrics = metrics_for(aggregate)

    panels = []
    for title, key, agg_key, direction, caption in metrics:
        values = {n: aggregate.get(n, {}).get(agg_key) for n in present}
        raw = collect(per_visit, key)
        spreads = {n: quartiles(raw.get(n, [])) for n in present}
        panels.append(
            bar_panel(
                title, direction, caption, values, spreads, present,
                full_caption=CAPTIONS.get(agg_key, ""),
            )
        )

    # coverage + common-subset table
    rows = []
    for name, label_parts, colour, note in SYSTEMS:
        if name not in aggregate:
            continue
        stats = aggregate[name]
        label = " + ".join(label_parts)
        cells = [
            f'<td class="sys"><span class="swatch" style="background:{colour}"></span>'
            f'{escape(label)}<span class="note">{escape(note)}</span></td>',
            f'<td class="num">{stats.get("visits", 0)}</td>',
        ]
        for _, _, agg_key, _, _ in metrics:
            value = stats.get(agg_key)
            cells.append(f'<td class="num">{value * 100:.1f}%</td>' if value is not None else '<td class="num">-</td>')
        for _, key, _, _, _ in metrics:
            value = subset_mean(per_visit, subset, name, key)
            cells.append(f'<td class="num sub">{value * 100:.1f}%</td>' if value is not None else '<td class="num sub">-</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    metric_heads = "".join(f"<th>{escape(t)}</th>" for t, *_ in metrics)

    # The old word-span DER is shown next to the corrected one so the metric
    # change is visible in the report rather than only in the commit log.
    der_rows = []
    for name, label_parts, colour, _ in SYSTEMS:
        if name not in aggregate:
            continue
        stats = aggregate[name]
        old, new = stats.get("DER_word_level"), stats.get("DER")
        if old is None or new is None:
            continue
        confusion = stats.get("DER_confusion")
        der_rows.append(
            f'<tr><td class="sys"><span class="swatch" style="background:{colour}"></span>'
            f'{escape(" + ".join(label_parts))}</td>'
            f'<td class="num">{old * 100:.1f}%</td><td class="num">{new * 100:.1f}%</td>'
            f'<td class="num">{confusion * 100:.1f}%</td>'
            f'<td class="num">{stats.get("no_timestamps", 0)}</td></tr>'
        )

    return f"""<title>Transcription system evaluation</title>
<style>
@font-face {{
  font-family: 'DM Sans';
  src: url(data:font/ttf;base64,{font_b64}) format('truetype');
  font-weight: 100 1000;
  font-display: block;
}}
:root {{
  --ground: #ffffff;
  --text: {TEXT};
  --muted: {MUTED};
  --muted-dark: {MUTED_DARK};
  --grid: {GRIDLINE};
  --axis: {AXIS};
  --winner: {WINNER};
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--ground);
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
  font-weight: 500;
  -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 1080px; margin: 0 auto; padding: 56px 28px 96px; }}
header {{ display: flex; flex-direction: column; gap: 10px; margin-bottom: 8px; }}
.eyebrow {{
  font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted);
}}
h1 {{ font-size: 34px; font-weight: 600; margin: 0; letter-spacing: -0.02em; text-wrap: balance; }}
.standfirst {{ font-size: 16px; color: var(--muted-dark); max-width: 62ch; line-height: 1.55; margin: 0; }}
h2 {{
  font-size: 12px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 18px; padding-bottom: 8px; border-bottom: 1px solid var(--grid);
}}
section {{ margin-top: 52px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 34px 40px; }}
.panel {{ margin: 0; display: flex; flex-direction: column; gap: 6px; }}
.panel figcaption {{ display: flex; flex-direction: column; gap: 3px; }}
.panel h3 {{ font-size: 17px; font-weight: 600; margin: 0; }}
.dir {{ font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }}
.panel figcaption p {{ font-size: 13px; color: var(--muted-dark); margin: 2px 0 4px; }}
.figcap {{
  font-size: 12.5px; line-height: 1.6; color: var(--muted); margin: 10px 0 0;
  max-width: 68ch; padding-top: 9px; border-top: 1px solid var(--grid);
}}
.reading {{
  font-size: 12.5px; line-height: 1.55; color: var(--muted-dark);
  margin: 7px 0 0; max-width: 68ch; font-weight: 600;
}}
.panel svg {{ width: 100%; height: auto; max-width: 100%; }}
svg .cat {{ font-family: 'DM Sans', sans-serif; font-size: 12px; font-weight: 500; fill: var(--text); }}
svg .val {{ font-family: 'DM Sans', sans-serif; font-size: 11px; font-weight: 500; fill: var(--muted); }}
svg .tick {{ font-family: 'DM Sans', sans-serif; font-size: 10px; font-weight: 500; fill: var(--muted); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px 22px; font-size: 12.5px; color: var(--muted-dark); margin-top: 14px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
.dot {{ width: 9px; height: 9px; border-radius: 50%; background: var(--winner); }}
.meddot {{ width: 11px; height: 11px; border-radius: 50%; background: var(--muted); }}
.meandot {{
  width: 10px; height: 10px; border-radius: 50%; background: var(--ground);
  border: 1.6px solid var(--muted-dark);
}}
.whisker {{ width: 24px; height: 4px; border-radius: 2px; background: var(--muted); opacity: 0.4; }}
.howto {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 20px 40px; }}
.howto h3 {{ font-size: 14px; font-weight: 600; margin: 0 0 5px; }}
.howto p {{ font-size: 13.5px; line-height: 1.6; color: var(--muted-dark); margin: 0; max-width: 58ch; }}
.howto b {{ font-weight: 600; color: var(--text); }}
.panel.wide {{ grid-column: 1 / -1; margin-top: 34px; }}
.legend.parts {{ flex-direction: column; gap: 6px; align-items: flex-start; }}
.legend.parts b {{ font-weight: 600; color: var(--text); }}
.sw {{ display: inline-block; width: 11px; height: 11px; border-radius: 2px; }}
svg .inbar {{
  font-family: 'DM Sans', sans-serif; font-size: 10.5px; font-weight: 600; fill: #ffffff;
}}
.condhead {{
  font-size: 15px; font-weight: 600; margin: 26px 0 8px;
  display: flex; align-items: baseline; gap: 10px;
}}
.overview {{ margin-top: 44px; }}
.stats {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 18px; margin-bottom: 28px;
}}
.stat {{
  display: flex; flex-direction: column; gap: 3px;
  padding: 16px 18px; background: #faf9f6; border: 1px solid var(--grid);
}}
.statlabel {{
  font-size: 10.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted);
}}
.statvalue {{
  font-size: 30px; font-weight: 600; letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums; line-height: 1.1;
}}
.statnote {{ font-size: 12px; color: var(--muted-dark); line-height: 1.4; }}
.statnote.dim {{ color: var(--muted); font-size: 11.5px; }}
.findings {{ margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 12px; }}
.findings li {{
  font-size: 14px; line-height: 1.6; color: var(--muted-dark); max-width: 76ch;
  padding-left: 15px; border-left: 2px solid var(--grid);
}}
.findings b {{ font-weight: 600; color: var(--text); }}
.tablewrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ text-align: right; padding: 9px 12px; border-bottom: 1px solid var(--grid); white-space: nowrap; }}
th {{
  font-size: 10.5px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); border-bottom: 1px solid var(--axis);
}}
thead tr:first-child th {{ border-bottom: none; padding-bottom: 2px; }}
th:first-child, td:first-child {{ text-align: left; }}
.num {{ font-variant-numeric: tabular-nums; }}
.sub {{ color: var(--muted-dark); background: #faf9f6; }}
.sys {{ font-weight: 500; }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 9px; vertical-align: baseline; }}
.note {{ display: block; font-size: 11.5px; color: var(--muted); margin-left: 19px; }}
.caveats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 22px 40px; }}
.caveat h3 {{ font-size: 14px; font-weight: 600; margin: 0 0 5px; }}
.caveat p {{ font-size: 13.5px; line-height: 1.6; color: var(--muted-dark); margin: 0; max-width: 60ch; }}
.prose {{ font-size: 14px; line-height: 1.65; color: var(--muted-dark); max-width: 68ch; margin: 0 0 20px; }}
.prose em {{ font-style: italic; color: var(--text); }}
footer {{ margin-top: 56px; padding-top: 16px; border-top: 1px solid var(--grid); font-size: 12px; color: var(--muted); }}
</style>

<div class="wrap">
<header>
  <span class="eyebrow">AMPSCZ / PSYCHS interviews &middot; DIALOG-DeID metrics</span>
  <h1>Six transcription systems, scored against human transcripts</h1>
  <p class="standfirst">Every system is scored on the same visits with the same
  reference: WER after filled-pause removal, speaker-attributed WER under
  Hungarian stream alignment, diarization error rate, and qualifier/temporal
  cue preservation.</p>
</header>

{summary_section(aggregate, redaction_data, taxonomy)}
<section>
  <h2>How to read these charts</h2>
  <div class="howto">
    <div>
      <h3>Every figure is a percentage</h3>
      <p>Every metric is a rate over reference words, or over speech time for the
      diarization figures. A speaker confusion of 15% means a seventh of speech time
      is on the wrong speaker &mdash; not a seventh of a percent.</p>
    </div>
    <div>
      <h3>Each row is a distribution, not one number</h3>
      <p>The <b>filled dot</b> is the median visit, the <b>hollow dot</b> is the
      mean (the figure printed at the right), and the <b>bar</b> spans the middle
      half of visits. A wide gap between the two dots means the mean is being
      carried by a few bad sessions rather than describing typical performance.</p>
    </div>
    <div>
      <h3>Compare systems, not absolute levels</h3>
      <p>Scoring covers only the span each human transcript reaches, because they
      stop early on a third of the cohort. Within that span the machine still keeps
      disfluencies and repetitions the transcriber deleted, so a point or so of every
      WER is convention rather than error. Use <b>WER excluding insertions</b>, or the
      error-type chart, for transcription quality.</p>
    </div>
    <div>
      <h3>Overlapping ranges are not a ranking</h3>
      <p>Where two systems' middle-50% bars overlap substantially, the difference
      between their averages is not established by these 269 visits.</p>
    </div>
  </div>
</section>

<section>
  <h2>Metrics by system</h2>
  <div class="grid">{"".join(panels)}</div>
  {composition_panel(aggregate, present)}
  <div class="legend">
    <span><span class="meddot"></span>median visit</span>
    <span><span class="meandot"></span>mean (the value shown)</span>
    <span><span class="whisker"></span>middle 50% of visits (p25&ndash;p75)</span>
    <span><span class="dot"></span>best on this metric</span>
  </div>
</section>

<section>
  <h2>All figures</h2>
  <div class="tablewrap">
  <table>
    <thead>
      <tr><th></th><th></th><th colspan="{len(metrics)}">All scored visits</th><th colspan="{len(metrics)}">Common subset (n={len(subset)})</th></tr>
      <tr><th>System</th><th>Visits</th>{metric_heads}{metric_heads}</tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  </div>
</section>

<section>
  <h2>Why DER was recomputed</h2>
  <p class="prose">The human transcripts carry turn <em>start</em> times only, so each
  turn's end is synthesized from the next turn's start. The reference therefore tiles
  the whole recording and declares no non-speech: measured across six sessions, false
  alarm was exactly 0.000 &mdash; nothing can be a false alarm when the reference calls
  every instant speech &mdash; while missed detection accounted for 0.41&ndash;0.63 of a
  0.54&ndash;0.76 DER and real speaker confusion for only 0.10&ndash;0.27. Scored that
  way, DER ranked systems by how much of the timeline their segments covered, penalising
  word-level output against systems that emit contiguous turns. The hypothesis is now
  built by the reference's own rule &mdash; same-speaker words grouped into turns, each
  extended to the next word's start &mdash; so both sides tile the timeline and only
  label disagreement moves the number.</p>
  <div class="tablewrap">
  <table>
    <thead><tr><th>System</th><th>DER, word spans (old)</th><th>DER, matched (new)</th><th>Confusion only</th><th>No timestamps</th></tr></thead>
    <tbody>{"".join(der_rows)}</tbody>
  </table>
  </div>
</section>
{taxonomy_panel(taxonomy)}
{conditions_section(mono, stereo)}
{redaction_panel(redaction_data, leaks, exposure)}
{partner_section(partner_summary or {})}
{jiwer_section(jiwer_summary or {})}
<section>
  <h2>Reading these numbers</h2>
  <div class="caveats">
    <div class="caveat">
      <h3>Coverage differs, so use the common subset</h3>
      <p>The sweeps finished different numbers of visits, so the left-hand block
      compares systems on partly different audio. The right-hand block restricts
      every system to the {len(subset)} visits all of them completed &mdash; that
      is the like-for-like comparison.</p>
    </div>
    <div class="caveat">
      <h3>Absolute WER is not a quality score</h3>
      <p>The ASR is verbatim and the human transcripts semi-verbatim, so WER still
      charges correctly-heard disfluencies as errors. Restricted to the covered span
      that costs about a point, not the thirty it cost before; the level is now
      interpretable, but the error-type split is the honest read.</p>
    </div>
    <div class="caveat">
      <h3>sWER was corrected, and the correction was large</h3>
      <p>A system can score a good pooled WER and still put words on the wrong
      speaker, which is what sWER is for. But as first computed it charged an
      unmatched reference stream a capped 1.0 while charging a badly matched one
      without limit &mdash; so emitting an extra stream cost more than losing a
      speaker outright. One stray one-word speaker line in a human transcript
      scored 88.0 on a single visit. Streams are now capped at 1.0 and reference
      streams under five words are discarded. An earlier version of this page
      reported a 32-point sWER gap on that basis; the corrected medians are
      within half a point of each other.</p>
    </div>
    <div class="caveat">
      <h3>Read the median dot, not the mean</h3>
      <p>These distributions are right-skewed: a minority of hard visits drags the
      mean above typical performance. The wider the gap between the hollow mean marker
      and the filled median dot, the more the headline figure is describing that tail
      rather than a normal session. Where two systems' middle-50% ranges overlap
      substantially, the ranking between them is not established by these 269 visits.</p>
    </div>
  </div>
</section>

<footer>Generated by scripts/plot_results.py from outputs/results.json.</footer>
</div>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="results.json from evaluate_systems.py")
    parser.add_argument("--font", default=None, help="DM Sans .ttf to embed")
    parser.add_argument(
        "--partner", default=None,
        help="partner_wer.json from score_partner_wer.py; adds their metric",
    )
    parser.add_argument(
        "--jiwer", default=None,
        help="jiwer_wer.json from score_jiwer_wer.py; adds the third WER metric",
    )
    parser.add_argument(
        "--mono", default=None,
        help="results.json for the forced-mono condition; adds the condition section",
    )
    parser.add_argument(
        "--stereo", default=None,
        help="results.json restricted to the real-stereo subset",
    )
    parser.add_argument(
        "--redaction", default=None,
        help="redaction.json from score_redaction.py; adds the PII section",
    )
    parser.add_argument(
        "--taxonomy", default=None,
        help="taxonomy.json from error_taxonomy.py; adds the WER decomposition",
    )
    parser.add_argument(
        "--exposure", default=None,
        help="exposure.json from exposure.py; adds the per-transcript panel",
    )
    parser.add_argument(
        "--leaks", default=None,
        help="leak_by_type.json from leak_by_type.py; adds the by-type leak chart",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = load_results(args.results)
    partner_summary = None
    if args.partner:
        partner_summary = merge_partner(data, load_results(args.partner))
    jiwer_summary = None
    if args.jiwer:
        jiwer_summary = merge_jiwer(data, load_results(args.jiwer))
    mono = load_results(args.mono)
    stereo = load_results(args.stereo)
    redaction_data = load_results(args.redaction)
    taxonomy = json.loads(Path(args.taxonomy).read_text()) if args.taxonomy else None
    leaks = json.loads(Path(args.leaks).read_text()) if args.leaks else None
    exposure = json.loads(Path(args.exposure).read_text()) if args.exposure else None
    font_b64 = ""
    if args.font and Path(args.font).exists():
        font_b64 = base64.b64encode(Path(args.font).read_bytes()).decode()

    Path(args.output).write_text(
        build_page(
            data, font_b64, partner_summary, jiwer_summary, mono, stereo,
            redaction_data, taxonomy, leaks, exposure,
        )
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --- PII redaction figures ---------------------------------------------------
#
# Built for the validation run, where several protocol variants are scored on
# one subset, so they take the results dict directly rather than the report's
# merged aggregate.

def _fade(colour: str, opacity: float) -> str:
    """A hex colour blended toward white, so a legend swatch can key an opacity.

    swatch_legend draws solid rectangles; the stacked bars distinguish their
    segments by opacity over one colour. Blending gives the key the same ramp
    the bars use without teaching the legend about transparency.
    """
    r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    mix = lambda c: round(c * opacity + 255 * (1 - opacity))
    return "#%02x%02x%02x" % (mix(r), mix(g), mix(b))


def _label_lines(parts: list[str], limit: int = 34) -> list[str]:
    """System label as wrapped lines, with the joining "+" kept on the break.

    The registry names run to "Gemma 4 31B redaction (turn rewrite, possessive
    rule)", which overflows the label column and is silently clipped at the left
    edge of the figure -- the component simply loses its first words.
    """
    lines: list[str] = []
    for index, part in enumerate(parts):
        text = part + (" +" if index < len(parts) - 1 else "")
        current = ""
        for word in text.split():
            if current and len(current) + len(word) + 1 > limit:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)
    return lines


def _redaction_rows(data: dict | None):
    """Registered systems present in a redaction results file, in registry order."""
    if not data:
        return []
    aggregate = data.get("aggregate", {})
    rows = []
    for name, parts, colour, _ in SYSTEMS:
        stats = registry.entry_of(aggregate, name)
        if stats:
            rows.append((name, parts, colour, stats))
    return rows


def pii_f1_svg(data: dict | None, legend: bool = False) -> tuple[str, int, int]:
    """Recall, precision and F1 per system, as three bars sharing a colour.

    One bar each rather than F1 alone: F1 hides which way a system is wrong, and
    on this corpus the two protocols differ precisely in that -- one finds more
    spans, the other marks fewer things that were never gold.
    """
    rows = _redaction_rows(data)
    if not rows:
        return "", 0, 0

    series = [("recall", "recall", 1.0), ("precision", "precision", 0.62),
              ("F1", "f1", 0.34)]
    bar_h, bar_gap = 11, 3
    line_h = 14
    max_lines = max(len(_label_lines(p)) for _, p, _, _ in rows)
    row_h = max(len(series) * (bar_h + bar_gap), max_lines * line_h + 6)
    gap, pad_l, pad_r, pad_t, pad_b = 16, 250, 54, 22, 30
    plot_w = 330
    height = pad_t + len(rows) * (row_h + gap) + pad_b
    width = pad_l + plot_w + pad_r

    out = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="PII span detection: recall, precision and F1">'
    ]
    for fraction in (0.25, 0.5, 0.75, 1.0):
        gx = pad_l + plot_w * fraction
        out.append(
            f'<line x1="{gx:.1f}" y1="{pad_t - 4}" x2="{gx:.1f}" y2="{height - pad_b}" '
            f'stroke="{GRIDLINE}" stroke-width="1"/>'
            f'<text x="{gx:.1f}" y="{height - pad_b + 14}" text-anchor="middle" '
            f'class="tick">{fraction * 100:.0f}%</text>'
        )

    best = max((s.get("f1") or 0) for _, _, _, s in rows)
    for index, (name, label_parts, colour, stats) in enumerate(rows):
        y = pad_t + index * (row_h + gap)
        lines = _label_lines(label_parts)
        block = len(lines) * line_h
        first = y + (row_h - block) / 2 + line_h - 4
        for line_index, text in enumerate(lines):
            out.append(
                f'<text x="{pad_l - 14}" y="{first + line_index * line_h:.1f}" '
                f'text-anchor="end" class="cat">{escape(text)}</text>'
            )
        top = y + (row_h - len(series) * (bar_h + bar_gap)) / 2
        for series_index, (label, key, opacity) in enumerate(series):
            value = stats.get(key) or 0.0
            by = top + series_index * (bar_h + bar_gap)
            out.append(
                f'<rect x="{pad_l}" y="{by:.1f}" width="{plot_w * value:.1f}" '
                f'height="{bar_h}" fill="{colour}" opacity="{opacity}" rx="1.5"/>'
                f'<text x="{pad_l + plot_w * value + 6:.1f}" y="{by + bar_h - 2:.1f}" '
                f'class="val">{value * 100:.1f}</text>'
            )
        if (stats.get("f1") or 0) == best:
            out.append(
                f'<circle cx="{pad_l - 6}" cy="{y + row_h / 2:.1f}" r="4" fill="{WINNER}"/>'
            )

    tail = 0
    if legend:
        items = [
            ("recall", _fade(RAMP_INK, 1.0),
             "share of the human-marked spans the system redacted"),
            ("precision", _fade(RAMP_INK, 0.62),
             "share of its redactions that landed on a marked span"),
            ("F1", _fade(RAMP_INK, 0.34), "their harmonic mean"),
        ]
        block, tail = swatch_legend(items, width - pad_l, height - pad_b + 26, columns=1)
        out.append(f'<g transform="translate({pad_l - 34},0)">{block}</g>')
        height += tail + 12
        out[0] = (
            f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="PII span detection: recall, precision and F1">'
        )
    out.append("</svg>")
    return "".join(out), width, height


def pii_confusion_svg(data: dict | None, legend: bool = False) -> tuple[str, int, int]:
    """Per-system counts as a matrix, because span detection has no true negatives.

    A 2x2 confusion matrix needs a count of correctly-unredacted spans, and there
    is no such quantity here: the gold marks where PII is, never where it is not,
    so the negatives are every other token in the transcript. The three cells
    that do exist are drawn as a matrix and the fourth is named as undefined,
    rather than being filled with a number that would flatter every system.
    """
    rows = _redaction_rows(data)
    if not rows:
        return "", 0, 0

    columns = [
        ("caught", "true_positives", "gold spans redacted"),
        ("missed", "false_negatives", "gold spans left in the clear"),
        ("extra", "false_positives", "redactions on unmarked text"),
    ]
    cell_w, cell_h, cell_gap = 96, 34, 6
    line_h = 14
    max_lines = max(len(_label_lines(p)) for _, p, _, _ in rows)
    row_h = max(cell_h, max_lines * line_h + 6)
    pad_l, pad_r, pad_t, pad_b = 250, 30, 46, 52
    width = pad_l + len(columns) * (cell_w + cell_gap) + pad_r
    height = pad_t + len(rows) * (row_h + cell_gap) + pad_b

    peak = {
        key: max((r[3].get(key) or 0) for r in rows) or 1
        for _, key, _ in columns
    }

    out = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="PII span detection counts by system">'
    ]
    for column_index, (title, _, _) in enumerate(columns):
        cx = pad_l + column_index * (cell_w + cell_gap) + cell_w / 2
        out.append(
            f'<text x="{cx:.1f}" y="{pad_t - 24}" text-anchor="middle" class="cat">'
            f'{escape(title)}</text>'
        )

    for index, (name, label_parts, colour, stats) in enumerate(rows):
        y = pad_t + index * (row_h + cell_gap)
        lines = _label_lines(label_parts)
        block = len(lines) * line_h
        first = y + (row_h - block) / 2 + line_h - 4
        for line_index, text in enumerate(lines):
            out.append(
                f'<text x="{pad_l - 14}" y="{first + line_index * line_h:.1f}" '
                f'text-anchor="end" class="cat">{escape(text)}</text>'
            )
        for column_index, (_, key, _) in enumerate(columns):
            value = stats.get(key) or 0
            x = pad_l + column_index * (cell_w + cell_gap)
            # Shade by the column's own maximum: the three quantities are on
            # different scales and a shared one would wash out the misses.
            shade = 0.14 + 0.5 * (value / peak[key])
            out.append(
                f'<rect x="{x}" y="{y + (row_h - cell_h) / 2:.1f}" width="{cell_w}" '
                f'height="{cell_h}" rx="3" fill="{colour}" opacity="{shade:.2f}"/>'
                f'<text x="{x + cell_w / 2:.1f}" y="{y + row_h / 2 + 5:.1f}" '
                f'text-anchor="middle" class="cat">{value}</text>'
            )

    # Two short lines rather than one long one: the note ran past the right
    # edge of the figure and was cut mid-sentence.
    notes = [
        f'caught + missed = {rows[0][3].get("gold_spans", 0)} gold spans; '
        f'extra are redactions on unmarked text',
        'no true-negative cell: the annotation marks where PII is, '
        'never where it is not',
    ]
    for note_index, note in enumerate(notes):
        # Anchored at the figure's left margin, not the label column: starting
        # them at pad_l left 336px for an 80-character line and cut both.
        out.append(
            f'<text x="2" y="{height - pad_b + 20 + note_index * 14}" '
            f'class="tick">{escape(note)}</text>'
        )
    out.append("</svg>")
    return "".join(out), width, height


def pii_leak_svg(data: dict | None, kinds: dict | None = None,
                 legend: bool = False) -> tuple[str, int, int]:
    """Leaked spans per system, split by what shape the leaked text is.

    The leak rate is the number a privacy reviewer asks for, but a leak is any
    gold span whose surface survives, and the braces mark material that does not
    identify anyone -- a bare month, a sentence about religion. Splitting the bar
    by shape keeps the headline honest: the dark segment is what still needs a
    human to look at it, the pale segments are shapes no name takes.
    """
    rows = _redaction_rows(data)
    if not rows:
        return "", 0, 0

    order = ["single word", "two or three words", "longer phrase", "month or date", "number"]
    opacity = {"single word": 1.0, "two or three words": 0.55, "longer phrase": 0.42,
               "month or date": 0.28, "number": 0.2}

    line_h = 14
    max_lines = max(len(_label_lines(p)) for _, p, _, _ in rows)
    row_h = max(26, max_lines * line_h + 6)
    gap, pad_l, pad_r, pad_t, pad_b = 16, 250, 96, 22, 30
    plot_w = 300
    height = pad_t + len(rows) * (row_h + gap) + pad_b
    width = pad_l + plot_w + pad_r

    def counts_for(name: str, stats: dict) -> list[tuple[str, int]]:
        entry = registry.entry_of(kinds or {}, name) or {}
        found = entry.get("kinds") or {}
        if found:
            return [(k, found[k]) for k in order if found.get(k)]
        return [("single word", stats.get("leaked") or 0)]

    peak = max(
        max((s.get("leaked") or 0) for _, _, _, s in rows),
        1,
    )
    out = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Leaked identifiers by system">'
    ]
    for fraction in (0.5, 1.0):
        gx = pad_l + plot_w * fraction
        out.append(
            f'<line x1="{gx:.1f}" y1="{pad_t - 4}" x2="{gx:.1f}" y2="{height - pad_b}" '
            f'stroke="{GRIDLINE}" stroke-width="1"/>'
            f'<text x="{gx:.1f}" y="{height - pad_b + 14}" text-anchor="middle" '
            f'class="tick">{peak * fraction:.0f}</text>'
        )

    for index, (name, label_parts, colour, stats) in enumerate(rows):
        y = pad_t + index * (row_h + gap)
        lines = _label_lines(label_parts)
        block = len(lines) * line_h
        first = y + (row_h - block) / 2 + line_h - 4
        for line_index, text in enumerate(lines):
            out.append(
                f'<text x="{pad_l - 14}" y="{first + line_index * line_h:.1f}" '
                f'text-anchor="end" class="cat">{escape(text)}</text>'
            )
        x = pad_l
        cy = y + (row_h - 14) / 2
        for kind_name, count in counts_for(name, stats):
            w = plot_w * count / peak
            out.append(
                f'<rect x="{x:.1f}" y="{cy:.1f}" width="{max(w, 0.6):.1f}" height="14" '
                f'fill="{colour}" opacity="{opacity.get(kind_name, 0.5)}"/>'
            )
            x += w
        rate = stats.get("leak_rate")
        testable = stats.get("leak_testable") or 0
        label = f'{stats.get("leaked") or 0} of {testable}'
        if rate is not None:
            label += f'  ({rate * 100:.1f}%)'
        out.append(
            f'<text x="{x + 8:.1f}" y="{cy + 11:.1f}" class="val">{label}</text>'
        )

    tail = 0
    if legend:
        block, tail = swatch_legend(
            [(k, _fade(RAMP_INK, opacity[k]), "") for k in order],
            width - pad_l, height - pad_b + 26, columns=3,
        )
        out.append(f'<g transform="translate({pad_l - 34},0)">{block}</g>')
        height += tail + 12
        out[0] = (
            f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="Leaked identifiers by system">'
        )
    out.append("</svg>")
    return "".join(out), width, height
