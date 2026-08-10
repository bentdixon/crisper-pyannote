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
from pathlib import Path

# --- design tokens (from design_language.md, Palette 1) ---------------------
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
TEXT = "#0b0b0b"
MUTED = "#5a5a56"
MUTED_DARK = "#3a3a37"
WINNER = "#1baf7a"

# Per-system colour is held constant across every panel so a system can be
# tracked by eye between charts. Green is deliberately NOT a system colour --
# the design language reserves it for the winner marker.
# Every label names the ASR model, then the diarization system, then the LLM
# reviewer when one is used, joined by "+" so the composition of each system is
# readable without reference to the prose.
SYSTEMS = [
    (
        "chirp3",
        ["Chirp-3 ASR", "Chirp-3 diarization"],
        "#898781",
        "incumbent, as delivered by the bucket",
    ),
    (
        "verbatimize",
        ["Chirp-3 ASR", "CrisperWhisper 2.0 verbatimize", "Chirp-3 diarization"],
        "#e87ba4",
        "Chirp text kept, disfluencies inserted by CW2",
    ),
    (
        "ours",
        ["CrisperWhisper 2.0 ASR", "pyannote community-1"],
        "#2a78d6",
        "our pipeline",
    ),
    (
        "ours_llm",
        ["CrisperWhisper 2.0 ASR", "pyannote community-1", "Qwen2.5-7B-Instruct"],
        "#7fb0e8",
        "our pipeline, LLM review applied",
    ),
    (
        "baseline",
        ["CrisperWhisper 2.0 ASR", "pyannote 3.1"],
        "#eda100",
        "other team's pipeline, ported to CW2",
    ),
    (
        "baseline_llm",
        ["CrisperWhisper 2.0 ASR", "pyannote 3.1", "Qwen2.5-7B-Instruct"],
        "#f5cc6b",
        "their pipeline, LLM review applied",
    ),
]

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

# Full figure captions, keyed by aggregate key. These travel with the exported
# charts, so each one has to stand on its own: what the metric is, how it is
# computed here, and the caveat that decides how far it can be trusted.
CAPTIONS = {
    "WER": (
        "Substitutions plus deletions plus insertions, divided by the number of "
        "reference words, after filled pauses are removed from both sides. WER is "
        "not capped at 100%: a system that inserts more words than the reference "
        "contains scores above it. On this corpus the machine transcribes verbatim "
        "while the human transcripts are semi-verbatim, so every system emits about "
        "a third more words than the reference and insertions dominate the total for "
        "all of them. Read the ordering between systems, not the absolute level, and "
        "note that the most faithful transcriber is penalised most."
    ),
    "WER_no_ins": (
        "Substitutions plus deletions only: the share of reference words the machine "
        "got wrong or missed outright. Dropping insertions removes the verbatim "
        "versus semi-verbatim mismatch, which inflates every system's WER by roughly "
        "the same amount and has nothing to do with transcription accuracy. This is "
        "the fairer measure of how well each system heard the words that both the "
        "machine and the human agreed were said."
    ),
    "sWER": (
        "Speaker-attributed word error rate. Reference and predicted speaker streams "
        "are matched by a Hungarian assignment on their time-overlap matrix, WER is "
        "then computed within each reference stream, and the per-stream scores are "
        "averaged. A system can score a respectable pooled WER and still put words in "
        "the wrong speaker's mouth; the gap between a system's WER and its sWER is a "
        "direct measure of that misattribution."
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
        # (chirp3 DER: mean 0.691, p75 0.337) -- so a bar growing from zero both
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


def composition_panel(aggregate: dict, present: list[str]) -> str:
    """Stacked WER composition: insertions vs substitutions vs deletions.

    WER is the one number on this page that is a sum of parts, so it is the one
    place a stacked bar is the right mark. It exists to answer the question the
    single WER figure cannot: how much of the error is the machine hearing
    fillers the human transcript leaves out, rather than mishearing words.
    """
    rows = [(n, parts, colour) for n, parts, colour, _ in SYSTEMS if n in present]
    rows = [r for r in rows if aggregate.get(r[0], {}).get("WER_ins") is not None]
    if not rows:
        return ""

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

    out = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="WER composition by system">'
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
        f'stroke="{AXIS}" stroke-width="1"/></svg>'
    )

    swatches = "".join(
        f'<span><span class="sw" style="background:{colour}"></span>'
        f'<b>{escape(label)}</b> &mdash; {escape(note)}</span>'
        for label, _, colour, note in WER_PARTS
    )
    return (
        '<figure class="panel wide">'
        '<figcaption><h3>What makes up the word error rate</h3>'
        '<span class="dir">lower is better</span>'
        '<p>The three error types sum to WER. Insertions dominate because the '
        'machine transcribes disfluencies the human transcripts leave out.</p>'
        '</figcaption>'
        + "".join(out)
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


def build_page(data: dict, font_b64: str) -> str:
    aggregate = data.get("aggregate", {})
    per_visit = data.get("per_visit", {})
    present = [n for n, *_ in SYSTEMS if n in aggregate]
    subset = common_subset(per_visit, present)

    panels = []
    for title, key, agg_key, direction, caption in METRICS:
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
        for _, _, agg_key, _, _ in METRICS:
            value = stats.get(agg_key)
            cells.append(f'<td class="num">{value * 100:.1f}%</td>' if value is not None else '<td class="num">-</td>')
        for _, key, _, _, _ in METRICS:
            value = subset_mean(per_visit, subset, name, key)
            cells.append(f'<td class="num sub">{value * 100:.1f}%</td>' if value is not None else '<td class="num sub">-</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    metric_heads = "".join(f"<th>{escape(t)}</th>" for t, *_ in METRICS)

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

<section>
  <h2>How to read these charts</h2>
  <div class="howto">
    <div>
      <h3>Every figure is a percentage</h3>
      <p>All six metrics are rates. A DER of 20.5% means a fifth of speech time
      is on the wrong speaker &mdash; not a fifth of a percent. WER can exceed
      100%, because a system can insert more words than the reference contains.</p>
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
      <p>The machine transcribes verbatim and the human transcripts are
      semi-verbatim, so every system emits about a third more words than the
      reference. That inflates every WER before a single word is misheard. Use
      <b>WER excluding insertions</b> for transcription quality.</p>
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
      <tr><th></th><th></th><th colspan="4">All scored visits</th><th colspan="4">Common subset (n={len(subset)})</th></tr>
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
      <p>The ASR is verbatim and the human transcripts are semi-verbatim, so WER
      runs above 1.0 and penalises correctly-heard disfluencies. Differences
      between systems are meaningful; the absolute level is not.</p>
    </div>
    <div class="caveat">
      <h3>sWER is the speaker-attribution number</h3>
      <p>A system can score a good pooled WER and still put words on the wrong
      speaker. sWER averages WER over reference streams after matching streams by
      time overlap, so it captures misattribution that WER hides.</p>
    </div>
    <div class="caveat">
      <h3>Read the median dot, not the mean</h3>
      <p>These distributions are strongly right-skewed: a minority of catastrophic
      visits drags the mean well above typical performance, and in several cases
      past the system's own p75 &mdash; Chirp-3's mean DER is 0.691 while its median
      visit is 0.216. The wider the gap between the hollow mean marker and the
      filled median dot, the more the headline figure is describing that tail.
      Where two systems' ranges overlap substantially, the ranking between them is
      not established.</p>
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text())
    font_b64 = ""
    if args.font and Path(args.font).exists():
        font_b64 = base64.b64encode(Path(args.font).read_bytes()).decode()

    Path(args.output).write_text(build_page(data, font_b64))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
