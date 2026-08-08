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

METRICS = [
    ("WER", "wer", "lower is better", "word error rate, filled pauses removed"),
    ("sWER", "swer", "lower is better", "speaker-attributed WER (word misattribution)"),
    ("DER", "der", "lower is better", "diarization error rate"),
    ("QTP-F1", "qtp_f1", "higher is better", "qualifier / temporal cue preservation"),
]


def quartiles(values: list[float]) -> tuple[float, float] | None:
    """p25/p75 of a per-visit metric, or None when there is too little data."""
    clean = sorted(v for v in values if v is not None)
    if len(clean) < 4:
        return None
    return (
        statistics.quantiles(clean, n=4)[0],
        statistics.quantiles(clean, n=4)[2],
    )


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


def bar_panel(
    title: str,
    direction: str,
    caption: str,
    values: dict[str, float | None],
    spreads: dict[str, tuple[float, float] | None],
    present: list[str],
) -> str:
    """One horizontal-bar small multiple as inline SVG.

    Horizontal because six system names do not fit legibly under vertical
    bars. Gridlines sit on the value axis only and are drawn before the bars
    so the data reads on top of them.
    """
    rows = [(n, parts, colour) for n, parts, colour, _ in SYSTEMS if n in present]
    if not rows:
        return ""

    # Labels are multi-line (one component per line), so the row has to be tall
    # enough for the longest label and the label gutter wide enough for the
    # longest component -- otherwise names silently overlap or clip.
    line_h = 14
    max_lines = max(len(parts) for _, parts, _ in rows)
    row_h = max(30, max_lines * line_h + 6)
    gap, pad_l, pad_r, pad_t, pad_b = 12, 250, 52, 14, 30
    plot_w = 300
    height = pad_t + len(rows) * (row_h + gap) + pad_b
    width = pad_l + plot_w + pad_r

    numeric = [v for v in values.values() if v is not None]
    spread_max = [s[1] for s in spreads.values() if s is not None]
    top = max(numeric + spread_max) if (numeric or spread_max) else 1.0
    top = max(top * 1.1, 0.05)

    better_low = direction.startswith("lower")
    ranked = [(n, v) for n, v in values.items() if v is not None]
    best = None
    if ranked:
        best = (min if better_low else max)(ranked, key=lambda kv: kv[1])[0]

    def x(value: float) -> float:
        return pad_l + (value / top) * plot_w

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="{escape(title)} by system">'
    ]

    # gridlines + value-axis ticks, behind the data
    ticks = 5
    for i in range(ticks + 1):
        value = top * i / ticks
        gx = round(x(value), 1)
        parts.append(
            f'<line x1="{gx}" y1="{pad_t}" x2="{gx}" y2="{height - pad_b}" '
            f'stroke="{GRIDLINE}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx}" y="{height - pad_b + 15}" text-anchor="middle" '
            f'class="tick">{value:.2f}</text>'
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

        bw = max(x(value) - pad_l, 1)
        parts.append(
            f'<rect x="{pad_l}" y="{y}" width="{bw:.1f}" height="{row_h}" '
            f'rx="3" fill="{colour}"/>'
        )

        spread = spreads.get(name)
        if spread:
            lo, hi = x(spread[0]), x(spread[1])
            cy = y + row_h / 2
            parts.append(
                f'<line x1="{lo:.1f}" y1="{cy}" x2="{hi:.1f}" y2="{cy}" '
                f'stroke="{MUTED_DARK}" stroke-width="1.2" opacity="0.55"/>'
            )
            for cap in (lo, hi):
                parts.append(
                    f'<line x1="{cap:.1f}" y1="{cy - 5}" x2="{cap:.1f}" y2="{cy + 5}" '
                    f'stroke="{MUTED_DARK}" stroke-width="1.2" opacity="0.55"/>'
                )

        label_x = min(x(value), pad_l + plot_w) + 7
        parts.append(
            f'<text x="{label_x:.1f}" y="{y + row_h / 2 + 4}" class="val">{value:.3f}</text>'
        )
        if name == best:
            parts.append(
                f'<circle cx="{pad_l - 12}" cy="{y + row_h / 2}" r="4" fill="{WINNER}"/>'
            )

    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    parts.append("</svg>")

    return (
        '<figure class="panel">'
        f'<figcaption><h3>{escape(title)}</h3>'
        f'<span class="dir">{escape(direction)}</span>'
        f'<p>{escape(caption)}</p></figcaption>'
        + "".join(parts)
        + "</figure>"
    )


def build_page(data: dict, font_b64: str) -> str:
    aggregate = data.get("aggregate", {})
    per_visit = data.get("per_visit", {})
    present = [n for n, *_ in SYSTEMS if n in aggregate]
    subset = common_subset(per_visit, present)

    panels = []
    for title, key, direction, caption in METRICS:
        agg_key = {"wer": "WER", "swer": "sWER", "der": "DER", "qtp_f1": "QTP_F1"}[key]
        values = {n: aggregate.get(n, {}).get(agg_key) for n in present}
        raw = collect(per_visit, key)
        spreads = {n: quartiles(raw.get(n, [])) for n in present}
        panels.append(bar_panel(title, direction, caption, values, spreads, present))

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
        for _, key, _, _ in METRICS:
            agg_key = {"wer": "WER", "swer": "sWER", "der": "DER", "qtp_f1": "QTP_F1"}[key]
            value = stats.get(agg_key)
            cells.append(f'<td class="num">{value:.4f}</td>' if value is not None else '<td class="num">-</td>')
        for _, key, _, _ in METRICS:
            value = subset_mean(per_visit, subset, name, key)
            cells.append(f'<td class="num sub">{value:.4f}</td>' if value is not None else '<td class="num sub">-</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    metric_heads = "".join(f"<th>{escape(t)}</th>" for t, *_ in METRICS)

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
svg .cat {{ font-family: 'DM Sans', sans-serif; font-size: 12px; font-weight: 500; fill: var(--text); }}
svg .val {{ font-family: 'DM Sans', sans-serif; font-size: 11px; font-weight: 500; fill: var(--muted); }}
svg .tick {{ font-family: 'DM Sans', sans-serif; font-size: 10px; font-weight: 500; fill: var(--muted); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px 22px; font-size: 12.5px; color: var(--muted-dark); margin-top: 14px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
.dot {{ width: 9px; height: 9px; border-radius: 50%; background: var(--winner); }}
.whisker {{ width: 22px; height: 1px; background: var(--muted-dark); opacity: 0.6; }}
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
  <h2>Metrics by system</h2>
  <div class="grid">{"".join(panels)}</div>
  <div class="legend">
    <span><span class="dot"></span>best on this metric</span>
    <span><span class="whisker"></span>p25&ndash;p75 across visits</span>
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
      <h3>Spread is wide relative to the gaps</h3>
      <p>The whiskers span the middle half of visits. Where two bars differ by
      less than that spread, the ranking between them is not established by these
      means alone.</p>
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
