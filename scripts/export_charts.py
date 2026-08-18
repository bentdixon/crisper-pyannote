"""Export each evaluation chart as a standalone HTML file and a matching PNG.

Both formats come from the same SVG that the combined report uses, so a chart
cannot drift between the page and the image. The PNG is produced by
screenshotting the HTML in headless Chrome rather than by a second rendering
library, for the same reason: one renderer, one result.

The page is sized to the chart exactly -- the window passed to Chrome is
computed from the SVG's own dimensions plus the caption block -- so the PNG
needs no cropping and carries no stray margin.

Usage:
    uv run python scripts/export_charts.py results.json --output-dir charts \
        --font dmsans.ttf
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_results import (  # noqa: E402
    CAPTIONS,
    GRIDLINE,
    MUTED,
    MUTED_DARK,
    TEXT,
    WINNER,
    chart_svg,
    collect,
    comparison_svg,
    composition_svg,
    escape,
    load_results,
    merge_jiwer,
    merge_partner,
    metrics_for,
    quartiles,
    exposure_svg,
    pii_confusion_svg,
    pii_f1_svg,
    pii_leak_svg,
    leak_type_svg,
    redaction_svg,
    taxonomy_svg,
)

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# The page is padding plus the SVG and nothing else, so the screenshot canvas
# is arithmetic rather than a prediction of how HTML will lay out.
PAD = 20


# Chrome's --window-size counts browser UI, so the usable viewport is ~88px
# shorter than asked for and the bottom of every export is silently dropped --
# axis labels and legend simply absent, with no error. Rather than depend on
# that offset staying constant, render with slack and crop to the exact size.
VIEWPORT_SLACK = 240


def slug(title: str) -> str:
    return title.lower().replace(" ", "-").replace("/", "-")


def decode_png(path: Path) -> tuple[int, int, int, list[bytearray]]:
    """Minimal PNG reader for 8-bit truecolour, enough to crop a screenshot."""
    raw = path.read_bytes()
    pos, idat = 8, b""
    width = height = channels = 0
    while pos < len(raw):
        length = struct.unpack(">I", raw[pos:pos + 4])[0]
        kind = raw[pos + 4:pos + 8]
        data = raw[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", data[:10])
            if depth != 8 or colour not in (2, 6):
                raise ValueError(f"unsupported PNG: depth {depth}, colour {colour}")
            channels = 3 if colour == 2 else 4
        elif kind == b"IDAT":
            idat += data
        pos += 12 + length

    buf = zlib.decompress(idat)
    stride = width * channels
    rows: list[bytearray] = []
    previous = bytearray(stride)
    offset = 0
    for _ in range(height):
        filter_type = buf[offset]
        offset += 1
        line = bytearray(buf[offset:offset + stride])
        offset += stride
        for x in range(stride):
            left = line[x - channels] if x >= channels else 0
            up = previous[x]
            upleft = previous[x - channels] if x >= channels else 0
            if filter_type == 1:
                line[x] = (line[x] + left) & 255
            elif filter_type == 2:
                line[x] = (line[x] + up) & 255
            elif filter_type == 3:
                line[x] = (line[x] + ((left + up) >> 1)) & 255
            elif filter_type == 4:
                p = left + up - upleft
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - upleft)
                pred = left if (pa <= pb and pa <= pc) else (up if pb <= pc else upleft)
                line[x] = (line[x] + pred) & 255
        rows.append(line)
        previous = line
    return width, height, channels, rows


def encode_png(path: Path, width: int, height: int, channels: int, rows) -> None:
    raw = b"".join(b"\x00" + bytes(row[: width * channels]) for row in rows[:height])
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )
    header = struct.pack(">IIBBBBB", width, height, 8, 2 if channels == 3 else 6, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def crop_png(path: Path, width: int, height: int) -> None:
    actual_w, actual_h, channels, rows = decode_png(path)
    if actual_w == width and actual_h == height:
        return
    encode_png(path, min(width, actual_w), min(height, actual_h), channels, rows)


def chart_page(title: str, svg: str, width: int, height: int, font_b64: str) -> str:
    """A page whose entire content is one fixed-size SVG.

    Title, caption and legend live inside the SVG (standalone=True), so there
    is no HTML flow around it whose height has to be guessed. An earlier
    version composed those in HTML and cropped the axis labels out of every
    PNG at 2x device scale.
    """
    face = (
        f"@font-face{{font-family:'DM Sans';"
        f"src:url(data:font/ttf;base64,{font_b64}) format('truetype');"
        f"font-weight:100 1000;font-display:block;}}"
        if font_b64 else ""
    )
    # The padding lives inside an outer SVG rather than as a CSS margin: a
    # margin on the only child collapses through the body, shifting the whole
    # page down and pushing the axis labels and legend out of the viewport.
    # With the padding in the SVG's own coordinate space the page height is
    # exactly the image height.
    outer = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width + PAD * 2}" '
        f'height="{height + PAD * 2}" viewBox="0 0 {width + PAD * 2} {height + PAD * 2}">'
        f'<rect width="100%" height="100%" fill="#ffffff"/>'
        f'<g transform="translate({PAD},{PAD})">{svg}</g></svg>'
    )
    return f"""<title>{escape(title)}</title>
<style>
{face}
html, body {{ margin: 0; padding: 0; background: #ffffff; }}
svg {{ display: block; }}
svg text {{ font-family: 'DM Sans', sans-serif; font-weight: 500; }}
svg .ttl {{ font-size: 17px; font-weight: 600; fill: {TEXT}; }}
svg .sub {{ font-size: 10.5px; letter-spacing: 0.1em; fill: {MUTED}; }}
svg .cap {{ font-size: 12.5px; fill: {MUTED_DARK}; }}
svg .cat {{ font-size: 12px; fill: {TEXT}; }}
svg .val {{ font-size: 11px; fill: {MUTED}; }}
svg .tick {{ font-size: 10px; fill: {MUTED}; }}
svg .leg {{ font-size: 11px; fill: {MUTED_DARK}; }}
svg .figcap {{ font-size: 10.5px; fill: {MUTED}; }}
</style>
{outer}
"""


def titled(title: str, subtitle: str, svg: str, width: int, height: int,
           caption: str, clean: bool = False) -> tuple[str, int, int]:
    """Wrap a bare chart SVG in the title block and caption the metric charts
    build for themselves.

    chart_svg renders its own heading when standalone; the composition,
    mono/stereo and redaction figures are built for the report page, where the
    heading is HTML. Exported alone they would arrive as an unlabelled diagram,
    so the same furniture is composed around them here via a nested <svg>.
    """
    def wrap(text: str, limit: int) -> list[str]:
        lines, line = [], ""
        for word in text.split():
            if len(line) + len(word) + 1 > limit:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            lines.append(line)
        return lines

    if clean:
        # "raw metrics and keys": the explanatory paragraph goes, the title,
        # the direction and the colour key stay -- a figure with no key cannot
        # be read at all.
        caption = ""
    head = wrap(subtitle, 82)
    head_h = 40 + len(head) * 15
    foot = wrap(caption, 96)
    foot_h = (18 + len(foot) * 14) if foot else 0
    total_h = head_h + height + foot_h
    total_w = max(width, 640)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {total_h}" '
        f'width="{total_w}" height="{total_h}">',
        f'<text x="0" y="17" class="ttl">{escape(title)}</text>',
    ]
    for index, line in enumerate(head):
        parts.append(f'<text x="0" y="{38 + index * 15}" class="cap">{escape(line)}</text>')
    parts.append(f'<g transform="translate(0,{head_h})">{svg}</g>')
    for index, line in enumerate(foot):
        parts.append(
            f'<text x="0" y="{head_h + height + 14 + index * 14}" class="figcap">'
            f'{escape(line)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts), total_w, total_h


def validation_figures(validation: dict | None, kinds: dict | None,
                       clean: bool = False) -> list[tuple[str, str, int, int]]:
    """The three PII figures for a redaction validation run."""
    out = []
    svg, w, h = pii_f1_svg(validation, legend=True)
    if not svg:
        return out
    visits = max((e.get("visits", 0) for e in validation["aggregate"].values()), default=0)
    gold = max((e.get("gold_spans", 0) for e in validation["aggregate"].values()), default=0)
    out.append(("pii-detection-f1", *titled(
        "PII span detection", "higher is better",
        svg, w, h,
        f"Span-level detection against the human transcripts' own annotation, over "
        f"{visits} transcripts and {gold} gold spans, each scored on the span the "
        f"transcript covers. Recall is the share of marked spans a system redacted; "
        f"precision the share of its redactions that landed on one. Precision is a "
        f"lower bound throughout: an identifier nobody marked counts against the "
        f"system that caught it, and only some transcripts carry any annotation.",
        clean=clean,
    )))
    svg, w, h = pii_confusion_svg(validation)
    out.append(("pii-confusion", *titled(
        "Where each redactor's decisions land", "counts, not rates",
        svg, w, h,
        "The three outcomes span detection admits. There is no true-negative cell: "
        "the annotation records where PII is, never where it is not, so the negatives "
        "are every other token in the transcript and any accuracy figure built on "
        "them would be dominated by ordinary speech. Cells are shaded within their "
        "own column, because caught, missed and extra are on different scales.",
        clean=clean,
    )))
    svg, w, h = pii_leak_svg(validation, kinds, legend=True)
    out.append(("pii-leak-rate", *titled(
        "Identifiers leaked", "lower is better",
        svg, w, h,
        "A leak is a gold span whose surface form survives verbatim in the output -- "
        "the privacy question asked directly, with no alignment step to trust. The "
        "split is by the shape of the leaked text, not by a verdict on it: the "
        "annotation marks material that identifies nobody, so a bare month or a "
        "sentence about religion counts as a leak while being harmless. The dark "
        "segment is single words, which is where a name would be and what still "
        "needs a human read.",
        clean=clean,
    )))
    return out


def extra_figures(data: dict, mono: dict | None, stereo: dict | None,
                  redaction: dict | None, taxonomy: dict | None = None,
                  clean: bool = False, leaks: dict | None = None,
                  exposure: dict | None = None,
                  ) -> list[tuple[str, str, int, int]]:
    """Figures the report builds as HTML sections, as standalone charts."""
    out = []
    aggregate = data.get("aggregate", {})
    present = [n for n, *_ in __import__("plot_results").SYSTEMS if n in aggregate]

    svg, w, h = taxonomy_svg(taxonomy, legend=True)
    if svg:
        out.append(("wer-error-types", *titled(
            "What the word error rate is made of",
            "lower is better; segments sum to the reported WER",
            svg, w, h,
            "Filled pauses are removed from both sides before scoring, so um and uh "
            "contribute nothing here, and scoring is restricted to the span each human "
            "transcript covers -- they stop early on a third of this cohort, most of "
            "those at a hard 32-minute cutoff, and including the remainder inflated "
            "WER from roughly 12% to roughly 42%. Warm segments are transcription "
            "error: words heard wrong or missed. Cool segments are speech the machine "
            "produced that the transcript does not record, a convention difference "
            "rather than a failure.", clean=clean,
        )))

    svg, w, h = composition_svg(aggregate, present, legend=True)
    if svg:
        out.append(("wer-composition", *titled(
            "What makes up the word error rate", "lower is better",
            svg, w, h, CAPTIONS["composition"], clean=clean,
        )))

    if mono and stereo:
        for title, key, slug in [
            ("Transcription accuracy: mono against stereo-container files",
             "WER_no_ins", "mono-vs-stereo-wer"),
            ("Speaker confusion: mono against stereo-container files",
             "DER_confusion", "mono-vs-stereo-der"),
        ]:
            svg, w, h = comparison_svg(mono, stereo, title, key)
            if svg:
                out.append((slug, *titled(
                    title, "lower is better; the number at the right is the shift in points",
                    svg, w, h,
                    "All 66 stereo files measure 0.000-0.082 dB of channel separation "
                    "against the 3.0 dB threshold their pipeline requires, so they are "
                    "stereo containers holding duplicated mono and no system read "
                    "speakers off a channel. This compares recording provenance, not "
                    "channel access, and the shift between conditions is larger than "
                    "any gap between systems.", clean=clean,
                )))

    svg, w, h = exposure_svg(exposure, legend=True)
    if svg:
        out.append(("pii-exposure-per-transcript", *titled(
            "Per-transcript PII exposure",
            "how many known identifier locations each transcript still leaves open",
            svg, w, h,
            "A PII location is any position the human transcriber marked -- including "
            "the 577 spans already scrubbed to {redacted}, which no verbatim search can "
            "test -- or where another system emitted a placeholder. A system is exposed "
            "where it redacted nothing. Judged leave-one-out, so a system is never "
            "scored on locations it proposed itself; including them hands the win to "
            "whichever redactor is most aggressive.", clean=clean,
        )))

    svg, w, h = leak_type_svg(leaks, legend=True)
    if svg:
        out.append(("pii-leak-by-type", *titled(
            "Leak rate by identifier type",
            "lower is better; share of gold spans surviving verbatim in the output",
            svg, w, h,
            "Each gold span is typed by what a system called it when some system caught "
            "it, pooled across every system and visit, which types 91 of 102 distinct "
            "surface forms. Only types with at least 20 gold spans get a rate: dates (4) "
            "and ages (2) are too few. Denominators count only spans whose surface form "
            "the transcriber left intact.", clean=clean,
        )))

    if redaction:
        svg, w, h = redaction_svg(redaction)
        if svg:
            out.append(("pii-redaction", *titled(
                "PII redaction: over and under the human annotation",
                "left of centre is left in the clear, right is redacted beyond the annotation",
                svg, w, h,
                "Gold spans are the curly braces in the human transcripts: 887 across "
                "142 of 269 sessions. Recall is trustworthy because a gold span is real "
                "PII; precision is a lower bound because only half the transcripts carry "
                "any annotation, so a genuine identifier nobody marked counts against a "
                "system that caught it.", clean=clean,
            )))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--font", default=None)
    parser.add_argument(
        "--partner", default=None,
        help="partner_wer.json from score_partner_wer.py; adds their chart",
    )
    parser.add_argument(
        "--validation", default=None,
        help="redaction results for a validation run; adds the three PII figures",
    )
    parser.add_argument(
        "--leak-kinds", default=None,
        help="leak_kinds.json from classify_leaks.py; splits the leak bars by shape",
    )
    parser.add_argument(
        "--jiwer", default=None,
        help="jiwer_wer.json from score_jiwer_wer.py; adds the third WER chart",
    )
    parser.add_argument("--mono", default=None, help="results.json for the mono subset")
    parser.add_argument("--stereo", default=None, help="results.json for the stereo subset")
    parser.add_argument("--redaction", default=None, help="redaction.json")
    parser.add_argument("--taxonomy", default=None, help="taxonomy.json")
    parser.add_argument("--leaks", default=None, help="leak_by_type.json")
    parser.add_argument("--exposure", default=None, help="exposure.json")
    parser.add_argument(
        "--clean", action="store_true",
        help=(
            "drop the explanatory caption from every figure, keeping the title, "
            "the direction and the colour key -- for slides and papers where the "
            "surrounding text supplies the context"
        ),
    )
    parser.add_argument("--scale", type=float, default=2.0, help="PNG device scale")
    parser.add_argument("--chrome", default=CHROME)
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()

    data = load_results(args.results)
    if args.partner:
        merge_partner(data, load_results(args.partner))
    if args.jiwer:
        merge_jiwer(data, load_results(args.jiwer))
    mono = load_results(args.mono)
    stereo = load_results(args.stereo)
    redaction = load_results(args.redaction)
    leaks = json.loads(Path(args.leaks).read_text()) if args.leaks else None
    exposure = json.loads(Path(args.exposure).read_text()) if args.exposure else None
    taxonomy = json.loads(Path(args.taxonomy).read_text()) if args.taxonomy else None
    aggregate = data.get("aggregate", {})
    per_visit = data.get("per_visit", {})
    present = [n for n in aggregate]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    font_b64 = ""
    if args.font and Path(args.font).exists():
        font_b64 = base64.b64encode(Path(args.font).read_bytes()).decode()

    chrome_available = Path(args.chrome).exists()
    if not args.no_png and not chrome_available:
        print(f"warning: {args.chrome} not found; writing HTML only", file=sys.stderr)

    written = []

    def emit(name: str, title: str, svg: str, width: int, height: int) -> None:
        html_path = out / f"{name}.html"
        html_path.write_text(chart_page(title, svg, width, height, font_b64))
        written.append(html_path)
        if args.no_png or not chrome_available:
            return
        page_w, page_h = width + PAD * 2, height + PAD * 2
        png_path = out / f"{name}.png"
        result = subprocess.run(
            [
                args.chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
                f"--force-device-scale-factor={args.scale}",
                f"--window-size={page_w},{page_h + VIEWPORT_SLACK}",
                f"--screenshot={png_path}",
                html_path.resolve().as_uri(),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not png_path.exists():
            print(f"error: PNG failed for {name}: {result.stderr.strip()[:200]}",
                  file=sys.stderr)
            return
        crop_png(png_path, int(page_w * args.scale), int(page_h * args.scale))
        written.append(png_path)

    for title, key, agg_key, direction, caption in metrics_for(aggregate):
        if args.clean:
            caption = None
        values = {n: aggregate.get(n, {}).get(agg_key) for n in present}
        raw = collect(per_visit, key)
        spreads = {n: quartiles(raw.get(n, [])) for n in present}
        svg, width, height = chart_svg(
            title, values, spreads, present, direction, caption=caption,
            standalone=True,
            footer="" if args.clean else CAPTIONS.get(agg_key, ""),
        )
        if not svg:
            continue

        emit(slug(title), title, svg, width, height)

    validation = load_results(args.validation)
    kinds = json.loads(Path(args.leak_kinds).read_text()) if args.leak_kinds else None
    for name, svg, width, height in (
        extra_figures(data, mono, stereo, redaction, taxonomy, clean=args.clean,
                      leaks=leaks, exposure=exposure)
        + validation_figures(validation, kinds, clean=args.clean)
    ):
        emit(name, name.replace('-', ' '), svg, width, height)

    for path in written:
        print(f"  {path}  ({path.stat().st_size // 1024} KB)")
    print(f"\n{len(written)} file(s) in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
