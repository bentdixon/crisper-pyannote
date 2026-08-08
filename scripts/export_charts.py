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
    GRIDLINE,
    METRICS,
    MUTED,
    MUTED_DARK,
    TEXT,
    WINNER,
    chart_svg,
    collect,
    escape,
    quartiles,
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
</style>
{outer}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--font", default=None)
    parser.add_argument("--scale", type=float, default=2.0, help="PNG device scale")
    parser.add_argument("--chrome", default=CHROME)
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text())
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
    for title, key, agg_key, direction, caption in METRICS:
        values = {n: aggregate.get(n, {}).get(agg_key) for n in present}
        raw = collect(per_visit, key)
        spreads = {n: quartiles(raw.get(n, [])) for n in present}
        svg, width, height = chart_svg(
            title, values, spreads, present, direction, caption=caption, standalone=True
        )
        if not svg:
            continue

        name = slug(title)
        html_path = out / f"{name}.html"
        html_path.write_text(chart_page(title, svg, width, height, font_b64))
        written.append(html_path)

        if args.no_png or not chrome_available:
            continue

        page_w = width + PAD * 2
        page_h = height + PAD * 2
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
            print(
                f"error: PNG failed for {name}: {result.stderr.strip()[:200]}",
                file=sys.stderr,
            )
            continue
        crop_png(png_path, int(page_w * args.scale), int(page_h * args.scale))
        written.append(png_path)

    for path in written:
        print(f"  {path}  ({path.stat().st_size // 1024} KB)")
    print(f"\n{len(written)} file(s) in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
