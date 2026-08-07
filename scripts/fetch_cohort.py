"""Fetch the audio / Chirp-3 / human-transcript cohort from the Pronet bucket.

Only sessions that have all three artefacts are downloaded, so the cohort is
directly usable for verbatimizing (audio + Chirp) and for later scoring
against the human transcripts. Everything pulled here is the redacted
variant: final_gcp_transcripts and human_transcription/*_REDACTED.txt.

The bucket lays each session out as
    <bucket>/<SITE>/<SUBJECT>/audio/<stem>.wav
    <bucket>/<SITE>/<SUBJECT>/final_gcp_transcripts/<stem>_final.json
    <bucket>/<SITE>/<SUBJECT>/human_transcription/<stem>_REDACTED.txt
where the audio stem additionally carries an "interviewAudioTranscript"
token, so files are paired on a normalized session key.

Usage:
    uv run python scripts/fetch_cohort.py --dest data/cohort --account you@example.edu
    uv run python scripts/fetch_cohort.py --dest data/cohort --limit 2 --skip audio
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("fetch_cohort")

BUCKET = "gs://pronet_data/NDA_4"
CATEGORIES = {
    "audio": ("audio", ".wav"),
    "chirp": ("final_gcp_transcripts", ".json"),
    "human": ("human_transcription", ".txt"),
}
NOISE = re.compile(
    r"(_interviewAudioTranscript|_final_humanReadable|_final|_REDACTED|_UNREDACTED)",
    re.IGNORECASE,
)


def session_key(site: str, subject: str, filename: str) -> str:
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename)
    return f"{site}/{subject}/{NOISE.sub('', stem)}"


def find_gcloud(explicit: str | None = None) -> str:
    """Locate the gcloud binary, including a home-directory SDK install."""
    if explicit:
        return explicit
    found = shutil.which("gcloud")
    if found:
        return found
    bundled = Path.home() / "google-cloud-sdk" / "bin" / "gcloud"
    if bundled.exists():
        return str(bundled)
    raise RuntimeError(
        "gcloud not found. Install the Cloud SDK, or pass --gcloud /path/to/gcloud."
    )


def list_bucket(bucket: str, account: str | None, cache: Path, gcloud: str) -> list[str]:
    """Recursively list the bucket once, caching the manifest next to the data."""
    if cache.exists():
        logger.info("Using cached manifest %s", cache)
        return [line for line in cache.read_text().splitlines() if line.startswith("gs://")]
    command = [gcloud, "storage", "ls", "--recursive", bucket]
    if account:
        command.append(f"--account={account}")
    logger.info("Listing %s (this takes a minute)", bucket)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Bucket listing failed:\n{result.stderr.strip()}")
    lines = [line for line in result.stdout.splitlines() if line.startswith("gs://")]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(lines) + "\n")
    logger.info("Cached %d object paths to %s", len(lines), cache)
    return lines


def index_objects(paths: list[str], bucket: str) -> dict[str, dict[str, str]]:
    """Group object paths by session key and category."""
    prefix = bucket.rstrip("/") + "/"
    by_session: dict[str, dict[str, str]] = defaultdict(dict)
    wanted = {folder: name for name, (folder, _) in CATEGORIES.items()}
    for path in paths:
        if not path.startswith(prefix) or path.endswith("/"):
            continue
        parts = path[len(prefix):].split("/")
        if len(parts) < 4:
            continue
        site, subject, folder, filename = parts[0], parts[1], parts[2], parts[-1]
        category = wanted.get(folder)
        if category is None:
            continue
        if not filename.endswith(CATEGORIES[category][1]):
            continue
        by_session[session_key(site, subject, filename)][category] = path
    return by_session


def target_for(
    layout: str, dest: Path, key: str, category: str, source: str
) -> Path:
    """Where a fetched object lands, under the chosen directory layout.

    "flat"    <dest>/<category>/<filename>          (one bucket of each type)
    "subject" <dest>/<SITE>/<SUBJECT>/<category>/<filename>
    """
    name = Path(source).name
    if layout == "flat":
        return dest / category / name
    site, subject, _ = key.split("/", 2)
    return dest / site / subject / category / name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--dest", default="data/cohort", help="destination directory")
    parser.add_argument("--account", default=None, help="gcloud account to authenticate as")
    parser.add_argument(
        "--layout", default="flat", choices=("flat", "subject"),
        help="flat: <dest>/<category>/; subject: <dest>/<SITE>/<SUBJECT>/<category>/",
    )
    parser.add_argument(
        "--skip", default="", help="comma-separated categories to skip (audio,chirp,human)"
    )
    parser.add_argument("--limit", type=int, default=None, help="only fetch the first N sessions")
    parser.add_argument("--dry-run", action="store_true", help="report what would be fetched")
    parser.add_argument("--gcloud", default=None, help="path to the gcloud binary (default: PATH, then ~/google-cloud-sdk)")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    dest = Path(args.dest)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    categories = [c for c in CATEGORIES if c not in skip]

    gcloud = find_gcloud(args.gcloud)
    logger.info("Using %s", gcloud)
    paths = list_bucket(args.bucket, args.account, dest / "manifest.txt", gcloud)
    by_session = index_objects(paths, args.bucket)
    complete = sorted(k for k, v in by_session.items() if len(v) == len(CATEGORIES))
    logger.info(
        "%d sessions indexed, %d have audio + chirp + human", len(by_session), len(complete)
    )
    if args.limit:
        complete = complete[: args.limit]
        logger.info("Limited to %d session(s)", len(complete))

    planned = [
        (
            category,
            by_session[key][category],
            target_for(args.layout, dest, key, category, by_session[key][category]),
        )
        for key in complete
        for category in categories
    ]
    todo = [item for item in planned if not item[2].exists()]
    logger.info(
        "%d file(s) planned, %d already present, %d to download",
        len(planned), len(planned) - len(todo), len(todo),
    )
    if args.dry_run:
        for category, source, target in todo[:20]:
            logger.info("would fetch [%s] -> %s", category, target)
        return 0

    failures = 0
    for index, (category, source, target) in enumerate(todo, start=1):
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [gcloud, "storage", "cp", source, str(target)]
        if args.account:
            command.append(f"--account={args.account}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            failures += 1
            logger.error("Failed [%s] %s: %s", category, source, result.stderr.strip()[:200])
        if index % 25 == 0 or index == len(todo):
            logger.info("  %d/%d", index, len(todo))

    logger.info("Done: %d fetched, %d failed", len(todo) - failures, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
