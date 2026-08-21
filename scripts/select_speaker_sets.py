"""Pick a pilot and a held-out set for the speaker-correction experiment.

Deliberately blind to the outcome: the visits are shuffled with a fixed seed
and sliced, so neither set can be chosen after seeing which transcripts the
corrector happens to help. The lists are committed, which is what makes the
validation result a held-out result rather than a second look at the pilot.

`select_validation.py` is not reusable here -- it requires PII spans whose
surface wording the typist kept, which exist on only five of the ten sites, so
its 24 visits are a biased sample for any question about speakers.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

SEED = 20260821
PILOT = 8
VALIDATION = 24


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--pattern", default="transcript.json")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    root = Path(args.outputs)
    visits = sorted(
        str(p.parent.relative_to(root)) for p in root.rglob(args.pattern)
        if not p.stem.endswith(("_speakers", "_speakers_rule"))
    )
    shuffled = list(visits)
    random.Random(SEED).shuffle(shuffled)
    pilot, validation = shuffled[:PILOT], shuffled[PILOT:PILOT + VALIDATION]
    assert not set(pilot) & set(validation)

    out = Path(args.output_dir)
    for name, rows in (("speaker_pilot.txt", pilot), ("speaker_validation.txt", validation)):
        (out / name).write_text("\n".join(sorted(rows)) + "\n")
        print(f"{name}: {len(rows)} visits")
    print("\npilot")
    for row in sorted(pilot):
        print(f"    {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
