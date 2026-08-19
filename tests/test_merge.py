"""Checks for the word-to-speaker assignment in crisper_pipeline.merge.

Plain asserts, no test framework in this environment:

    uv run python tests/test_merge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crisper_pipeline import merge  # noqa: E402

SEGMENTS = [
    {"start": 0.0, "end": 5.0, "speaker": "A"},
    {"start": 6.0, "end": 10.0, "speaker": "B"},
]


def speakers(words, fill_nearest=False):
    return [
        w["speaker"]
        for w in merge.assign_speakers(SEGMENTS, [dict(w) for w in words], fill_nearest)
    ]


def test_zero_duration_word_inside_a_segment():
    assert speakers([{"start": 2.0, "end": 2.0}]) == ["A"]
    assert speakers([{"start": 7.5, "end": 7.5}]) == ["B"]


def test_zero_duration_word_on_a_boundary():
    # An instant on a shared edge belongs to the segment that starts first.
    assert speakers([{"start": 5.0, "end": 5.0}]) == ["A"]
    assert speakers([{"start": 6.0, "end": 6.0}]) == ["B"]


def test_zero_duration_word_in_a_gap():
    # A genuine gap is a genuine gap, whatever the word's duration: the flag
    # decides, exactly as it does for a word with real extent.
    assert speakers([{"start": 5.5, "end": 5.5}]) == ["UNKNOWN"]
    # fill_nearest compares midpoints, not edges, so both instants in this
    # one-second gap go to the longer-reaching B (midpoint 8.0) rather than
    # to whichever segment they sit closer to.
    assert speakers([{"start": 5.4, "end": 5.4}], fill_nearest=True) == ["B"]
    assert speakers([{"start": 5.9, "end": 5.9}], fill_nearest=True) == ["B"]


def test_zero_duration_word_outside_the_timeline():
    assert speakers([{"start": -1.0, "end": -1.0}]) == ["UNKNOWN"]
    assert speakers([{"start": 12.0, "end": 12.0}]) == ["UNKNOWN"]
    assert speakers([{"start": -1.0, "end": -1.0}], fill_nearest=True) == ["A"]
    assert speakers([{"start": 12.0, "end": 12.0}], fill_nearest=True) == ["B"]


def test_ordinary_words_are_unchanged():
    words = [
        {"start": 0.5, "end": 1.0},
        {"start": 4.5, "end": 5.5},   # straddles the gap, more of it in A
        {"start": 6.5, "end": 7.0},
        {"start": 5.2, "end": 5.8},   # wholly inside the gap
        {"start": 9.5, "end": 11.0},
    ]
    assert speakers(words) == ["A", "A", "B", "UNKNOWN", "B"]
    assert speakers(words, fill_nearest=True) == ["A", "A", "B", "B", "B"]


def test_no_diarization_at_all():
    assert merge.assign_speakers([], [{"start": 1.0, "end": 1.0}])[0]["speaker"] == (
        "UNKNOWN"
    )


def main() -> int:
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for check in checks:
        check()
        print(f"ok  {check.__name__}")
    print(f"{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
