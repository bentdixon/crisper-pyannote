"""Checks for the speaker-rewrite plumbing and the deterministic rule.

Run with `uv run python tests/test_speaker_rewrite.py` -- there is no pytest in
this environment, same as tests/test_merge.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import speaker_rewrite as rewrite  # noqa: E402
from correct_speakers_rule import propose  # noqa: E402


def words(spec: list[tuple[str, int]]) -> list[dict]:
    """Word list from (speaker, count) pairs, one second per word."""
    out: list[dict] = []
    clock = 0.0
    for speaker, count in spec:
        for _ in range(count):
            out.append({"word": "x", "start": clock, "end": clock + 1, "speaker": speaker})
            clock += 1
    return out


def test_runs_are_maximal():
    assert rewrite.speaker_runs(words([("A", 2), ("B", 1), ("A", 3)])) == [(0, 2), (2, 3), (3, 6)]


def test_single_word_transcript_is_one_run():
    assert rewrite.speaker_runs(words([("A", 1)])) == [(0, 1)]


def test_empty_word_list_has_no_runs():
    assert rewrite.speaker_runs([]) == []


def test_interjection_moves_to_the_surrounding_speaker():
    changes, reasons = propose(words([("A", 10), ("B", 1), ("A", 10)]))
    assert changes == {10: "A"}
    assert reasons["moved"] == 1


def test_run_longer_than_the_cap_is_left_alone():
    changes, _ = propose(words([("A", 10), ("B", 4), ("A", 10)]))
    assert changes == {}


def test_short_run_takes_its_longer_neighbour():
    # Both flanking runs must themselves be longer than the cap, or they are
    # short runs too and get moved in the same pass.
    changes, _ = propose(words([("A", 4), ("B", 2), ("C", 10)]))
    assert changes == {4: "C", 5: "C"}


def test_a_tie_between_neighbours_goes_to_the_preceding_run():
    changes, _ = propose(words([("A", 8), ("B", 1), ("C", 8)]))
    assert changes == {8: "A"}


def test_run_at_the_start_takes_its_only_neighbour():
    changes, _ = propose(words([("B", 1), ("A", 10)]))
    assert changes == {0: "A"}


def test_transcript_with_one_run_has_no_neighbour_to_take():
    changes, reasons = propose(words([("A", 2)]))
    assert changes == {}
    assert reasons["kept: no neighbour"] == 1


def test_neighbours_are_read_before_any_change_is_applied():
    # Two short runs of B either side of a short run of A. If changes were
    # applied as we went, moving the first B to A would lengthen the A run and
    # change what the second B sees. Reading the original layout keeps the two
    # decisions independent.
    changes, _ = propose(words([("Z", 10), ("B", 1), ("A", 1), ("B", 1), ("Z", 10)]))
    assert changes[10] == "Z"
    assert changes[12] == "Z"


def test_provenance_is_recorded_only_on_words_that_moved():
    source = words([("A", 2)])
    out = rewrite.apply_changes(source, {0: "B", 1: "A"}, "rule")
    assert out[0]["speaker"] == "B" and out[0]["speaker_from"] == "A"
    assert out[0]["speaker_by"] == "rule"
    assert "speaker_from" not in out[1]


def test_apply_changes_does_not_mutate_its_input():
    source = words([("A", 2)])
    rewrite.apply_changes(source, {0: "B"}, "rule")
    assert source[0]["speaker"] == "A"


def test_intact_check_accepts_a_pure_relabel():
    source = words([("A", 2)])
    rewrite.assert_words_intact(source, rewrite.apply_changes(source, {0: "B"}, "rule"))


def test_intact_check_rejects_a_changed_word():
    source = words([("A", 2)])
    broken = [dict(w) for w in source]
    broken[1]["word"] = "y"
    try:
        rewrite.assert_words_intact(source, broken)
    except ValueError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("a changed word must be rejected")


def test_intact_check_rejects_a_retimed_word():
    source = words([("A", 2)])
    broken = [dict(w) for w in source]
    broken[0]["end"] = 99.0
    try:
        rewrite.assert_words_intact(source, broken)
    except ValueError as error:
        assert "retimed" in str(error)
    else:
        raise AssertionError("a retimed word must be rejected")


def test_intact_check_rejects_a_dropped_word():
    source = words([("A", 2)])
    try:
        rewrite.assert_words_intact(source, source[:1])
    except ValueError as error:
        assert "word count" in str(error)
    else:
        raise AssertionError("a dropped word must be rejected")


def test_destination_sits_beside_its_source():
    assert rewrite.destination_for(Path("a/b/transcript.json"), "_speakers.json") == Path(
        "a/b/transcript_speakers.json"
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
