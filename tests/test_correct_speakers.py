"""Checks for the Gemma speaker corrector's protocol, without a model.

Everything that can be wrong in a way that still produces plausible output --
addressing, quote validation, the rejection counters, how a correction lands on
the word list -- is here. The model itself is stubbed: `correct_words` takes the
generation step as a callable precisely so this file can run on a laptop.

Run with `uv run python tests/test_correct_speakers.py` -- there is no pytest in
this environment, same as tests/test_merge.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import correct_speakers_llm as corrector  # noqa: E402
from speaker_rewrite import speaker_runs  # noqa: E402


def words_from(spec: list[tuple[str, str]]) -> list[dict]:
    """Word list from (speaker, text) pairs, one second per word, no gaps."""
    out: list[dict] = []
    clock = 0.0
    for speaker, text in spec:
        for token in text.split():
            out.append({"word": token, "start": clock, "end": clock + 1, "speaker": speaker})
            clock += 1
    return out


# One interviewer question with the participant's "no" absorbed at the end,
# then a participant turn. The shape the corrector exists for.
ABSORBED = [
    ("SPEAKER_00", "and have you had trouble sleeping no"),
    ("SPEAKER_01", "it has been alright I think"),
]


def reply(corrections: list[dict]) -> str:
    return json.dumps({"corrections": corrections})


def run(words: list[dict], corrections: list[dict], budget: int = 10_000):
    """Drive correct_words with a stubbed model returning `corrections`."""
    return corrector.correct_words(
        words, lambda numbered: reply(corrections), lambda text: len(text.split()), budget,
    )


def test_a_located_quote_is_applied():
    words = words_from(ABSORBED)
    corrected, report = run(words, [
        {"turn": 0, "text": "no", "speaker": "SPEAKER_01", "action": "reassign"},
    ])
    assert corrected[6]["speaker"] == "SPEAKER_01"
    assert corrected[6]["speaker_from"] == "SPEAKER_00"
    assert corrected[6]["speaker_by"] == "gemma"
    assert report["corrections_applied"] == 1
    assert report["words_moved"] == 1


def test_an_invented_quote_is_dropped_and_counted():
    words = words_from(ABSORBED)
    corrected, report = run(words, [
        {"turn": 0, "text": "absolutely not", "speaker": "SPEAKER_01", "action": "reassign"},
    ])
    assert report["unmatched"] == 1
    assert report["corrections_applied"] == 0
    assert [w["speaker"] for w in corrected] == [w["speaker"] for w in words]


def test_a_quote_from_the_wrong_turn_is_dropped_and_counted():
    # "alright" is in the transcript, but not in turn 0. A correction is only
    # valid against the turn it names.
    words = words_from(ABSORBED)
    _, report = run(words, [
        {"turn": 0, "text": "alright", "speaker": "SPEAKER_01", "action": "reassign"},
    ])
    assert report["unmatched"] == 1


def test_an_out_of_range_turn_number_is_dropped_and_counted():
    words = words_from(ABSORBED)
    _, report = run(words, [
        {"turn": 9, "text": "no", "speaker": "SPEAKER_01", "action": "reassign"},
    ])
    assert report["unmatched"] == 1
    assert report["corrections_applied"] == 0


def test_a_speaker_never_shown_is_rejected():
    words = words_from(ABSORBED)
    _, report = run(words, [
        {"turn": 0, "text": "no", "speaker": "SPEAKER_07", "action": "reassign"},
    ])
    assert report["rejected_speaker"] == 1


def test_naming_the_turns_own_speaker_is_rejected():
    words = words_from(ABSORBED)
    _, report = run(words, [
        {"turn": 0, "text": "no", "speaker": "SPEAKER_00", "action": "reassign"},
    ])
    assert report["rejected_speaker"] == 1


def test_a_correction_moving_seven_words_is_rejected():
    words = words_from([
        ("SPEAKER_00", "and have you had any trouble sleeping at all recently"),
        ("SPEAKER_01", "not really no"),
    ])
    _, report = run(words, [
        {"turn": 0, "text": "have you had any trouble sleeping at",
         "speaker": "SPEAKER_01", "action": "reassign"},
    ])
    assert report["rejected_long"] == 1
    assert report["corrections_applied"] == 0


def test_a_correction_moving_six_words_is_allowed():
    words = words_from([
        ("SPEAKER_00", "and have you had any trouble sleeping at all recently"),
        ("SPEAKER_01", "not really no"),
    ])
    _, report = run(words, [
        {"turn": 0, "text": "have you had any trouble sleeping",
         "speaker": "SPEAKER_01", "action": "reassign"},
    ])
    assert report["rejected_long"] == 0
    assert report["corrections_applied"] == 1


def test_two_corrections_claiming_one_word_conflict():
    words = words_from([
        ("SPEAKER_00", "and have you had trouble sleeping no"),
        ("SPEAKER_01", "it has been alright"),
        ("SPEAKER_02", "mm hmm"),
    ])
    _, report = run(words, [
        {"turn": 0, "text": "no", "speaker": "SPEAKER_01", "action": "reassign"},
        {"turn": 0, "text": "no", "speaker": "SPEAKER_02", "action": "reassign"},
    ])
    assert report["corrections_applied"] == 1
    assert report["rejected_conflict"] == 1


def test_the_same_correction_twice_is_a_duplicate_not_a_conflict():
    words = words_from(ABSORBED)
    _, report = run(words, [
        {"turn": 0, "text": "no", "speaker": "SPEAKER_01", "action": "reassign"},
        {"turn": 0, "text": "no", "speaker": "SPEAKER_01", "action": "reassign"},
    ])
    assert report["corrections_applied"] == 1
    assert report["duplicate"] == 1
    assert report["rejected_conflict"] == 0


def test_a_split_turns_one_run_into_three():
    words = words_from([
        ("SPEAKER_00", "have you had trouble sleeping no okay and your appetite"),
        ("SPEAKER_01", "that has been fine"),
    ])
    assert len(speaker_runs(words)) == 2
    corrected, report = run(words, [
        {"turn": 0, "text": "no", "speaker": "SPEAKER_01", "action": "split"},
    ])
    # The turn structure is recomputed from the labels, so one differing label
    # inside a run splits it into three without any turn surgery here.
    assert len(speaker_runs(corrected)) == 4
    assert report["applied_split"] == 1
    assert report["applied_reassign"] == 0


def test_an_unparseable_reply_is_a_failed_window_not_a_clean_one():
    words = words_from(ABSORBED)
    corrected, report = corrector.correct_words(
        words, lambda numbered: "I could not find anything wrong.",
        lambda text: len(text.split()),
    )
    assert report["chunk_failures"] == 1
    assert report["corrections_applied"] == 0
    assert [w["speaker"] for w in corrected] == [w["speaker"] for w in words]


def test_a_raising_model_is_a_failed_window():
    def explode(numbered):
        raise RuntimeError("out of memory")

    _, report = corrector.correct_words(
        words_from(ABSORBED), explode, lambda text: len(text.split()),
    )
    assert report["chunk_failures"] == 1


def test_a_malformed_entry_is_counted_not_swallowed():
    words = words_from(ABSORBED)
    _, report = run(words, [
        {"turn": 0, "text": "no", "speaker": "SPEAKER_01", "action": "rewrite"},
    ])
    assert report["malformed"] == 1
    assert report["corrections_proposed"] == 1
    assert report["corrections_applied"] == 0


def test_an_empty_corrections_list_is_a_valid_answer():
    words = words_from(ABSORBED)
    corrected, report = run(words, [])
    assert report["chunk_failures"] == 0
    assert report["corrections_proposed"] == 0
    assert [w["speaker"] for w in corrected] == [w["speaker"] for w in words]


def test_a_fenced_reply_is_still_read():
    words = words_from(ABSORBED)
    body = reply([{"turn": 0, "text": "no", "speaker": "SPEAKER_01", "action": "reassign"}])
    _, report = corrector.correct_words(
        words, lambda numbered: f"```json\n{body}\n```", lambda text: len(text.split()),
    )
    assert report["corrections_applied"] == 1


def test_turn_numbering_is_request_local():
    words = words_from([
        ("SPEAKER_00", "one one one"),
        ("SPEAKER_01", "two two two"),
        ("SPEAKER_00", "three three three"),
        ("SPEAKER_01", "four four four"),
    ])
    runs = speaker_runs(words)
    rendered = corrector.render_window(words, runs, [2, 3])
    numbers = [line.split()[0] for line in rendered.splitlines()]
    # The window starts at absolute turn 2 and is still numbered from zero, so
    # the model never reasons about an offset into a transcript it cannot see.
    assert numbers == ["0", "1"]
    assert "three" in rendered.splitlines()[0]


def test_a_correction_addresses_the_window_not_the_transcript():
    words = words_from([
        ("SPEAKER_00", "alpha alpha alpha"),
        ("SPEAKER_01", "beta beta beta"),
        ("SPEAKER_00", "gamma gamma no"),
        ("SPEAKER_01", "delta delta delta"),
    ])
    runs = speaker_runs(words)
    changes, counters = corrector.interpret_window(
        words, runs, [2, 3],
        [{"turn": 0, "text": "no", "speaker": "SPEAKER_01", "action": "reassign"}],
        {},
    )
    # Turn 0 of this window is absolute turn 2, whose last word is index 8.
    assert changes == {8: "SPEAKER_01"}
    assert counters["applied"] == 1


def test_a_short_transcript_is_one_window():
    words = words_from(ABSORBED)
    runs = speaker_runs(words)
    windows = corrector.build_windows(words, runs, lambda text: len(text.split()), budget=1000)
    assert windows == [[0, 1]]


def test_a_long_transcript_windows_with_overlap():
    words = words_from([("SPEAKER_00" if i % 2 else "SPEAKER_01", f"w{i} w{i} w{i}")
                        for i in range(40)])
    runs = speaker_runs(words)
    windows = corrector.build_windows(
        words, runs, lambda text: len(text.split()), budget=60, overlap=2,
    )
    assert len(windows) > 1
    # Every window after the first re-seeds the previous window's tail, so a
    # turn at a boundary is judged at least once with its left context.
    assert windows[1][:2] == windows[0][-2:]
    assert set().union(*windows) == set(range(len(runs)))


def test_the_pause_before_a_turn_is_rendered():
    words = [
        {"word": "yes", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"word": "no", "start": 3.5, "end": 4.0, "speaker": "SPEAKER_01"},
    ]
    runs = speaker_runs(words)
    lines = corrector.render_window(words, runs, [0, 1]).splitlines()
    # The first turn has no predecessor and so no pause to report.
    assert "----" in lines[0]
    assert "+2.5s" in lines[1]


def test_the_prompt_and_examples_render():
    # The prompt carries a literal JSON object and is .format()ted, so an
    # undoubled brace would raise here rather than at the first GPU call.
    messages = corrector.build_messages("0  SPEAKER_00  [00:00:00.0  ----]  hello")
    assert messages[0]["role"] == "user"
    assert messages[-1]["role"] == "user"
    assert "hello" in messages[-1]["content"]
    for _, wanted in corrector.EXAMPLES:
        assert corrector.parse_corrections(wanted) is not None


def test_the_worked_examples_would_survive_their_own_validation():
    # An example the corrector would itself reject teaches the model to produce
    # rejected output. Each example is replayed against its own transcript.
    for given, wanted in corrector.EXAMPLES:
        entries, malformed = corrector.parse_corrections(wanted)
        assert malformed == 0
        assert entries
        speakers = {line.split()[1] for line in given.splitlines()}
        for entry in entries:
            assert 0 <= entry["turn"] < len(given.splitlines())
            assert entry["speaker"] in speakers
            assert len(entry["text"].split()) <= corrector.MAX_MOVE_WORDS


def main() -> int:
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for check in checks:
        check()
        print(f"ok  {check.__name__}")
    print(f"{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
