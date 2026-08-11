"""Restrict scoring to the span of audio the human transcript actually covers.

The human transcripts stop early on a third of this corpus -- 97 of 269 visits
are under 80% covered, 34 under 50%, and the worst stop at a hard 32-minute
cutoff regardless of session length (one covers 32.0 of 122.8 minutes). Scoring
a full-session hypothesis against a partial reference charges every untranscribed
minute as insertions: two systems sharing no code both insert the same ~8400
words on that visit, which is not hallucination, it is the reference ending.

Everything here exists so both sides describe the same stretch of audio.

The window ends at the *start* of the last turn, not its end. Turn ends are
synthesized from the following turn's start (see prepare_data), so the final
turn has no real end -- load_timestamped_text stretches it to the full audio
duration, which is exactly the untranscribed remainder we are trying to
exclude. Dropping that one turn costs a few words out of hundreds of turns and
removes the guesswork.
"""

from __future__ import annotations


def covered_turns(turns: list[dict], max_seconds: float | None = None,
                  ) -> tuple[list[dict], float | None, float | None]:
    """Turns whose time span is real, plus the window they cover.

    Returns ([], None, None) when there is nothing scoreable, which the caller
    must treat as "skip this visit", not as "no errors".
    """
    if len(turns) < 2:
        return [], None, None
    start = float(turns[0]["start"])
    end = float(turns[-1]["start"])
    if end <= start:
        return [], None, None

    if max_seconds is not None and end - start > max_seconds:
        # A hard cap on top of the per-transcript window, for checking that the
        # comparison does not rest on the tail of the longer transcripts. Turns
        # beyond the cap are dropped whole rather than split, so the reference
        # stays a sequence of complete turns.
        end = start + max_seconds
        turns = [t for t in turns if float(t["start"]) < end]
        if len(turns) < 2:
            return [], None, None

    kept = [dict(turn) for turn in turns[:-1]]
    # The last kept turn ends where the dropped one begins; it already should,
    # but stating it keeps the window and the reference exactly consistent.
    kept[-1]["end"] = min(float(turns[-1]["start"]), end)
    return kept, start, end


def clip_words(words: list[dict], start: float | None, end: float | None) -> list[dict]:
    """Words that begin inside the window, with their ends clamped to it.

    Clamping matters for DER: a word straddling the boundary would otherwise
    contribute speech time the reference cannot describe.
    """
    if start is None or end is None:
        return words
    clipped = []
    for word in words:
        when = word.get("start")
        if when is None:
            continue
        when = float(when)
        if when < start or when >= end:
            continue
        copy = dict(word)
        finish = word.get("end")
        copy["end"] = min(float(finish), end) if finish is not None else end
        clipped.append(copy)
    return clipped


def coverage_fraction(turns: list[dict], duration: float | None) -> float | None:
    """How much of the recording the transcript reaches, for reporting."""
    if not turns or not duration:
        return None
    return min(float(turns[-1]["start"]) / float(duration), 1.0)
