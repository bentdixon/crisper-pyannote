# What the de-identification numbers actually mean

Written to be pasted into slides. Every term we use in the charts, said in
words that need no glossary, with the one-line version you can put on a slide.

---

## The setup, in four sentences

People at TranscribeMe typed up these interviews by hand. Whenever someone said
something that could identify a person -- a name, a specific date, a street, a
town -- the typist wrapped it in curly braces, like `{isaiah}`. That is our
answer key: it tells us where the identifying information is in each interview.
Our job is to see whether an automatic system blanks out those same words.

**Slide line:** *The human typists marked every name, date and place. We check
whether the software finds the same ones.*

---

## The words we use, and what they mean

### "Marked item" (charts and papers say **gold span**)

One thing a human typist flagged as identifying someone. A first name is one
marked item. "San Francisco" is one marked item, not two.

There are **876** of these across the 269 interviews, and **180** in the 24
interviews used for the head-to-head test.

**Slide line:** *876 pieces of identifying information, marked by hand.*

### "Blanked out" (charts say **redacted**, or **placeholder**)

The software replaced the words with a label saying what kind of thing it was:
`Zoe's gonna hop on` becomes `[PERSON_NAME]'s gonna hop on`. The sentence still
reads; the name is gone.

### "Found" (papers say **recall**, or **sensitivity**)

Of everything the typist marked, how much did the software blank out?

Found 90% means: of 100 names and dates the typist flagged, the software
blanked out 90 and walked past 10.

**Slide line:** *Found = how much of the identifying information the software
catches.*

### "Correct" (papers say **precision**)

Of everything the software blanked out, how much was actually something the
typist had marked?

Correct 65% means: the software blanked out 100 things, and 65 of them matched
something a human had flagged. The other 35 were its own idea.

**Important caveat:** those 35 are not necessarily mistakes. Only about half our
interviews carry any marking at all, and typists miss things. When the software
blanked out a hospital name that nobody marked, it counts against this number
even though blanking it was the right call. **So "correct" is a floor, not a
verdict** -- the real figure is better than the chart shows.

**Slide line:** *Correct = how much of what the software blanked out really
needed blanking. It is an underestimate.*

### "Combined score" (papers say **F1**)

One number that balances the two above, so systems can be ranked without
arguing about which matters more. It is high only when both are high: a system
that blanks out every word in the transcript would score perfectly on "found"
and terribly here.

**Slide line:** *Combined score = a single number balancing what it catches
against what it over-blanks.*

### "Still readable" (papers say **leaked**)

The strictest test, and the one a privacy reviewer actually asks about: after
the software has run, is the person's real name still sitting there in the
transcript, spelled out, where anyone can read it?

**Slide line:** *Still readable = a real name you could point to in the
finished transcript.*

---

## Two numbers that sound the same and are not

This trips everyone up, so it is worth a slide of its own.

**"Missed 16"** means: in 16 places, the typist marked something and the
software put no blank there.

**"Still readable 2"** means: in 2 places, a real name is sitting in the
finished transcript where you could read it.

Both are out of the same 180 marked items. They differ because of what happened
in the other 14 places: **the identifying word never made it into our transcript
at all.** The typist heard a name; our speech recognition heard a different word
and wrote that instead. There was nothing to blank out, and there is nothing
readable now.

A real example from these interviews: the typist typed "gonna hop on", our
system heard "gonna help out". When the same thing happens one word earlier, a
name simply is not in our transcript.

The reverse also happens, and it is worse. A system can blank out a name in one
place and leave the same name readable three sentences later. The transcript
looks cleaned up and is not. Our chunk-by-chunk method does this 6 times; the
turn-by-turn method does it zero times.

**Slide line:** *Missing a blank is not the same as exposing a name. Count both.*

---

## Why there is no "accuracy" number

Every other classification task reports accuracy, and someone will ask why we
do not. The answer:

Accuracy needs a count of things correctly left alone. Our answer key only
records where the identifying information **is** -- it never records where it
is not. So "correctly left alone" would be every ordinary word in the
transcript: every "um", every "I think so". There are about a million of them.

Any accuracy figure would therefore be about 99.9% for every system, including
one that blanks out nothing at all. It would be a true number that tells you
nothing.

**Slide line:** *We do not report accuracy: it would be 99.9% for every system,
including one that does nothing.*

---

## Two things about the source material

### Only the first part of each interview was typed up

The human transcripts stop early on about a third of the interviews -- often at
roughly the 30-minute mark, whatever the interview's length. Everything is
therefore measured only over the stretch each typist actually covered. Without
that restriction, every system gets blamed for the hour nobody typed up.

### Half the marked items cannot be checked for readability

Five of the ten sites scrubbed the identifying words out as they typed, writing
`{redacted}` in place of the name. That marks the position but destroys the
word, so there is nothing left to search the transcript for. Those items still
count for "found" and "correct", but **only 306 of the 876 can be checked for
whether a name is still readable.**

**Slide line:** *876 marked items; 306 of them can be checked for whether the
name is still readable.*

---

## The two automatic methods we compared

Both use the same model (Gemma 4, 31 billion parameters) on the same
transcripts. They differ in what we ask it to do.

**Chunk by chunk.** Hand the model a few pages of transcript, ask it to list
the identifying words it sees, then blank out those words ourselves. Fast --
about 25 minutes for all 269 interviews.

**Turn by turn.** Hand the model one person's turn of speech and ask it to type
the whole turn back out with the names already replaced by labels. Slower --
about 12 GPU-hours for all 269 -- because it processes every turn separately
and rewrites all the words, not just the names.

We never trust the rewrite: we compare it word by word against the original and
only accept the specific places where a name was swapped for a label. If the
model paraphrases anything, that turn is thrown out and counted.

**Slide line:** *Same model, two ways of asking. One lists names; the other
retypes the sentence with the names removed.*

---

## What the systems are called in the charts

| Chart label | What it is |
|---|---|
| Chirp-3 | Google's speech recognition, with the de-identification it does by itself |
| CrisperWhisper 2.0 + pyannote community-1 | Our transcription: the words, and who said them |
| + Gemma 4 31B redaction | Our transcription, then chunk-by-chunk blanking |
| + (turn rewrite) | Our transcription, then turn-by-turn blanking |
| + (possessive rule) | The same, after we fixed a bug where "Zoe's" could never be blanked |

---

## One sentence per chart

- **Finding identifying information** -- how much each system catches, how much
  of what it blanks really needed blanking, and the two combined.
- **Where each system's decisions land** -- the raw counts: caught, walked
  past, and blanked out beyond what was marked.
- **Names left readable** -- the privacy number: how many real identifiers you
  could still read in the finished transcript, and what kind of thing each one
  was.
