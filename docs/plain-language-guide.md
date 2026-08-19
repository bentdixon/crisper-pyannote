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

There are **876** of these across the 269 interviews. Every number in this deck
is measured over all 269.

**Slide line:** *876 pieces of identifying information, marked by hand.*

### "Blanked out" (charts say **redacted**, or **placeholder**)

The software replaced the words with a label saying what kind of thing it was:
`Zoe's gonna hop on` becomes `[PERSON_NAME]'s gonna hop on`. The sentence still
reads; the name is gone.

### "Found" (papers say **recall**, or **sensitivity**)

Of everything the typist marked, how much did the software blank out?

Found 81% means: of 100 names and dates the typist flagged, the software
blanked out 81 and walked past 19. That is our best method; Google's Chirp-3
finds 65%.

**Slide line:** *Found = how much of the identifying information the software
catches.*

### "Correct" (papers say **precision**)

Of everything the software blanked out, how much was actually something the
typist had marked?

Correct 44% means: the software blanked out 100 things, and 44 of them matched
something a human had flagged. The other 56 were its own idea. Chirp-3 scores
28% here -- it blanks out far more, and far more loosely.

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

**"Missed 170"** means: in 170 places, the typist marked something and the
software put no blank there.

**"Still readable 17"** means: in 17 places, a real name is sitting in the
finished transcript where you could read it.

These come from different denominators -- 876 marked items for the first, and
the 306 of them whose original wording survives for the second. They differ because of what happened
in the other 14 places: **the identifying word never made it into our transcript
at all.** The typist heard a name; our speech recognition heard a different word
and wrote that instead. There was nothing to blank out, and there is nothing
readable now.

A real example from these interviews: the typist typed "gonna hop on", our
system heard "gonna help out". When the same thing happens one word earlier, a
name simply is not in our transcript.

The reverse also happens, and it is worse. A system can blank out a name in one
place and leave the same name readable three sentences later. The transcript
looks cleaned up and is not. Across all 269 interviews the chunk-by-chunk method does this far more often
than the turn-by-turn one, which is most of why its readable count is twice as
high (31 against 17) while the two find almost the same amount.

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
the identifying words it sees, then blank out those words ourselves. Fast -- about 90 minutes
for all 269 interviews on one GPU.

**Turn by turn.** Hand the model one person's turn of speech and ask it to type
the whole turn back out with the names already replaced by labels. Slower -- about two hours
across three GPUs -- because it processes every turn separately
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

---

## The headline numbers, all 269 interviews

| | finds | correct | mentions still readable | people still identifiable |
|---|---:|---:|---:|---:|
| **our turn-by-turn method** | **81%** | 44% | **15 of 306** | **14 of 122** |
| our chunk-by-chunk method | 78% | 45% | 18 of 306 | 16 of 122 |
| Google Chirp-3, on its own | 65% | 28% | 31 of 306 | 21 of 122 |
| Chirp-3 put through verbatimize | 9% | 27% | 144 of 306 | 72 of 122 |

The last two columns answer different questions and both are worth showing. A
first name said seventeen times in one interview is seventeen mentions but one
person. If a system blanks sixteen of them, that is sixteen mentions handled and
one missed -- and one person who is still identifiable. Counting per mention
says how thorough a system is; counting per person says whether the interview
could be released.

**Slide line:** *Our best method finds a quarter more of the identifying
information than Google's, blanks out less of everything else, and leaves a
third as many names readable.*

The last row is a warning, not a candidate. Feeding Chirp-3's already-cleaned
text through the disfluency step **puts the names back**: it hears the audio and
types what was said where Chirp had written a label. It undoes de-identification
and should not be released.

---

# The transcription accuracy figures

Same idea, different question: not "did it hide the names" but "did it get the
words right, and did anything go missing from the conversation".

## "Word error rate" (charts say **WER**)

Line the machine's transcript up against the typist's, word by word, and count
the words it got wrong, missed, or added that were never said. Divide by how
many words the typist wrote. Lower is better. 12% means about one word in eight
is wrong somewhere.

Two things to know before quoting it. The typists wrote down what people meant,
tidying as they went, while the machines write down every "um" and false start,
so a machine can be charged for words that really were spoken. And the typists
stopped early on a third of these interviews, so we only score the part they
covered -- otherwise the untranscribed hour counts as the machine making things
up.

**Slide line:** *Word error rate is the share of words that came out wrong. We
only compare against the part a person actually typed up.*

## "Average" versus "the typical interview"

These disagree here, and the disagreement is the point. Averaged over all 269
interviews our pipeline looks better than Google's Chirp-3, 14.3% against 17.5%.
But interview by interview, **Chirp-3 is better on 151 of the 269 and ours on
118**. The average is being decided by about ten interviews where Chirp-3 goes
badly wrong -- getting more than half the words wrong -- and ours does not.
Take those ten out and the gap shrinks from 3.2 points to 1.2.

Neither number is a trick. They answer different questions: *what should I
expect on a randomly chosen interview* (Chirp-3, slightly) and *how bad can it
get* (Chirp-3, much worse).

**Slide line:** *Chirp-3 is better on most interviews. Ours is better on
average, because Chirp-3 fails badly on about ten of them and we do not.*

## The curve chart (papers say **cumulative distribution**)

Read one point at a time: "of all 269 interviews, this share came in at or below
this error rate". A line that is higher at a given error rate is the better
system there. The lines cross at about 13%, which is exactly the finding above
-- one system leads on the ordinary interviews and the other leads once things
get hard.

**Slide line:** *One system wins the easy interviews, the other wins the hard
ones. A single average per system hides that completely.*

## "Lost turn"

A turn is one uninterrupted stretch of one person speaking. A turn is lost when
the system produced no word at all within half a second of the time the typist
says someone was talking -- the sentence is simply not in the transcript.

The half second is not a fudge. The typists marked turn times to about a third
of a second, so demanding a word land exactly inside the marked span counts an
approximately drawn boundary as a missing sentence. That mattered a lot: on a
strict test our pipeline appeared to lose 6.3% of turns, but 85% of those had a
word within one second of the span. They were not missing.

This is worth its own number because word error rate hardly notices it. A lost
four-word question is four missing words out of five thousand, a rounding error
in the percentage, but a reader looking at the transcript sees a question gone
from the conversation and an answer to nothing.

Across all 269 interviews, of 68,950 turns:

| | turns lost |
|---|---:|
| CrisperWhisper with pyannote 3.1 | 1.0% |
| CrisperWhisper with community-1 (ours) | 1.3% |
| Google Chirp-3 | 4.4% |
| Chirp-3 put through verbatimize | 7.4% |

The two kinds of loss are not alike. When Chirp-3 loses a turn, the nearest word
it did transcribe is a median of 44 seconds away -- it drops whole stretches of
the interview, not single sentences. When ours loses one, the nearest word is a
tenth of a second away. Chirp-3 also loses turns at the same rate whatever their
length, while ours lose short turns about twice as often as long ones.

**Slide line:** *Chirp-3 leaves three to four times as much of the conversation
out of the transcript as either of our pipelines, and it goes missing in long
stretches rather than odd sentences.*

### The four charts about turns

- **What became of each turn** -- the three things that can go wrong with a
  turn, on one scale. Losing it outright is the rarest. Getting the words right
  and the speaker wrong is by far the most common, between an eighth and a
  quarter of every turn in the corpus.
- **Turns that never made it into the transcript** -- the loss rate on its own,
  split by how long the turn ran.
- **When a turn is missing, how far away is the nearest word** -- the chart that
  says how to read the other two. A missing turn with a word a tenth of a second
  away is a boundary drawn approximately. A missing turn with nothing for
  forty-four seconds is a lost stretch of interview.
- **What became of each turn, by who was speaking** -- the same three failures
  for the interviewer and for the participant. Our pipeline puts the
  participant's words on the wrong speaker in 29% of their turns against 20% for
  the interviewer. Since the participant's answers are the data, that is the
  worse way round, and it is the one number here that argues for the other
  team's diarization.

## "Speaker error" (charts say **DER confusion**, or **sWER**)

Two different ways of asking whether the right words got attached to the right
person. Speaker confusion is measured on the clock -- what fraction of the
speaking time is credited to the wrong person. The speaker-aware word error rate
pools everything one person said into a block and scores that block.

The second one has a blind spot worth stating out loud: because it pools a whole
interview's words per speaker, a lost or misattributed turn disappears into a
five-thousand-word block. That is why the lost-turn count above exists as its
own measure rather than being read off a speaker error rate.

**Slide line:** *Speaker error says how much speech is credited to the wrong
person. It cannot tell you whether a particular exchange survived -- that needs
counting turns.*
