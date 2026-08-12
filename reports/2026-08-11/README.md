# Transcription system evaluation — 2026-08-11

Six systems scored against the human transcripts over all 269 AMPSCZ/PSYCHS
visits, plus PII redaction. Everything here is generated; see "Regenerating"
below.

**Scoring is restricted to the span each human transcript actually covers.**
The transcripts stop early on a third of this cohort -- 97 of 269 visits under
80% covered, 34 under 50%, and the worst stop at a hard 32-minute cutoff
regardless of session length. Scoring whole sessions against partial references
charged every system for minutes the reference never described and inflated WER
from roughly 12% to roughly 42%.

Interactive version: https://claude.ai/code/artifact/bf6f5c1c-6f6c-411d-9c81-c832db2d52f5

## Headline numbers

Per 100 reference words, all 269 visits (`tables/metrics-all-visits.csv`):

| System | WER | excl. insertions | sWER | DER confusion | QTP-F1 |
|---|---:|---:|---:|---:|---:|
| CrisperWhisper 2.0 + pyannote community-1 | **11.5** | **10.4** | **19.3** | 15.5 | **92.1** |
| CrisperWhisper 2.0 + pyannote 3.1 | 12.8 | 10.6 | 19.8 | **12.4** | 90.5 |
| Chirp-3 | 16.7 | 13.8 | 29.8 | 16.0 | 87.5 |
| Chirp-3 + CrisperWhisper 2.0 verbatimize | 16.9 | 14.1 | 29.9 | 15.8 | 88.1 |

Adding the Qwen2.5-7B review makes every metric worse for both pipelines, on
269 of 269 visits. It is not worth running.

These are ordinary ASR numbers. The decomposition in `wer-error-types.csv`
splits them further: for the community-1 pipeline, **4.0% of reference words
misheard, 6.5% missed, 1.0% added**. Deletions are now the largest term, and
insertions -- which used to carry half the total -- are about a point.

Two independent WER implementations agree on that ordering. The partner team's
compareFiles.py: 14.3% / 15.7% / 18.9% filler-normalized (`partner-wer.csv`).
The study team's jiwer script, vendored unmodified: 14.5% / 15.8% / 17.5%
pooled (`jiwer-wer.csv`). Three implementations, three alignment rules, one
ranking.

## Figures (PNG at 2x)

Two sets of the same fifteen charts:

- **`figures/`** — with the explanatory caption under each chart. Use these when
  the figure travels on its own and has to carry its own caveats.
- **`figures-clean/`** — title, direction, values, axes and colour key only, no
  prose. Use these in slides and papers where the surrounding text supplies the
  context.

The key stays in the clean set on purpose: several of these charts stack seven
colours, and without the legend they cannot be read at all.


| File | What it shows |
|---|---|
| `wer-error-types.png` | **Start here.** WER split into misheard / missed / spoken-but-unrecorded. Explains the headline. |
| `wer.png` | WER by system, with per-visit median and middle-50% range |
| `wer-excluding-insertions.png` | The fair transcription-quality comparison |
| `wer-composition.png` | WER as substitutions + deletions + insertions |
| `swer.png` | Speaker-attributed WER (capped at 1.0; see caveat below) |
| `der.png`, `der-confusion.png` | Diarization error; confusion alone is the clean attribution number |
| `qtp-f1.png` | Negation / modality / temporal cue preservation |
| `mono-vs-stereo-wer.png`, `mono-vs-stereo-der.png` | 203 mono files against 66 stereo-container files |
| `pii-redaction.png` | Over- and under-redaction against the human annotation |
| `partner-wer.png` | The partner team's own WER implementation, run unmodified |
| `jiwer-wer.png` | The study team's jiwer WER script, run unmodified |

Vector versions of every chart are inline in `eval_report.html`. Regenerate
either set with `scripts/export_charts.py`, adding `--clean` for the second.

## Tables (`tables/`, CSV)

| File | Contents |
|---|---|
| `metrics-all-visits.csv` | All metrics, six systems, 269 visits |
| `metrics-mono-files.csv` | Same, restricted to the 203 mono files |
| `metrics-stereo-container-files.csv` | Same, the 66 stereo-container files |
| `wer-error-types.csv` | WER by edit category; columns sum to WER |
| `pii-redaction.csv` | Span P/R/F1, over/under rates, leak rate |
| `partner-wer.csv` | Their three-tier WER and word ratio |
| `jiwer-wer.csv` | jiwer WER pooled/mean/median, S/D/I rates, and the filler-symmetric variant |

Rates are percentages of reference words (or of gold spans, for redaction).

## Raw results (`data/`, JSON)

`results.json`, `results_mono.json`, `results_stereo.json`, `taxonomy.json`,
`redaction.json`, `partner_wer.json`, `jiwer_wer.json`. Each carries a `per_visit` block, so any
figure here can be traced to the visits behind it. Keyed by full system name.

## Caveats that change how these read

- **Coverage-restricted scoring is what makes these numbers meaningful.** Any
  figure computed over whole sessions against these transcripts is measuring
  reference truncation, not transcription. The window runs from the first turn
  to the *start* of the last one: turn ends are synthesized from the next turn's
  start, so the final turn has no real end.
- **The machine is still verbatim and the reference semi-verbatim.** That
  remains a convention gap, now worth about a point rather than thirty.
- **sWER was corrected on 2026-08-11.** Streams are capped at 1.0 and reference
  streams under five words dropped. An earlier version reported a 32-point gap
  between the two diarizers; the real gap is 0.3 points.
- **Pooled WER was corrected on 2026-08-11.** The reference had been ordered by
  speaker while the hypothesis was chronological, inflating every WER by ~33
  points. Any figure dated before this directory is wrong.
- **Redaction precision is a lower bound.** Only 142 of 269 transcripts carry
  any PII annotation, so an identifier nobody marked counts against a system
  that caught it. Recall is trustworthy.
- **No file used stereo channels.** All 66 stereo files measure 0.000-0.082 dB
  channel separation against a 3.0 dB threshold: stereo containers holding
  duplicated mono. The mono/stereo charts compare recording provenance, not
  channel access.
- **`verbatimize` re-identifies redacted text**, taking the leak rate from 18.7%
  to 67.1%. See CLAUDE.md.

## Not in this directory

`outputs/private/leaks.csv` on gpu2 lists every leaked identifier with context.
It is PII in the clear by construction and is deliberately kept out of the repo,
out of the report and out of any shared artifact.

## Regenerating

From the cluster, with the sweeps already in `outputs/`:

```bash
COHORT=/data/data/wolfflab/Data/Chirp3_PSYCHS_NDA4
uv run python scripts/evaluate_systems.py --cohort $COHORT \
    --system chirp3 --system ours=outputs/ours \
    --system verbatimize=outputs/verbatimize --system baseline=baseline/outputs \
    --system baseline_llm=baseline/outputs --system ours_llm=outputs/ours_roles \
    --output outputs/results.json
uv run python scripts/error_taxonomy.py --cohort $COHORT ... --output outputs/taxonomy.json
uv run python scripts/score_redaction.py --cohort $COHORT ... --output outputs/redaction.json
uv run python scripts/score_partner_wer.py --cohort $COHORT ... --output outputs/partner_wer.json
uv run python scripts/score_jiwer_wer.py --cohort $COHORT ... --output outputs/jiwer_wer.json
```

Then locally, with the JSON fetched into `data/`:

```bash
uv run python scripts/plot_results.py data/results.json \
    --partner data/partner_wer.json --jiwer data/jiwer_wer.json --mono data/results_mono.json \
    --stereo data/results_stereo.json --redaction data/redaction.json \
    --taxonomy data/taxonomy.json --font dmsans.ttf --output eval_report.html
uv run python scripts/export_charts.py data/results.json \
    --partner data/partner_wer.json --jiwer data/jiwer_wer.json --mono data/results_mono.json \
    --stereo data/results_stereo.json --redaction data/redaction.json \
    --taxonomy data/taxonomy.json --output-dir figures --font dmsans.ttf
# same again with --clean --output-dir figures-clean for the caption-free set
uv run python scripts/export_tables.py --output-dir tables \
    --results data/results.json --mono data/results_mono.json \
    --stereo data/results_stereo.json --taxonomy data/taxonomy.json \
    --redaction data/redaction.json --partner data/partner_wer.json \
    --jiwer data/jiwer_wer.json
```
