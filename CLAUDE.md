# crisper-whisper-2

Verbatim, word-level, speaker-attributed transcription pipeline for wav files.
ASR by CrisperWhisper 2.0 (verbatim mode, CTranslate2 backend with speculative
decoding), speaker diarization by pyannote/speaker-diarization-community-1,
merged with the pyannoteAI diarization-ASR overlap algorithm. See spec.txt.

## Environment

- Development happens locally (macOS); deployment target is
  /data/data/wolfflab/btdixon/Dixon/crisper-whisper-2 on a remote cluster
  with 4x H200 GPUs (CUDA). Code is committed to git locally and pulled on
  the cluster for testing.
- Managed with uv. `uv sync` installs everything. Linux resolves
  torch/torchaudio from the pytorch cu128 index and gets the ct2 backend
  (`crisperwhisper[ct2]`, which pulls the `ctranslate2-crisperwhisper` fork
  plus CUDA cuBLAS libs); macOS falls back to CPU torch and the base
  crisperwhisper package (the ct2 fork ships Linux x86_64 wheels only), so
  local development imports and unit logic run, but model inference is
  cluster-only.
- HuggingFace auth: both pyannote/speaker-diarization-community-1 and the
  nyralabs CrisperWhisper models are hosted on HF (community-1 is gated;
  terms already accepted for the bentdixon account). IMPORTANT: the user's
  ~/.bashrc on gpu3 exports HF_HOME=/data/data/wolfflab/btdixon/caches/hf,
  and huggingface_hub reads the token from $HF_HOME/token — a token at the
  default ~/.cache/huggingface/token is ignored when HF_HOME is set. This
  caused a GatedRepoError 401 in interactive shells while non-interactive
  ssh sessions (no HF_HOME) authenticated fine. The token now exists in
  both locations, and the `hf` CLI (huggingface_hub 1.26.0, installed via
  `uv tool install huggingface_hub`) manages login: `hf auth login` /
  `hf auth whoami`. HF_TOKEN / --hf-token still override when needed.
- Model caches on gpu3: HF downloads land in $HF_HOME/hub in interactive
  shells (~/.cache/huggingface/hub without HF_HOME); converted CT2 models
  in ~/.cache/crisperwhisper (HF_HOME-independent). Reruns skip download
  and conversion.

## Layout

- src/crisper_pipeline/asr.py — CrisperWhisper 2.0 wrapper. Loads "large"
  with a "turbo" draft model for speculative decoding, transcribes in
  verbatim mode with word_timestamps=True, returns a plain dict of text +
  word list.
- src/crisper_pipeline/diarization.py — pyannote community-1 wrapper. Uses
  the pipeline's exclusive_speaker_diarization (non-overlapping segments,
  what the merge tutorial expects) and returns [{start, end, speaker}].
- src/crisper_pipeline/merge.py — assign_speakers() reproduces the
  https://docs.pyannote.ai/tutorials/diarization-asr-merge code exactly
  (maximum temporal overlap per transcript segment, optional fill_nearest
  fallback, else UNKNOWN); each CrisperWhisper word acts as one transcript
  segment. group_into_turns() folds consecutive same-speaker words into
  turns for the human-readable output.
- src/crisper_pipeline/outputs.py — writes, per input file, under
  <output-dir>/<stem>/: transcript.json (full word-level with speakers),
  transcript.txt (timestamped speaker turns), diarization.json, and
  speakers/<SPEAKER>.json + .txt per participant.
- src/crisper_pipeline/cli.py — `transcribe-session` entry point. Accepts
  wav files and/or directories, loads both models once, processes each file,
  continues past per-file failures.

## Running on the cluster

```bash
uv sync
export HF_TOKEN=hf_...
uv run transcribe-session /path/to/session.wav --output-dir outputs
# or a whole directory, with known speaker count:
uv run transcribe-session /path/to/wavs/ --num-speakers 2 --output-dir outputs
```

Useful flags: --fill-nearest (assign nearest speaker instead of UNKNOWN for
words falling in diarization gaps — likely desirable for verbatim fillers),
--no-speculative, --compute-type int8_float16, --device-index N,
--min-speakers/--max-speakers, --language.

## Design decisions

- Word-as-segment merge: the tutorial merges at ASR segment level, but
  CrisperWhisper's word timestamps are its headline feature (~30-41 ms
  boundary error), so the identical overlap algorithm is applied per word.
  This gives per-word speaker labels directly, which step 5 (per-participant
  word-level JSON) needs.
- Exclusive diarization: community-1 exposes exclusive_speaker_diarization
  precisely for ASR reconciliation (verified against pyannote-audio 4.0.7
  source); using it means overlap ties are rare and segments never overlap.
- fill_nearest defaults to False (matching the tutorial) both in the
  function and the CLI; enable with --fill-nearest after seeing real data.
- transformers is a hard dependency on Linux (crisperwhisper[ct2,transformers])
  even though inference runs on CTranslate2: the one-time HF -> CT2 model
  conversion calls transformers.AutoConfig and fails with a NameError
  without it.
- Audio is loaded in-memory (soundfile -> torch tensor dict) before being
  passed to pyannote, bypassing pyannote's torchcodec decoder. The PyPI
  torchcodec wheel is built against CUDA 13 (libnvrtc.so.13) and cannot load
  against our cu128 torch stack; the cu128 index only carries torchcodec
  0.9.x, which pairs with older torch. torchcodec stays installed (hard dep
  of pyannote-audio) and prints a loud import-time warning — it is harmless
  because the decoder is never used.
- Locked versions: crisperwhisper 2.0.1, pyannote-audio 4.0.7,
  transformers 5.14.1, torch 2.11.0+cu128 (Linux) / 2.13.0 (macOS),
  Python 3.12.

## Progress

- 2026-07-30: Steps 1-5 of spec.txt implemented. uv env resolves and syncs
  locally; merge algorithm, turn grouping, timestamp formatting, and CLI
  parsing unit-checked; CrisperWhisper and pyannote API surfaces verified
  against the installed packages.
- 2026-07-30 (after first cluster run): fixed two deployment failures found
  on gpu3 — missing transformers for the CT2 conversion, and the
  torchcodec/CUDA-13 mismatch (see Design decisions). Verified end-to-end
  on gpu3 (H200) with a 60 s clip of real interview audio: conversion,
  speculative decoding, diarization (num_speakers=2), merge, and all output
  files correct; 125 words attributed across 2 speakers plus 1 UNKNOWN word
  in a diarization gap. ASR ~1.5 s and diarization ~1.3 s for the clip.
  Repo on GitHub: bentdixon/crisper-pyannote. Next: full-length runs on the
  AMPSCZ wav files; consider --fill-nearest to eliminate UNKNOWN words.

- 2026-07-30 (merged to main, overlap-test branch deleted): added
  --diarization-mode overlap, an
  alternative merge that keeps pyannote's raw overlapping segments
  (output.speaker_diarization) and assigns each segment the contiguous ASR
  word chunk maximizing temporal IoU with it; words are unique to one
  segment (better initial fit claims contested words first, others re-fit
  on the remainder; leftovers become UNKNOWN turns). Turn dicts carry a
  "segment" key with the originating diarization bounds (None for
  UNKNOWN). Verified on gpu3 with the 60 s clip: same 113/11/1 word split
  as exclusive mode, 9 per-segment turns.
- 2026-07-30: outputs are per-run directories <stem>_<YYYYMMDD-HHMMSS>
  (same-second collisions get -2, -3, ...) each containing metadata.json
  with run timestamp, package versions, and transcription/diarization/merge
  settings including diarization mode.

## Fine-tuning community-1 (finetune/)

Answer to spec "For Future" item 1: YES, fine-tunable (CC-BY-4.0). The
segmentation model (a PyanNet; trunk attribute is sincnet, not wav2vec) is
the trainable component; embedding + PLDA + VBx clustering stay stock and
their hyperparameters get re-optimized instead. Full plan and rationale in
the conversation of 2026-07-30; workflow:

1. prepare_data.py: (transcript JSON, wav) pairs -> RTTM/UEM/LST +
   database.yml (protocol AMPSCZ.SpeakerDiarization.Interviews, scope:
   file). Accepts three transcript shapes: timestamped text ("S1
   00:00:01.451 text ..." — the human transcription format; start times
   only, ends synthesized from the next turn's start, bracketed markup like
   [psychs?] stripped for alignment), word-level JSON (pipeline
   transcript.json), or turn-level JSON ({"turns": [...]}). Turn-timed
   transcripts are refined via CrisperWhisper forced_align (drift check is
   start-anchored because text-format ends include trailing silence;
   fallback to human bounds when start drift > --max-drift). Splits
   80/10/10 by participant (AMPSCZ id regex).
   Known label caveats: overlap under-labeled (single ASR stream),
   untranscribed vocalizations become false non-speech.
2. evaluate.py: DER (collar 0/0.25 x with/without overlap) for any pipeline
   config on any subset; --word-attribution also scores word-level speaker
   attribution with optimal label mapping.
3. train_segmentation.py: Lightning fine-tune keeping the pretrained
   powerset geometry (duration 10 s, 3 classes, max 2 per frame); two-stage
   freeze (sincnet trunk frozen for --freeze-epochs, then unfrozen);
   monitors "loss/val"; links best checkpoint to finetune/checkpoints/best.ckpt.
4. optimize_pipeline.py: Optuna Optimizer on dev tuning min_duration_off +
   VBx threshold/Fa/Fb. Must build the pipeline with legacy=True (v4
   returns DiarizeOutput otherwise, which breaks the metric); the written
   finetuned-config.yaml deliberately omits legacy and uses explicit
   checkpoint dicts (the $model/... indirection resolves against the local
   config directory, not the hub).
5. transcribe-session --diarization-model finetune/finetuned-config.yaml.

Whole toolchain smoke-tested on gpu3 with synthetic data (10 files):
prepare -> 2-epoch train -> 2-trial optimize -> evaluate all ran end to
end. DER on synthetic noise is 1.0 because the model correctly finds no
speech in noise; real-data runs pending the human transcript location.

torchcodec note: the PyPI torchcodec wheel needs CUDA 13 libs
(libcudart.so.13, libnvrtc.so.13). nvidia-cuda-runtime + nvidia-cuda-nvrtc
are now Linux dependencies and crisper_pipeline.cuda_preload dlopens them
before pyannote imports, which enables pyannote's training dataloader
decoding and removes the old import-time warning wall. The in-memory
load_audio path remains the production decode route.

## Verbatimizing Chirp-3 transcripts (verbatimize-session)

CrisperWhisper 2.0 ships a dedicated `model.verbatimize(audio, transcript)`
task: it reproduces a trusted clean transcript word-for-word and inserts only
the disfluencies and vocal events present in the audio. This is neither
`transcribe(mode="verbatim")` (which re-transcribes from scratch) nor
`forced_align` (whose public API returns one word per *reference* token and
discards the hypothesis-only insertions -- so it can never verbatimize).

- Hard constraint: verbatimize takes a single decoder prompt, has no
  longform strategy, and warns above 30 s. Sessions are 4-75 min, so
  `verbatimize.build_windows` splits on Chirp's own word offsets into
  <=26 s windows, preferring pauses >=2 s as cut points. Window audio
  boundaries are the midpoint of the surrounding silence, except across
  Chirp's long gaps (failed chunks, up to ~800 s) where the slice hugs the
  words instead.
- Speaker labels come from Chirp (`speakerLabel` 0/1/2 -> SPEAKER_00/...);
  each output word is difflib-aligned back to the window's input words,
  matched words inherit that speaker, inserted disfluencies inherit the
  previous word's. Words carry `origin`: chirp | inserted | chirp-fallback.
- Guards: a window whose output is <0.5x or >3x the input word count (or
  empty, or raises) falls back to the original Chirp words for that window.

Data: two disjoint prefixes hold complete (audio + Chirp + human) sessions.

    prefix                            sessions  subjects  sites  audio
    gs://pronet_data/NDA_4/                 221       131      5  35 GiB
    gs://pronet_data/study_data_test/        48        23      5   7 GiB
    total                                   269       154     10  42 GiB (~198 h)

The two never overlap: NDA_4's Chirp finals cover PronetCA/CM/GA/HA/IR,
study_data_test covers PronetBI/SD/SF/SI/YA. An early count of "221" missed
the second prefix -- always check both. (gs://pronet_data/transcripts/ is a
dev scratch area: 140 objects, 3 sessions, chunk-level only, no finals.)
NDA_4 also holds 1480 audio / 1494 human transcripts whose sessions have no
Chirp output at all.

The study is **longitudinal**: 269 sessions across 154 participants, since a
participant is interviewed at several timepoints (sessions per participant:
1 x 69, 2 x 41, 3 x 14, 4 x 7 within NDA_4). The `dayNNNN` in each filename
counts from that participant's study baseline.

Staged on the cluster at /data/data/wolfflab/Data/Chirp3_PSYCHS_NDA4 as
<SITE>/<SUBJECT>/<dayNNNN_sessionNNN>/{audio,chirp,human}/ -- one directory
per visit, so the repeat structure is visible and a visit's three artefacts
sit together. Zero-padded days sort chronologically.

`final_gcp_transcripts/*.json` is the machine-readable
Chirp output ({startOffset,endOffset,word,speakerLabel}); the
`final_human_readable_transcripts/*.txt` are a rendering of the same 221.
Access needs the Yale account (`--account bd584@yale.edu`); the gmail
account 403s. gcloud/gsutil/rclone are NOT installed on gpu2 or gpu3, so
bucket->cluster transfer currently goes through the laptop.

### Result of the 4-session pilot (2026-08-07, gpu2, 4 sites)

90 windows, 0 fallbacks, 0 dropped Chirp words, timestamps monotonic and in
range, ~64 s wall for 44 min of audio (=> ~3-4 h for all 221 on one GPU).

`scripts/compare_verbatim.py` counted disfluencies three ways over the same
audio:

    source        words   fillers   events   filler rate
    chirp          4224        92        1       0.0218
    verbatimize    4266        99       40       0.0232
    verbatim       3624        60       65       0.0166

Conclusion: **Chirp-3 is already largely verbatim** (2.2% filler rate), so
verbatimize adds few fillers (+7) -- its real contribution is vocal-event
tokens (1 -> 40) and content preservation. From-scratch verbatim
transcription found 14% *fewer* words and *fewer* fillers than Chirp, which
is the strongest argument against re-transcribing this corpus.

### Timestamp accuracy: do NOT use --realign (2026-08-07)

`scripts/score_timestamps.py` scores word times against the human
transcripts' turn start times -- the only ground truth independent of both
Chirp and CrisperWhisper. 345 matched turn boundaries over the 4 sessions,
absolute error:

    source         turns   median    mean     p90   <=0.1s   <=0.5s
    chirp            339   0.327s  0.345s  0.553s    15.6%    89.1%
    windowed         345   0.317s  0.323s  0.486s    17.1%    89.9%
    forced_align     345   0.431s  2.745s  3.164s     8.1%    53.6%
    anchored         345   0.373s  1.900s  1.984s    11.9%    69.9%

Re-aligning after verbatimizing **hurts**, in both implementations tried.
`model.forced_align` was worst: it returns a timestamp for every reference
word, interpolating those its hypothesis never found, and a from-scratch
verbatim pass on this audio recovers ~14% fewer words than Chirp, so those
interpolations span wide gaps. Aligning manually and re-timing only
difflib-matched words ("anchored", what `realign_timestamps` now does) halves
the damage but still loses: only 46-89% of words anchor per session (worst
case 319/690, where the verbatim pass heard 424 words against Chirp's 684).
The flag is kept for future re-testing but defaults off.

Caveat on the ground truth: human turn times are annotated to roughly a third
of a second, so the ~0.32s floor shared by chirp and windowed is annotation
granularity, not ASR error -- this test cannot rank those two, it can only
detect the large regressions. Chirp's timestamps are *not* the weak point
they were assumed to be.

Known defect: the model normalizes filler surface forms inconsistently --
in one session 12 of 21 "Um" became "[UM]" and 2 of 4 "Uh" became "[UH]",
leaving mixed notation in one transcript. Needs a normalization pass before
the outputs are used for analysis.

## Other team's pipeline (baseline/) and its LLM review

`baseline/` is another team's transcription_core, ported to CrisperWhisper 2.0
(their v1 transformers ASR swapped for our CT2 2.0 model; `adjust_pauses`
dropped as a v1-only timestamp fix; segment audio passed in memory). Their
diarization stays: silero VAD + stereo channel-dominance gated on a real
loudness-separation check, pyannote 3.1 fallback for mono/fake stereo, plus
INTERVIEWER/PARTICIPANT role assignment and VTT/SRT export. pyannote 3.1 does
load correctly under pyannote-audio 4.0.7 (wespeaker + AgglomerativeClustering,
verified -- not community-1). Runs default to non-speculative decoding because
speculative decoding degrades word timings.

Their stereo guard earns its keep: the first test file reported 2 channels but
measured 0.03 dB average L/R separation (threshold 3.0), so it correctly fell
back to pyannote instead of assigning speakers from noise. 66 of our 269
sessions are stereo.

### LLM review verdict (2026-08-07): no measurable improvement

`baseline/apply_llm_corrections.py` closes the loop their pipeline leaves open
-- same prompt, same `_validate_llm_output`, but it applies the suggestions to a
*separate* corrected copy so both can be scored. Over 4 transcripts,
Qwen2.5-7B-Instruct suggested **0 word_corrections**, 1 speaker_flag, and
judged role_mapping correct every time. The JSON parsed cleanly in all four
(no `raw_response` key), so this is a real null result, not a decode failure.
`scripts/score_wer.py` confirms: WER 1.0199 for both variants, delta +0.0000,
0 visits better, 0 worse, 4 unchanged.

Two caveats on that verdict:
- The single speaker_flag did not apply -- the model's `turn_text` did not
  match any turn line verbatim, so the apply step could not locate it. Any
  future use of speaker_flags needs fuzzy turn matching.
- One transcript exceeded Qwen's 32768-token context. Sessions run up to
  123 min, so long files get a degraded review. Chunking the transcript would
  be required before trusting this at scale.

Absolute WER (>1.0) is NOT yet a usable quality number: the ASR is verbatim and
the human transcripts are semi-verbatim, and ASR/human word-count ratios range
0.89-1.41 across visits. Fix the metric before using it for the Chirp-3
comparison; the LLM delta is trustworthy only because it is exactly zero.

## Full six-system evaluation (2026-08-08, scripts/evaluate_systems.py)

Systems scored against the human transcripts with the DIALOG-DeID metrics, all
over the complete 269-visit cohort. Sweeps all reached 269/269.

**Naming**: short keys (`ours`, `baseline`) survive only as CLI arguments,
output directory names and dict keys. Every rendered output -- console tables,
JSON keys, the leaks CSV, chart labels -- names the ASR model, the diarization
system and any post-processing, from the one registry in `scripts/systems.py`.
Add a system there and nowhere else. `plot_results.load_results` folds either
form to the short key on read, so results files written before the rename still
render.

Final numbers (`outputs/results.json`, rescored 2026-08-11 after the sWER, WER
ordering and coverage fixes below):

    system         visits     WER  WERnoIns    sWER     DER  DERconf  QTP-F1
    chirp3            269  0.1667    0.1376  0.2984  0.1860   0.1599  0.8752
    verbatimize       269  0.1686    0.1412  0.2985  0.1796   0.1584  0.8814
    ours              269  0.1150    0.1041  0.1930  0.1620   0.1553  0.9214
    ours_llm          269  0.1157    0.1046  0.1970  0.1633   0.1564  0.9203
    baseline          269  0.1284    0.1063  0.1983  0.1487   0.1240  0.9048
    baseline_llm      269  0.1284    0.1063  0.2040  0.1508   0.1258  0.9037

Readings that survive scrutiny:

- **community-1 and pyannote 3.1 are level on sWER**: 0.193 vs 0.198, a gap of
  half a point. An earlier version of this file claimed 0.835 vs 0.517 and
  called it "pyannote 3.1 beats community-1 on speaker attribution". That was
  a metric artifact, not a result -- see the sWER section below. Do not quote
  the old figure.
- **pyannote 3.1 does lead DER confusion**, 0.124 vs 0.155. This is bounded,
  consistent and real, but it is 3 points, not the 32 the broken sWER implied.
- **community-1 leads WER (0.115), WER excluding insertions (0.104), sWER
  (0.193) and QTP-F1 (0.921)**, so pyannote 3.1's only win is DER confusion.
- **The LLM review is worse on every metric for both trees, 269/269 visits.**
  Never better, not once. Its speaker flags also carry fabricated
  justifications (one flagged a turn for "contains '?' and starts with 'how
  often'" when it contained neither) and only 15 of 163 applied. Drop it.
- Within the covered span all systems sit at 0.92-0.94 hypothesis words per
  reference word: slightly fewer words than the transcript, because deletions
  now outweigh insertions.

### sWER was measuring stream count, not attribution (fixed 2026-08-11)

The 32-point sWER gap was an artifact of two flaws in the metric, both mine.
Traced on `PronetCM/CM05540/day0082_session001`:

    ours      ref "S2:"  1 word -> hyp UNKNOWN  88 words   WER 88.000
    baseline  ref "S2:"  1 word -> hyp None      0 words   WER  1.000

The human transcript has a stray `S2:` line carrying one word. Our pipeline
exposes a third stream (the `UNKNOWN` bucket for words in diarization gaps,
about 1% of words), so `linear_sum_assignment` must match the phantom to it and
scored 88.0; the other pipeline emits two streams, so the phantom went
unmatched and was charged the capped 1.0. **An unmatched stream was capped at
1.0 while a badly matched one was unbounded**, so emitting an extra stream cost
more than losing a speaker outright. 8 visits exceeded sWER 3.0 for ours
against zero for baseline, and those 8 carried the entire corpus gap; the
medians were 0.297 and 0.295 throughout.

Two corrections in `score_visit`: every matched stream is capped at 1.0, and
reference streams under `MIN_REFERENCE_WORDS` (5) are dropped as transcript
formatting artifacts (12 corpus-wide, identical across all six systems -- the
count is logged per system and a mismatch is an error, since the filter is a
property of the reference). `swer_uncapped` is retained the way
`DER_word_level` is, so the correction stays auditable.

Community-1 itself was never at fault: it found exactly 2 speakers on all 269
files.

### The human transcripts stop early, and scoring must respect that

The single largest distortion in this evaluation, found on 2026-08-11 by asking
what the insertions actually were. `scripts/coverage.py` now restricts every
scorer to the span each human transcript covers; before that, whole sessions
were scored against partial references and the untranscribed remainder was
charged as insertions.

    human transcript coverage of audio, 269 visits
    min 26%   p10 47%   p25 64%   median 96%   p75 100%   max 100%
    97 visits under 80% covered, 34 under 50%
    mean transcribed 26.1 min against mean audio 37.2 min

The worst cases stop at a hard cutoff regardless of session length -- 32.0 of
122.8 minutes, 31.9 of 118.2, 31.9 of 107.6, 31.9 of 102.5, 32.0 of 100.9 --
so roughly the first half hour was transcribed and the rest was not.

The tell was that two systems sharing no code inserted nearly the same 8400
words on one visit (Chirp-3 from 0:53:20, ours from 0:59:22, both to 2:02:48).
Independent systems do not hallucinate an hour in agreement; the reference had
ended. Corpus-wide, 97% of inserted words sat in unbroken runs of 20+ words.

Effect of restricting to the covered span, for our pipeline:

    WER          0.419 -> 0.115
    insertions   0.314 -> 0.011
    sWER         0.421 -> 0.193
    QTP-F1       0.883 -> 0.921

The window runs from the first turn's start to the **last turn's start**, not
its end: turn ends are synthesized from the following turn's start, so the
final turn has no real end and `load_timestamped_text` stretches it to the full
audio duration -- exactly the untranscribed remainder being excluded. Dropping
that one turn costs a few words out of hundreds.

Applied in `evaluate_systems.py`, `score_partner_wer.py`, `error_taxonomy.py`,
`score_redaction.py` and `export_insertions.py`. Any number produced before
this change is measuring reference truncation.

### There is no stereo channel advantage on this corpus

The obvious explanation for the DER-confusion gap was that the other team's
pipeline reads speakers off stereo channels rather than diarizing. It does not,
here. `scripts/classify_channels.py` runs *their* `has_real_channel_separation`
(silero VAD per channel, then average |L-R| dB at real speech moments, 3.0 dB
threshold) over every file: 203 are mono, and all 66 stereo files measure
**0.000 to 0.082 dB**. Every one is a stereo container holding duplicated mono,
so the channel-dominance path never fired and all 269 files were diarized by
pyannote from a downmix. The forced-mono re-run (`--force-mono` in
`transcription_core.py`, kept for future corpora) was therefore unnecessary and
was not run.

What the split does show is that audio provenance dominates everything else:
`outputs/results_mono.json` (203) against `outputs/results_stereo.json` (66),
chirp3 WER 90.9% vs 77.1%. A 14-point swing between recording conditions, where
the best system beats the worst by 4.

### The longform ASR bug -- ours was transcribing 34% less audio

Do not use `model.transcribe()` directly on a full session. CrisperWhisper's
default `longform_strategy="continuation"` drops most of the transcript on this
corpus. On a 1094 s interview:

    continuation (default, stride 26)   651 words   <- what we shipped
    ... stride 20                       802
    ... stride 15                      1203
    chunked_lcs                        1637        (no word timestamps)
    other team's VAD segments          1752
    Chirp-3                           ~1730
    windowed short-form (the fix)      1725

Each 30 s chunk emitted only ~16 words regardless of stride, where dense speech
gives 60-90, so chunks were ending early and a smaller stride merely packed in
more of them. Every other knob was inert (hallucination_mitigation 654,
max_new_tokens 448 -> 655, boundary drop off 658) and disabling
early_eot_recovery made it *worse* (616). The same audio cut into 25 s clips and
transcribed in isolation came back complete, so the loss is in stitching, not
decoding.

The trap: `continuation` is both the model default and the only strategy that
implements `word_timestamps=True`, which this whole design rests on --
`chunked_lcs`/`token_lcs` raise NotImplementedError. So the only usable strategy
was the broken one, and it fails silently with no warning.

Fix in `asr.py`: audio over 30 s is split on silero VAD into <=25 s windows,
transcribed short-form, and each window's timestamps offset back onto the
session clock. Cohort-wide this moved ours from 0.656 to 0.973 of baseline's
word count (median per visit 0.975, min 0.833, none below 0.80; previously as
low as 0.01). The old path remains as `longform="continuation"`; the pre-fix
outputs are kept at `outputs/ours_continuation_broken/` for audit.

Consequence for reading older notes: ours' *apparent* WER win of 0.7005 was an
artifact of under-transcribing -- fewer words means fewer insertions against a
semi-verbatim reference. Any metric computed before 2026-08-08 21:40 on the
ours/ours_llm trees is invalid, and any WER computed before the coverage fix of
2026-08-11 is invalid for a second, larger reason.

Two data defects, not pipeline defects:

- `PronetGA/GA06750/day0088_session001`: 25 words for a 2037 s file, unchanged
  by the fix; the other team's pipeline finds 30. The audio is near-silent or
  corrupt. It drags every system equally.
- A tail of visits at 0.83-0.87 of baseline's word count (HA32687 twice,
  CA09370, YA03473, CA00152) where VAD may be clipping quiet speech. Not
  distorting the comparison; worth a look if that tail matters.

### DER as first written was invalid -- do not revert it

`load_timestamped_text` synthesizes each turn's end from the *next* turn's
start, so the reference tiles the entire recording and contains no non-speech
(measured ref coverage: exactly 1.00 on every visit). Scoring raw word spans
against that reference charges every inter-word silence as missed detection.
Decomposition over six sessions (`scripts/diagnose_der.py`):

    false alarm        0.000  on every visit -- structurally impossible in a
                              real DER, and the tell that the reference admits
                              no silence
    missed detection   0.41-0.63  of a 0.54-0.76 DER
    confusion          0.10-0.27  -- the only part that is speaker error

So that DER ranked systems by how much of the timeline their segments covered,
penalising word-level output (ours, verbatimize) against systems emitting
contiguous turns (baseline). It is why baseline's 0.56 first appeared to beat
ours at 0.75.

Fix: build the hypothesis by the reference's own rule -- consecutive
same-speaker words grouped into a turn, each extended to the next word's start
-- so both sides tile the timeline and only label disagreement moves DER. On a
6-visit check DER roughly halved (chirp3 0.733 -> 0.369, ours 0.808 -> 0.408).
`DER_confusion` is reported separately as the pure speaker-attribution error
and `DER_word_level` retains the old number so the change stays auditable.

### Two bugs that hid themselves, both worth remembering

- `load_chirp` took (visit, root) while adapters are called with
  (visit, root, relative). Every call raised TypeError, and the caller's
  `except Exception -> words = None` turned that into "this system has no
  output", so chirp3 -- the incumbent -- silently scored 0 of 257 visits and
  simply vanished from the results table. Adapter exceptions are now counted
  and a system scoring zero visits is logged as a failure.
- `scripts/finalize_after_transfer.sh` read cluster state from a remote
  pgrep's exit status, so ssh failing (255) was indistinguishable from pgrep
  finding nothing (1). When the bastion (`ecco`, the only route to gpu2/gpu3)
  went down for ~10 h mid-run, every wait loop fell through: it logged
  "top-up sweeps finished", failed to launch the scoring chain because that
  ssh also failed, and logged COMPLETE having run nothing. Cluster state is
  now three-valued (busy/idle/unknown) from a sentinel the remote shell
  prints; only an affirmative idle advances.

General lesson for this repo's remote tooling: over ssh, an error path and a
legitimate state must never share a signal. The same class of bug appeared
three times (pgrep self-match, adapter except, ssh-vs-pgrep exit status).

A second family, all of which produced plausible-looking tables rather than
errors:

- **A permissive pattern reading its own output.** `redaction.PLACEHOLDER`
  started as `\[[A-Z][A-Z_]*\]`, which matches CrisperWhisper's `[UM]` and
  `[UH]`. The unredacted pipeline scored 30,487 "redactions" over 269 visits
  and produced a complete, believable table. The label set is now closed
  (`PII_LABELS`) and must stay in sync with `redact_llm.LABELS`.
- **Two output shapes behind one adapter.** The other team's pipeline writes a
  bare JSON array; everything this repo writes is an object with a `words` key.
  The redacted copies of their files are objects, so the adapter returned a
  dict and the caller iterated its keys, failing as `'str' object has no
  attribute 'get'` far from the actual mistake. `as_words` now accepts both.
- **Shell precedence in a two-job launch.** `cd X && A & B &` parses as
  `(cd X && A) & (B) &`, so the second scoring job ran from the home directory
  and died on a missing file while the first succeeded. Launch background jobs
  one `ssh` invocation each, or repeat the `cd`.
- **A regex written for one of three spellings.** The partner-WER reference
  parser matched only `S1 HH:MM:SS`, but 56k of 69k transcript lines use
  `INTERVIEWER:`/`PARTICIPANT:`. Speaker tags and timestamp digits stayed in
  the reference on most files, inflating every system's WER by ~7 points
  equally, so nothing looked wrong. The word-ratio cross-check caught it.

### Known caveats on the numbers

- Coverage was equal (269) for ours/verbatimize/baseline at final scoring, but
  the report also computes every metric over the common subset of visits all
  systems completed, because sweeps finish at different times.
- Absolute WER > 1.0 remains uninterpretable (verbatim ASR vs semi-verbatim
  reference). Differences between systems are meaningful; the level is not.
- verbatimize carries ~124 window-overflow fallbacks corpus-wide
  (`RuntimeError: No position encodings are defined for positions >= 448` --
  the deferred window-clamp defect). Those windows keep their original Chirp
  words, so no words are lost but they are un-verbatimized (~0.2% of windows).
- At least one Chirp file has degenerate offsets (BI11459/day0154_session001,
  the one file lacking a `_final` suffix): zero word-span coverage.
- Chirp sometimes emits 3 speaker labels where the human transcript has 2,
  which inflates sWER via unmatched reference streams under the Hungarian
  assignment.

Report generator: `scripts/plot_results.py` renders results.json to a
self-contained HTML page (DM Sans embedded as a data URI -- the artifact host
blocks font CDNs, so a linked webfont silently falls back).
`scripts/export_charts.py` writes each figure as standalone HTML + PNG (11 of
them: six metrics, the partner metric, WER composition, the two mono/stereo
dumbbells, and PII redaction). Both take `--partner`, `--mono`, `--stereo` and
`--redaction`. PNGs are screenshotted in headless Chrome with viewport slack
and cropped in Python, because `--window-size` counts browser UI and silently
drops the bottom of every image; always verify by decoding pixels rather than
trusting the exit code.

Full regeneration:

    uv run python scripts/plot_results.py results.json --partner partner_wer.json \
        --mono results_mono.json --stereo results_stereo.json \
        --redaction redaction.json --font dmsans.ttf --output eval_report.html
    uv run python scripts/export_charts.py results.json --partner partner_wer.json \
        --mono results_mono.json --stereo results_stereo.json \
        --redaction redaction.json --output-dir charts --font dmsans.ttf

### Cross-check with the partner team's WER (2026-08-11)

`scripts/partner_compare.py` is their compareFiles.py vendored unmodified;
`scripts/score_partner_wer.py` imports its `tokenize`/`replace_fillers`/
`analyze` and runs them over the same 269 visits and the same adapters as
`evaluate_systems.py`. Their metric is not ours: difflib `SequenceMatcher`
(a "replace" block costs `max(ref_len, hyp_len)`, and it maximises the
matching subsequence rather than minimising edit distance, so it is an upper
bound on WER), three tiers (raw / punctuation- and case-normalized /
filler-normalized), and fillers *collapsed to a `[filler]` token on both
sides* rather than deleted. Results (`outputs/partner_wer.json`):

    system         raw     norm  filler-norm   median  ratio
    ours         56.44%   43.77%     43.74%    21.87%  1.305
    ours_llm     56.55%   43.89%     43.86%    21.87%  1.306
    baseline     58.05%   45.60%     45.56%    23.51%  1.331
    baseline_llm 58.05%   45.60%     45.56%    23.51%  1.331
    chirp3       60.33%   47.95%     47.69%    28.87%  1.339
    verbatimize  60.80%   49.00%     48.82%    30.55%  1.320

**Same ranking as our WER-excluding-insertions, from independent code**: ours
< baseline < chirp3 < verbatimize, and the LLM review is again never better
(9 of 269 visits worse for ours, 3 for baseline, rest identical). Per-visit
Spearman against our WER is 0.70-0.75. The two metrics also agree on word
ratio (1.30-1.34 here, 1.33-1.36 ours), which is the check that the reference
text is the same on both sides.

Two things this run added that ours did not show:

- **Mean and median diverge hugely** (ours: 43.7% vs 21.9%). A tail of ~30
  visits above 100% WER drags every mean. On 21 visits *every* system emits
  more than 2x the reference words -- those are truncated human transcripts,
  not ASR failures, and they average 173% WER. Excluding them: ours 32.8%
  mean / 19.2% median, baseline 34.4 / 19.5, chirp3 36.4 / 25.8. The median
  is the number to quote.
- **Chirp-3 wins the easy visits and loses the hard ones.** At p10/p25 it is
  the best system (4.7% / 8.7% vs ours 9.4% / 11.7%), but its p90 is worst
  (115% vs 103%). Head-to-head ours beats chirp3 on only 124 of 269 visits
  and its *median* visit is 1.7 points worse -- ours' aggregate win comes
  entirely from the failure tail. Against baseline the win is broad instead:
  ours better on 215 of 269.

Defect found and fixed while writing this: the first reference parser matched
only the `S1 HH:MM:SS` prefix, but 56k of 69k transcript lines use
`INTERVIEWER:`/`PARTICIPANT:`. Speaker tags and timestamp digits survived into
the reference on most files, inflating every system's WER by ~7 points and
depressing the word ratio to 1.13. `reference_prose` now reuses
prepare_data's `TIMESTAMPED_LINE`, whose speaker field is a bare `(\S+)`.
The word-ratio agreement with our metric is what caught it.

### Third WER implementation: the study team's jiwer script (2026-08-12)

`scripts/nvidia_wer.py` is that script vendored unmodified;
`scripts/score_jiwer_wer.py` imports its `preprocess_transcript` and
`calculate_wer` and runs them over the same 269 visits, the same adapters and
the same coverage window. Its rule: delete everything inside square brackets,
strip all punctuation, lowercase, then jiwer's edit distance. Unlike the
partner metric this is a true minimum edit distance, so it lands between ours
and theirs.

    system         visits   pooled    mean   median     sub     del     ins   no-fill
    ours              269   0.1447  0.1427   0.1222  0.0436  0.0906  0.0105    0.1216
    ours_llm          269   0.1455  0.1435   0.1223  0.0441  0.0906  0.0109    0.1225
    baseline          269   0.1584  0.1546   0.1360  0.0559  0.0815  0.0209    0.1364
    baseline_llm      269   0.1584  0.1546   0.1360  0.0559  0.0815  0.0209    0.1364
    chirp3            269   0.1746  0.1747   0.1273  0.0516  0.0926  0.0303    0.1704
    verbatimize       269   0.1872  0.1876   0.1318  0.0556  0.1039  0.0277    0.1732

**Same ordering again, from a third independent implementation**: ours <
baseline < chirp3 < verbatimize, and the LLM review is never better. Three
alignment rules (Levenshtein via jiwer, Levenshtein via our scorer, difflib via
theirs) and three normalization conventions now agree, which is the strongest
statement available about the ranking.

Two properties of their rule worth knowing:

- **Deleting bracketed content is not system-neutral.** It removes the human
  transcripts' `[inaudible]`, which is the intent, but it also removes every
  CrisperWhisper filled pause -- CW2 writes `[UM]`/`[UH]` while Chirp writes a
  plain "um" that survives -- so the CW2 arms are charged deletions for
  disfluencies they did transcribe. The `no-fill` column drops filled pauses
  from both sides for every system alike (`FILLER_RE` from partner_compare):
  the CW2 lead *widens*, 0.122 against chirp3's 0.170, so the asymmetry was
  working against the systems that win. Deletions are the largest term for
  every system either way.
- **Their file loader is not usable on this corpus** and is bypassed: it
  matches only `^S\d+:\s+HH:MM:SS.mmm`, so on the 56k `INTERVIEWER:` lines the
  tag and timestamp digits would become reference words. The reference comes
  from the windowed turns instead, same as every other scorer. Their brace
  handling is left alone (unlike the partner tokenizer, they do not strip
  `{...}`, so 876 PII surface forms stay in the reference -- a rounding error).

Reported as `JiwerWER` in the report, `figures/jiwer-wer.png` and
`tables/jiwer-wer.csv`. The per-visit key is `jiwer_wer`, deliberately not
`wer`, so merging it cannot overwrite our own per-visit WER.

### Our aggregate WER win is ten interviews, not a broad lead (2026-08-19)

Per-visit jiwer WER, ours against chirp3, all 269:

    ours better on 118 interviews, chirp3 better on 151
    median paired difference          chirp3 better by 2.4 points
    mean                              chirp3 0.175, ours 0.143
    mean without chirp3's 10 worst    chirp3 0.155, ours 0.143

So the corpus-level ranking is real but it is not a broad lead: **chirp3 wins
the typical interview** and our mean win comes from a tail where chirp3 exceeds
50% WER and we do not. Two figures make this the readable claim rather than a
caveat -- `wer-head-to-head` (one sorted bar per interview) and
`wer-distribution` (cumulative curves, which cross at about 13% WER). Both are
built from per-visit rows in `reports/2026-08-11/data/jiwer_wer.json` by
`headtohead_svg` / `ecdf_svg`, exported via `export_charts.py --jiwer`.

Two visits run off the head-to-head scale (worst -67 points), including the
known near-silent GA06750 file; they are drawn as wedges at the edge and
counted in the caption rather than allowed to set the axis.

### Lost turns: what sWER cannot see, and why the strict test was wrong

`scripts/lost_turns.py` (CPU, ~90 s for four systems over 269 visits) asks per
human turn whether any word landed inside its span and whether any of those
carried the matched speaker. 68,950 turns:

    system              lost   strict   wrong speaker or missing
    CW2 + pyannote 3.1  1.04%   2.82%                     14.95%
    CW2 + community-1   1.28%   6.35%                     25.18%
    chirp3              4.37%   5.82%                     22.53%
    verbatimize         7.40%   8.96%                     24.65%

**The strict column is not usable and the first version of this table quoted
it.** A turn counted as lost when no word fell inside its annotated span, which
cannot separate a boundary drawn approximately from speech that was never
transcribed. Distance to the nearest word when a turn is "lost":

    system              within 0.25s   within 1s    median   over 5s
    chirp3                     21.8%       26.9%    44.18s     67.7%
    verbatimize                15.2%       18.8%    47.59s     75.3%
    ours                       68.3%       85.0%     0.13s     10.5%
    baseline                   54.9%       70.4%     0.20s     22.8%

Two different failures. Chirp-3 drops whole regions -- a median of 44 seconds
to the nearest transcribed word, which is the known long-gap behaviour also seen
by verbatimize's window builder. Ours mostly has the words a tenth of a second
outside an approximate boundary. `score_timestamps.py` put the transcripts'
annotation granularity at roughly a third of a second, and turns under one
second are shorter than three times that, so the strict test was measuring
annotation on exactly the bucket the finding rested on. `TOLERANCE = 0.5` is
the reported measure now; the strict count is kept beside it.

Consequences of the correction:

- **The other team's pipeline does not lose materially fewer turns.** 1.04%
  against our 1.28%, not 2.8% against 6.3%. Both are 3-4x better than chirp3.
- **Chirp-3's turn loss is flat across turn length** (4.2-4.6% in every bucket),
  which is what losing regions rather than sentences looks like. The two CW2
  pipelines lose short turns about twice as often as long ones (2.0% under a
  second against 0.8-1.0% over five).
- **The overlap story weakens further.** Under tolerance, ours by previous-turn
  length runs 0.72% / 1.20% / 1.61% / 1.36% -- no longer monotonic, and the
  short-turn cross-tab peaks in the middle buckets on counts of 21-91 turns.
  Neither the rapid-exchange framing nor the long-stretch framing survives as a
  clean effect. That split stays in the JSON and the console output; the figure
  groups by turn length, where the systems genuinely differ in shape.

**sWER remains the wrong instrument for any of this** and no fix makes it the
right one: it pools every word a speaker said into one stream, so a lost
four-word turn is four deletions among five thousand words; it cannot separate
"never transcribed" from "wrong speaker"; and comparing concatenated text says
nothing about whether the exchange survived as an exchange.

Four figures, all from `export_charts.py --lost-turns`:

    lost-turns              lost rate by turn length, one group per length
    turn-outcomes           every turn: lost / just outside the boundary /
                            wrong speaker, stacked, so their relative size is
                            visible -- wrong speaker is 12-19% and dwarfs loss
    lost-turn-distance      distance to the nearest word when a turn is lost;
                            the figure that says how to read the others
    turn-outcomes-by-role   the same split by interviewer against participant

`turn-outcomes-by-role` carries a finding of its own: our pipeline misattributes
the **participant** far more than the interviewer (29.4% of participant turns
against 20.2%), while pyannote 3.1 is near-even (15.7 / 14.5). The participant
stream is the data, so that is the worse way round.

### Zero-duration words were all UNKNOWN (fixed 2026-08-19)

`merge.assign_speakers` assigns by maximum temporal overlap, and an instant
overlaps nothing, so every word whose end equalled its start fell through to
UNKNOWN by construction: 2,809 corpus-wide, 24% of all 11,544 UNKNOWN labels,
zero exceptions. `asr.transcribe_windowed` produces them via
`"end": max(word_end, word_start)`. Fixed in merge as a containment query on the
instant -- padding the duration in asr would move every reported timestamp and
DER with it. `fill_nearest` still decides genuine gaps, so a zero-duration word
in a gap behaves like any other word there. Checks in `tests/test_merge.py`
(`uv run python tests/test_merge.py`; no pytest in this environment).

Bound on the effect: WER cannot move, it ignores speakers. sWER and DER can move
by at most the 0.24% of words involved. The existing 269-visit outputs were not
rescored -- re-deriving speakers needs diarization output the sweep did not
keep, ~3 GPU-hours -- so published sWER/DER predate the fix by that bound.

Same commit, same family: `score_visit` matched each reference speaker to at
most one predicted speaker and **dropped the unmatched predicted streams
entirely** rather than charging them. Our UNKNOWN bucket therefore cost nothing,
while a system putting the same words on the wrong speaker paid for them; chirp3
emits no UNKNOWN stream, so only we benefited. Those words are now charged to
the reference speaker whose turns cover that moment. Old value retained as
`swer_unmatched_dropped`, the way `swer_uncapped` is.

## PII redaction (2026-08-11)

Answers the DIALOG-DeID section 2.3 question on this corpus: span-level
precision / recall / micro-F1 against gold PII spans.

**The gold already exists in the data.** Transcribers wrap identifying material
in curly braces: 887 spans across 142 of 269 sessions. (This is why the partner
team's `compareFiles.py` strips `\{[^}]*\}` before scoring WER.) The convention
splits cleanly by data prefix, which decides what each span can be used for:

    annotation style          sites                        spans
    {redacted}, scrubbed      CA CM GA HA IR (NDA_4)         577
    {isaiah}, surface kept    BI SD SF SI YA (study_test)    310

Scrubbed spans still count for span-level P/R/F1, which is positional, but
cannot be leak-tested. Zero crossover between the two styles.

**Chirp-3 already redacts** into `[PERSON_NAME]`, `[DATE]`, `[LOCATION]`,
`[AGE]`, `[GENDER]`, `[US_STATE]`, `[DATE_OF_BIRTH]` -- 4803 tokens corpus-wide.
Our pipelines redact nothing, so `scripts/redact_llm.py` runs Gemma 4 31B over
their output to make the comparison redactor-against-redactor.

Matching is by token alignment, not timestamps: a gold span and a placeholder
share no surface text ("isaiah" against "[PERSON_NAME]"), so they land in the
same difflib replace block, which is exactly the correspondence needed.

`outputs/redaction.json`, all 269 visits:

    system                          recall  precision      F1   leak
    community-1 + Gemma 4 31B        73.5%      44.1%   55.1%  10.3%
    pyannote 3.1 + Gemma 4 31B       73.8%      41.6%   53.2%  12.9%
    chirp3 (native)                  63.8%      27.9%   38.8%  16.8%
    verbatimize                       8.5%      27.4%   12.9%  64.2%
    ours / baseline (no redaction)     0 %          -       0%  ~75%

- **Gemma beats Chirp-3's native redaction on every measure** while redacting
  *less* (1449 spans vs 1996): 10 points more sensitive, 16 more precise, and
  a third lower leak rate.
- **verbatimize re-identifies Chirp's redacted output.** It destroys 87% of
  Chirp's redactions and takes the leak rate from 16.8% to 64.2%. Verified by
  hand: `[DATE]. [DATE].` -> "May ninth", `[AGE]` -> "three". The verbatimize
  task transcribes from audio using Chirp's text as a guide, so where Chirp
  wrote a placeholder the model simply hears the real words. A pipeline that
  consumes de-identified input and emits identified output is a privacy
  regression -- this alone probably disqualifies verbatimize for release.
- **Recall is trustworthy, precision is a lower bound.** A gold span is real
  PII, so a miss is a real miss. But only 142 of 269 transcripts carry any
  annotation, so a genuine identifier nobody marked counts against a system
  that caught it. Gemma flagging "Boylston" and "MBTA" (city-identifying,
  unmarked by the transcriber) is the canonical case.

### Turn-rewrite mode, and why it did not replace chunking (2026-08-13)

`redact_llm.py --mode turn` is a second protocol: one speaker turn per call, the
model returning the turn verbatim with PII replaced inline by `[LABEL]`. The
rewrite is never trusted -- `align_rewrite` difflib-aligns it back onto the
original words, only placeholder-for-word substitutions are honoured, and a turn
altered past `MIN_VERBATIM_RATIO` (0.85, counting redacted words as accounted
for) is dropped unredacted and counted.

A 4-transcript pilot looked decisive: recall 89.8% against chunk mode's 84.7%,
leaks 6 against 16 of 59 spans. **It did not survive a held-out set.** On 24
transcripts and 180 leak-testable gold spans (`scripts/select_validation.py`,
which requires surface-kept spans -- only 35 visits corpus-wide have them):

    system         gold   TP   FP  recall  precision     F1   leak
    turn            180  160   97   88.9%      62.3%  73.2%   1.7%
    chunk           180  157   84   87.2%      65.1%  74.6%   5.0%
    chirp3          180  134  141   74.4%      48.7%  58.9%   7.8%

The spans are paired, so the test is exact McNemar on the discordant ones, not
two independent proportions:

    detection   chunk vs turn   only-chunk  1   only-turn  4    p=0.375
    leaks       chunk vs turn   only-chunk  7   only-turn  1    p=0.070
    leaks       chirp vs turn   only-chirp 12   only-turn  1    p=0.0034
    detection   chirp vs turn   only-chirp  9   only-turn 35    p=0.0001

So: **turn mode is indistinguishable from chunk mode on detection** (a 3-span
difference), its lower leak rate is suggestive but not significant at n=180, and
it over-redacts more (FP 97 vs 84, F1 slightly *worse*). Both LLM modes beat
Chirp-3 decisively on both. The pilot's apparent win was small-sample regression:
its entire leak advantage was ten spans in one visit.

Decision: **chunk mode stays the default.** Turn mode costs ~26x the calls
(7,116 turns for 24 transcripts against 2,814 chunks for 269) for no established
accuracy gain. It is kept, working and validated, behind `--mode turn`.

Reliability was never the issue: 5 turns of 7,116 fell back (0.07%), zero
unmatched spans across the pilot and validation runs combined.

Cost, after `--batch-size` and `--prefix-cache` (below): 0.0356 s/word, i.e.
11.6 GPU-hours per tree corpus-wide, 23 minutes wall for the 24 validation
transcripts across three H100s.

### Possessive names were structurally unredactable in chunk mode (2026-08-13)

`locate()` matches the model's quoted span against normalized word tokens, and
`normalize` keeps apostrophes: "Zoe" -> `zoe`, "Zoe's" -> `zoe's`. The prompt
said *quote ONLY the identifying words themselves*, so the model quoted "Zoe",
which can never match the token "Zoe's" -- the span was dropped and counted as
an unmatched quote. Possessive names could not be redacted by chunk mode no
matter how well the model did its job, and this was invisible because a dropped
quote looks exactly like a chunk with no PII in it.

Both prompts now require the possessive form, and `apply_labels` carries a
trailing possessive onto the placeholder, so a redacted sentence still says
whose thing it is: "Zoe's gonna hop on" -> "[PERSON_NAME]'s gonna hop on".

Turn mode had a second, self-inflicted version of the same defect, found only
because the first rerun's result looked implausible. The new rule tells the model
to write "[PERSON_NAME]'s", but `PLACEHOLDER_TOKEN` allowed brackets plus
trailing *punctuation* only -- so the aligner read the possessive placeholder as
an ordinary word, saw an edit block with no placeholder in it, and discarded a
correct redaction through the `if not labels: continue` path **without counting
it**. The prompt and the parser were changed in one commit and only the prompt
was tested. A probe sending that turn to the model three ways showed all three
replies contained "[PERSON_NAME]'s gonna help out" while the aligner returned
`{}` with zero unmatched: `redacted_words: 0` on a transcript the model had
handled correctly. `PLACEHOLDER_TOKEN` now accepts an optional possessive (both
apostrophe forms), and bracket-shaped output the aligner cannot parse increments
`unmatched` so this can never be silent again.

Rerun on the same 24 validation transcripts and 180 spans, after both fixes:

    system                  TP   FP  recall  precision     F1   leaked
    chunk + possessive     161   81   89.4%      66.5%  76.3%     8
    chunk                  157   84   87.2%      65.1%  74.6%     9
    turn + possessive      164   98   91.1%      62.6%  74.2%     2
    turn                   160   97   88.9%      62.3%  73.2%     3
    chirp3                 134  141   74.4%      48.7%  58.9%    14

**Both protocols gained exactly 4 spans and lost none** -- strictly
one-directional, which is what a mechanical fix should look like rather than a
prompt nudge trading one error for another. Turn mode now leads recall (91.1%),
chunk mode leads precision and F1 (66.5%, 76.3), and the detection difference
between them is 4 spans against 1 (p=0.375), still not significant.

Leak counts need the same scepticism as everything else here: a leak is any gold
span whose surface survives, and the braces mark material that is not
identifying. Classified, **turn mode's 2 remaining leaks are a bare month and a
three-word statement about religion -- zero genuine identifiers**, while chunk
mode's 8 are 7 single words plus that same phrase. If those 7 are names, turn
mode's leak advantage (7 discordant against 1, p=0.070) is real and is the
argument for its ~26x compute cost; judging them needs a human read of
`outputs/private/leaks_validation_poss.csv`.

### Batching and prefix caching for turn mode

The pilot's per-call time was flat at 1.6-2.6 s while turn length varied
twofold, so the cost was fixed per call, not per word: a 564-token shared prompt
prefix re-encoded in front of a turn averaging twenty tokens.

- `--batch-size N` groups length-sorted turns into one `generate()`.
- `--prefix-cache` prefills the instructions and both worked examples once and
  reuses the KV cache. Padding sits *between* the prefix and the turn, so the
  prefix stays a true token prefix of every row -- left-padding the whole prompt
  would put pad tokens in front of it and misalign the cache. Whether it really
  is a token prefix is checked per batch (`PrefixCache.covers`), because
  tokenizers can merge across a string boundary; a mismatch falls back to full
  prompts with a warning.

Measured 2.76x on the pilot's four transcripts (2,044 s -> 740 s) with
**bit-identical output**: 100 redactions, zero differences. The remaining loss is
structural to static batching -- `generate()` runs a batch until every sequence
finishes, so a batch pays for its longest member (uniform-length SI00132 got
4.0x, ragged YA03473 got 1.8x). Continuous batching (vLLM) would fix that, and
would subsume the static prefix cache; vLLM is not installed in this project's
environment.

Two operational notes for this environment: torch orders CUDA devices
fastest-first by default, so `cuda:0` is the first H100 (nvidia-smi index 4) and
not the A100 at index 0; and `ssh host 'cmd &'` still waits for the channel, so
detached launches need `-n` and `</dev/null`, or `setsid`.

### The chunking protocol, and why it addresses sentences

`redact_llm.py` cuts the transcript into chunks of at most 5000 characters,
always closing on a sentence boundary, one sentence of overlap. The model
returns `{"sentence": N, "text": "<exact words>", "label": ...}` -- not
character offsets, not word indices:

- character offsets need the model to count characters, and an off-by-a-few
  silently redacts the wrong span;
- word indices need a running count over hundreds of tokens, and a wrong index
  is *undetectable* -- it names a real word, just not the intended one;
- a sentence number plus the exact words is checkable. The quote is searched
  for inside the sentence it was attributed to, and a quote that is not there
  is a hallucination: dropped and counted as `unmatched_quotes`.

Corpus-wide: 2814 chunks, **0 chunk failures**, 78 unmatched quotes (2.8%),
5694 words redacted. A failed chunk keeps its original words and is counted --
it must never look like a chunk with no PII in it.

`scripts/score_redaction.py --leaks-csv` writes one row per leaked identifier
(site, subject, session, system, identifier, both contexts) for inspection.
**That file is PII in the clear by construction** -- it lives at
`outputs/private/leaks.csv` on the cluster, never in the repo, never in an
artifact. 736 rows across 18 participants, of which 32 are `redacted=1` yet
still leaked: the system caught one occurrence of a name and missed another,
which is the worst outcome because the transcript looks de-identified.

Not yet done: the paper reports per-category F1 (names / dates / locations),
which needs labelled gold. The brace convention carries no label. If the
PSYCHS-Bench gold PHI annotations from the paper cover any of these 269
sessions, they would allow the per-category table to be reproduced exactly.

## Open questions / future (spec section "For Future")

- Fine-tuning CrisperWhisper 2.0: not yet investigated; the ct2 backend is
  inference-only, so training would go through the transformers backend or
  upstream nyrahealth tooling.

## Conventions

- No emoji in code or output. Follow existing module structure (thin
  wrappers per stage, plain dicts between stages, logging via module
  loggers).
