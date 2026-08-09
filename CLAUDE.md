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
over the complete 269-visit cohort: chirp3, verbatimize, ours (community-1),
ours_llm, baseline (pyannote 3.1), baseline_llm. Sweeps all reached 269/269.

Final numbers, after the longform ASR fix below (`outputs/results.json`):

    system         visits     WER    sWER     DER  DERconf  QTP-F1
    chirp3            269  0.8748  0.7121  0.6907   0.2071  0.8388
    verbatimize       269  0.8751  0.6747  0.6346   0.2108  0.8463
    ours              269  0.8374  0.8347  0.2242   0.2146  0.8790
    ours_llm          269  0.8385  0.8700  0.2254   0.2157  0.8780
    baseline          269  0.8670  0.5174  0.2054   0.1774  0.8711
    baseline_llm      269  0.8670  0.5231  0.2072   0.1793  0.8701

Readings that survive scrutiny:

- **pyannote 3.1 beats community-1 on speaker attribution**: sWER 0.517 vs
  0.835, confusion 0.177 vs 0.215. This held after the ASR fix (community-1
  had previously been judged on a transcript missing a third of its words),
  though the confusion gap narrowed from 0.331 to 0.215.
- **community-1 leads QTP-F1** (0.879, best in table) and marginally WER, so
  it is not strictly dominated.
- **The LLM review is worse on every metric for both trees, 269/269 visits.**
  Never better, not once. Its speaker flags also carry fabricated
  justifications (one flagged a turn for "contains '?' and starts with 'how
  often'" when it contained neither) and only 15 of 163 applied. Drop it.
- All six systems now sit at 1.33-1.36 hypothesis words per reference word.
  That agreement is the check that matters: ours and chirp3 match to three
  decimals while sharing no code.

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
semi-verbatim reference. Corrected, it is 0.8374 and the system is better, not
worse. Any metric computed before 2026-08-08 21:40 on the ours/ours_llm trees
is invalid.

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

## Open questions / future (spec section "For Future")

- Fine-tuning CrisperWhisper 2.0: not yet investigated; the ct2 backend is
  inference-only, so training would go through the transformers backend or
  upstream nyrahealth tooling.

## Conventions

- No emoji in code or output. Follow existing module structure (thin
  wrappers per stage, plain dicts between stages, logging via module
  loggers).
