# crisper-whisper-2: Verbatim Speaker-Attributed Transcription

This repository contains a pipeline that turns wav files into verbatim, word-level, speaker-attributed transcripts.

## Dependencies and Sources

- [CrisperWhisper 2.0](https://github.com/nyrahealth/CrisperWhisper) — verbatim ASR with word-level timestamps, CTranslate2 backend with speculative decoding
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) — speaker diarization ([pyannote-audio](https://github.com/pyannote/pyannote-audio) 4.x)
- [pyannoteAI diarization-ASR merge](https://docs.pyannote.ai/tutorials/diarization-asr-merge) — speaker assignment by maximum temporal overlap, applied per word

Managed with [uv](https://docs.astral.sh/uv/). Linux installs CUDA torch (cu128) and the CTranslate2 backend; macOS installs a CPU environment for development only.

## Installation

```bash
git clone https://github.com/bentdixon/crisper-pyannote.git
cd crisper-pyannote
uv sync
```

Authenticate with HuggingFace (community-1 is gated; accept its terms once on huggingface.co):

```bash
uv tool install huggingface_hub
hf auth login
```

## Usage

```bash
uv run transcribe-session session.wav --output-dir outputs
uv run transcribe-session /path/to/wavs/ --num-speakers 2 --output-dir outputs
```

Outputs per run, under `outputs/<stem>_<timestamp>/` (repeat runs of the same file never overwrite each other):

- `metadata.json` — run timestamp, package versions, transcription and diarization settings
- `transcript.json` — full word-level transcript with speaker labels
- `transcript.txt` — human-readable, timestamped speaker turns
- `diarization.json` — raw diarization segments
- `speakers/<SPEAKER>.json`, `speakers/<SPEAKER>.txt` — per-participant word-level JSON and readable transcript

Useful flags: `--diarization-mode overlap` (keep raw overlapping diarization segments; each segment claims its best-fitting contiguous word chunk by temporal IoU), `--fill-nearest` (attribute words in diarization gaps to the nearest speaker instead of UNKNOWN), `--diarization-model` (HuggingFace id or local pipeline config, e.g. a fine-tuned one), `--min-speakers`/`--max-speakers`, `--language`, `--no-speculative`, `--device-index`.

## Fine-tuning diarization

`finetune/` adapts the community-1 segmentation model to our corpus using
human-annotated transcripts paired with wavs. Transcripts may be timestamped
text (`S1 00:00:01.451 ...`), pipeline word-level JSON, or turn-level JSON.
Embedding, PLDA, and clustering stay stock; their hyperparameters are
re-optimized around the fine-tuned model.

**1. Prepare the training protocol.** Converts transcripts to time-aligned
speaker labels (RTTM/UEM), refining turn boundaries by forced alignment, and
splits 80/10/10 by participant:

```bash
uv run python finetune/prepare_data.py --transcripts <dir> --wavs <dir>
```

Inspect the printed report before continuing; a high alignment-fallback
count means noisy labels.

**2. Measure the baseline.** DER of stock community-1 on the development
set (add `--word-attribution` to also score word-level speaker attribution):

```bash
uv run python finetune/evaluate.py --subset development --output baseline.json
```

**3. Fine-tune the segmentation model.** Trains on one GPU with the trunk
frozen for the first epochs, early-stops on validation loss, and links the
best checkpoint to `finetune/checkpoints/best.ckpt`:

```bash
uv run python finetune/train_segmentation.py
```

**4. Re-optimize pipeline hyperparameters.** Tunes segmentation and VBx
clustering parameters against DER on the development set and writes a
ready-to-use pipeline config:

```bash
uv run python finetune/optimize_pipeline.py --iterations 100
```

**5. Evaluate on the held-out test set.** Compare against step 2; only
these numbers count:

```bash
uv run python finetune/evaluate.py --subset test --pipeline finetune/finetuned-config.yaml --output finetuned.json
```

**6. Use it.** Point the main pipeline at the fine-tuned config:

```bash
uv run transcribe-session audio.wav --diarization-model finetune/finetuned-config.yaml
```
