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

Useful flags: `--fill-nearest` (attribute words in diarization gaps to the nearest speaker instead of UNKNOWN), `--min-speakers`/`--max-speakers`, `--language`, `--no-speculative`, `--device-index`.
