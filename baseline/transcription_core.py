"""Another team's transcription + diarization core, ported to CrisperWhisper 2.0.

This is their pipeline, kept structurally intact so a comparison against the
Chirp-3 route is a fair one. Only the ASR stage has been swapped:

    was   nyrahealth/CrisperWhisper (v1) through a transformers
          AutoModelForSpeechSeq2Seq + "automatic-speech-recognition" pipeline
    now   CrisperWhisper 2.0 (nyralabs/CrisperWhisper2.0_large) through the
          CTranslate2 backend, the same model the rest of this repo uses

Why the swap is a favour rather than a liberty:
  - v1 is deprecated upstream and warns on load; 2.0 has better verbatim
    accuracy and 3-5x faster inference.
  - `adjust_pauses` existed to repair a known v1 timestamp quirk. 2.0's
    word timestamps come from proper cross-attention alignment, so the
    correction is not needed and has been dropped rather than reapplied
    to timings that are already correct.
  - Segments here are capped at MAX_SEGMENT_SECONDS (25 s), comfortably
    inside 2.0's 30 s short-audio path, so no longform strategy is needed.

Everything else is theirs: stereo channel-dominance diarization gated on a
real loudness-separation check, silero VAD, the pyannote fallback for mono
or fake-stereo files, INTERVIEWER/PARTICIPANT role assignment, VTT/SRT
export, and the optional LLM review.

Pipeline for a single file:
  1. Check number of audio channels
  2. If stereo (2 channels): verify the channels carry REAL separate
     signal (not just a stereo container with near-identical channels,
     e.g. Zoom exports) -- if real, channel-dominance assigns speakers;
     if not, falls back to pyannote on a proper mono downmix
  3. If mono (1 channel): pyannote diarization assigns speakers
  4. Transcribe with CrisperWhisper 2.0 (word-level timestamps)
  5. Assign INTERVIEWER/PARTICIPANT roles (deterministic heuristic)
  6. Save {base_name}_transcript.txt, {base_name}.vtt, {base_name}.srt
  7. OPTIONAL: LLM review (word_corrections + speaker_flags +
     role_mapping_check) if an LLM model/tokenizer is provided
"""

import json
import math
import os

import numpy as np
import torch
from crisper_pipeline.cuda_preload import preload_torchcodec_libs

# The PyPI torchcodec wheel links CUDA 13 libs that our cu128 torch stack does
# not ship; dlopen them first or importing pyannote raises on libnvrtc.so.13.
# Must happen before the pyannote import below (see CLAUDE.md).
preload_torchcodec_libs()

# pyannote 3.x checkpoints predate PyTorch 2.6's stricter torch.load default.
# We trust the official pyannote checkpoints from Hugging Face, so restore the
# old, more permissive loading behaviour for this trusted source.
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from crisperwhisper.audio import SAMPLE_RATE  # noqa: E402
from pyannote.audio import Pipeline as DiarizationPipeline  # noqa: E402
from pydub import AudioSegment  # noqa: E402

# ============================================================
# CONFIG
# ============================================================

NUM_SPEAKERS = 2
MAX_SEGMENT_SECONDS = 25
VAD_THRESHOLD = 0.3
SEGMENT_PAD_SECONDS = 0.3

MAX_TURN_PAUSE_SECONDS = 2.0

CAPTION_MAX_PAUSE_SECONDS = 1.5
CAPTION_MAX_CUE_DURATION = 6.0
CAPTION_MAX_CHARS_PER_CUE = 84

MIN_AVG_DB_SEPARATION = 3.0  # below this, treat stereo channels as not really separated

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ============================================================
# STEP 1: CHECK CHANNELS
# ============================================================

def load_audio_and_check_channels(audio_path):
    stereo_or_mono = AudioSegment.from_file(audio_path).set_frame_rate(SAMPLE_RATE)
    channels = stereo_or_mono.split_to_mono()
    return channels


# ============================================================
# STEP 2: CHANNEL-DOMINANCE DIARIZATION (stereo path)
# ============================================================

def get_vad_ranges(audio_segment, tmp_wav_path, threshold=VAD_THRESHOLD):
    audio_segment.export(tmp_wav_path, format="wav")
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad",
        force_reload=False, trust_repo=True,
    )
    get_speech_timestamps, _, read_audio, _, _ = vad_utils
    wav = read_audio(tmp_wav_path, sampling_rate=SAMPLE_RATE)
    speech_timestamps = get_speech_timestamps(
        wav, vad_model, sampling_rate=SAMPLE_RATE, threshold=threshold
    )
    os.remove(tmp_wav_path)
    return [(ts["start"] / SAMPLE_RATE, ts["end"] / SAMPLE_RATE) for ts in speech_timestamps]


def has_real_channel_separation(left_ranges, right_ranges, left_audio, right_audio):
    """
    ffprobe reporting 2 channels only means the CONTAINER is stereo -- it
    says nothing about whether the channels carry meaningfully different
    content. Some recording setups (e.g. Zoom's stereo export) don't
    isolate speakers per channel at all, producing near-identical
    left/right signal. Trusting channel-dominance on such a file means
    picking a "winner" based on measurement noise, not real speaker
    identity. This checks actual loudness divergence at real speech
    moments before deciding channel-dominance is trustworthy.
    """
    all_ranges = sorted(left_ranges + right_ranges, key=lambda r: r[0])
    if not all_ranges:
        return False

    diffs = []
    for start, end in all_ranges:
        start_ms, end_ms = int(start * 1000), int(end * 1000)
        left_db = left_audio[start_ms:end_ms].dBFS
        right_db = right_audio[start_ms:end_ms].dBFS
        if left_db == float("-inf") or right_db == float("-inf"):
            continue
        diffs.append(abs(left_db - right_db))

    if not diffs:
        return False

    avg_diff = sum(diffs) / len(diffs)
    print(f"Channel separation check: average |LEFT-RIGHT| dB across "
          f"{len(diffs)} speech windows = {avg_diff:.2f} dB "
          f"(threshold: {MIN_AVG_DB_SEPARATION} dB)")

    return avg_diff >= MIN_AVG_DB_SEPARATION


def build_dominant_timeline(left_ranges, right_ranges, left_audio, right_audio):
    points = sorted(set(s for s, e in left_ranges + right_ranges) |
                    set(e for s, e in left_ranges + right_ranges))

    def is_active(ranges, t_mid):
        return any(s <= t_mid <= e for s, e in ranges)

    sub_intervals = []
    for i in range(len(points) - 1):
        seg_start, seg_end = points[i], points[i + 1]
        if seg_end - seg_start < 1e-6:
            continue
        mid = (seg_start + seg_end) / 2
        left_active = is_active(left_ranges, mid)
        right_active = is_active(right_ranges, mid)

        if not left_active and not right_active:
            continue
        elif left_active and not right_active:
            speaker = "LEFT"
        elif right_active and not left_active:
            speaker = "RIGHT"
        else:
            start_ms, end_ms = int(seg_start * 1000), int(seg_end * 1000)
            left_db = left_audio[start_ms:end_ms].dBFS
            right_db = right_audio[start_ms:end_ms].dBFS
            speaker = "LEFT" if left_db >= right_db else "RIGHT"

        sub_intervals.append((seg_start, seg_end, speaker))

    if not sub_intervals:
        raise RuntimeError("No speech detected on either channel.")

    return _merge_and_cap(sub_intervals)


def _merge_and_cap(labeled_ranges, max_segment_seconds=MAX_SEGMENT_SECONDS, max_gap=2.0):
    merged = []
    cur_start, cur_end, cur_speaker = labeled_ranges[0]
    for s, e, sp in labeled_ranges[1:]:
        gap = s - cur_end
        span = e - cur_start
        if sp == cur_speaker and span <= max_segment_seconds and gap <= max_gap:
            cur_end = e
        else:
            merged.append((cur_start, cur_end, cur_speaker))
            cur_start, cur_end, cur_speaker = s, e, sp
    merged.append((cur_start, cur_end, cur_speaker))

    final = []
    for start, end, speaker in merged:
        span = end - start
        if span <= max_segment_seconds:
            final.append((start, end, speaker))
        else:
            n_pieces = math.ceil(span / max_segment_seconds)
            piece_len = span / n_pieces
            for i in range(n_pieces):
                final.append((start + i * piece_len, min(start + (i + 1) * piece_len, end), speaker))
    return final


# ============================================================
# STEP 3: PYANNOTE DIARIZATION (mono / fallback path)
# ============================================================

def run_pyannote_diarization(audio_path, num_speakers=NUM_SPEAKERS, token=None):
    token = token or os.environ.get("HF_TOKEN")

    # pyannote.audio 4.x renamed use_auth_token -> token, and returns a
    # DiarizeOutput for community-1 style pipelines; the 3.1 config still
    # yields a plain Annotation.
    diarization_pipeline = DiarizationPipeline.from_pretrained(
        DIARIZATION_MODEL, token=token,
    )
    if diarization_pipeline is None:
        raise RuntimeError(
            f"Could not load {DIARIZATION_MODEL}. Accept its terms on "
            "huggingface.co and make sure a token is available."
        )
    diarization_pipeline.to(torch.device(DEVICE))
    if hasattr(diarization_pipeline, "segmentation_batch_size"):
        diarization_pipeline.segmentation_batch_size = 32

    result = diarization_pipeline(audio_path, num_speakers=num_speakers)
    annotation = getattr(result, "speaker_diarization", result)
    speaker_segments = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    if not speaker_segments:
        raise RuntimeError("Diarization returned no speech segments.")
    return _merge_and_cap(speaker_segments, max_gap=1.0)


# ============================================================
# STEP 4: TRANSCRIPTION (CrisperWhisper 2.0)
# ============================================================

def load_crisperwhisper(model_name="large", draft_model="turbo",
                        compute_type="float16", device_index=0):
    """Load CrisperWhisper 2.0 on the CTranslate2 backend.

    Replaces the v1 transformers ASR pipeline. Returns the model itself;
    `transcribe_segments` calls it directly rather than through a
    transformers pipeline object.
    """
    from crisper_pipeline import asr

    return asr.load_model(
        model_name,
        draft_model=draft_model,
        compute_type=compute_type if torch.cuda.is_available() else "float32",
        device="auto",
        device_index=device_index,
    )


def _segment_to_array(audio_segment):
    """pydub AudioSegment -> float32 numpy at SAMPLE_RATE, as CT2 expects."""
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels)).mean(axis=1)
    scale = float(1 << (8 * audio_segment.sample_width - 1))
    return (samples.astype(np.float32) / scale)


def transcribe_segments(asr_model, source_channels, segments, work_dir,
                        language="en", speculative_decoding=True):
    """Transcribe each diarization segment from that speaker's own audio.

    Word timestamps come back relative to the padded segment and are shifted
    onto the session timeline, exactly as in the original. Audio is passed as
    an in-memory array instead of being written to a temporary wav per
    segment, which removes thousands of disk round-trips per interview.
    """
    os.makedirs(work_dir, exist_ok=True)
    duration_sec = min(len(a) for a in source_channels.values()) / 1000.0

    all_chunks = []
    for start_sec, end_sec, speaker in segments:
        padded_start = max(0.0, start_sec - SEGMENT_PAD_SECONDS)
        padded_end = min(duration_sec, end_sec + SEGMENT_PAD_SECONDS)
        start_ms, end_ms = int(padded_start * 1000), int(padded_end * 1000)

        clip = source_channels[speaker][start_ms:end_ms]
        samples = _segment_to_array(clip)
        if samples.size == 0:
            continue

        result = asr_model.transcribe(
            samples,
            language=language,
            mode="verbatim",
            sr=SAMPLE_RATE,
            word_timestamps=True,
            speculative_decoding=speculative_decoding,
        )

        for word in (result.words or []):
            w_start = float(word.start) if word.start is not None else None
            w_end = float(word.end) if word.end is not None else None
            all_chunks.append({
                # Leading space matches the transformers pipeline convention
                # the downstream "".join(...) calls were written against.
                "text": " " + word.word.strip(),
                "timestamp": (
                    (w_start + padded_start) if w_start is not None else None,
                    (w_end + padded_start) if w_end is not None else None,
                ),
                "speaker": speaker,
            })

    return all_chunks


def group_into_turns(chunks, max_pause_seconds=MAX_TURN_PAUSE_SECONDS):
    turns = []
    current = None
    for c in chunks:
        start, end = c["timestamp"]
        speaker = c["speaker"]

        pause = None
        if current is not None and current["end"] is not None and start is not None:
            pause = start - current["end"]

        starts_new_turn = (
            current is None
            or current["speaker"] != speaker
            or (pause is not None and pause > max_pause_seconds)
        )

        if starts_new_turn:
            if current is not None:
                turns.append(current)
            current = {"speaker": speaker, "start": start, "end": end, "words": [c["text"]]}
        else:
            current["words"].append(c["text"])
            if end is not None:
                current["end"] = end
    if current is not None:
        turns.append(current)
    return turns


# ============================================================
# CAPTION EXPORT (VTT / SRT)
# ============================================================

def build_caption_cues(chunks, max_cue_duration=CAPTION_MAX_CUE_DURATION,
                       max_chars_per_cue=CAPTION_MAX_CHARS_PER_CUE,
                       max_pause_seconds=CAPTION_MAX_PAUSE_SECONDS):
    cues = []
    current = None

    def _finalize():
        if current is not None:
            text = "".join(current["words"]).strip()
            if text:
                cues.append({
                    "speaker": current["speaker"],
                    "start": current["start"] if current["start"] is not None else 0.0,
                    "end": current["end"] if current["end"] is not None else current["start"],
                    "text": text,
                })

    for c in chunks:
        start, end = c["timestamp"]
        speaker = c["speaker"]
        word = c["text"]

        if current is None:
            current = {"speaker": speaker, "start": start, "end": end, "words": [word]}
            continue

        pause = None
        if current["end"] is not None and start is not None:
            pause = start - current["end"]

        prospective_duration = None
        if current["start"] is not None and end is not None:
            prospective_duration = end - current["start"]

        prospective_len = len("".join(current["words"] + [word]).strip())

        must_break = (
            current["speaker"] != speaker
            or (pause is not None and pause > max_pause_seconds)
            or (prospective_duration is not None and prospective_duration > max_cue_duration)
            or (prospective_len > max_chars_per_cue)
        )

        if must_break:
            _finalize()
            current = {"speaker": speaker, "start": start, "end": end, "words": [word]}
        else:
            current["words"].append(word)
            if end is not None:
                current["end"] = end

    _finalize()
    return cues


def _format_vtt_timestamp(seconds):
    seconds = seconds if seconds is not None else 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _format_srt_timestamp(seconds):
    seconds = seconds if seconds is not None else 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_vtt(cues, output_path):
    lines = ["WEBVTT", ""]
    for cue in cues:
        start_str = _format_vtt_timestamp(cue["start"])
        end_str = _format_vtt_timestamp(cue["end"])
        lines.append(f"{start_str} --> {end_str}")
        lines.append(f"<v {cue['speaker']}>{cue['text']}")
        lines.append("")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def write_srt(cues, output_path):
    lines = []
    for i, cue in enumerate(cues, start=1):
        start_str = _format_srt_timestamp(cue["start"])
        end_str = _format_srt_timestamp(cue["end"])
        lines.append(str(i))
        lines.append(f"{start_str} --> {end_str}")
        lines.append(f"{cue['speaker']}: {cue['text']}")
        lines.append("")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ============================================================
# STEP 5: ASSIGN ROLES (INTERVIEWER / PARTICIPANT)
# ============================================================

QUESTION_OPENERS = (
    "how", "what", "when", "where", "why", "who", "which",
    "do you", "does", "did you", "have you", "has", "are you",
    "is there", "can you", "could you", "would you", "in the past",
)


def heuristic_assign_roles(turns):
    """Whichever channel has a higher rate of question-like turns is almost
    certainly the INTERVIEWER -- clinical interviews are structurally
    question-heavy on one side. Fully deterministic and auditable.

    This dataset only ever has 1 speaker (participant only) or 2 speakers
    (participant + interviewer) -- never more. Pyannote occasionally only
    detects ONE speaker (e.g. a short recording, or one person's audio
    dominating so heavily the other never gets its own cluster). Rather
    than crashing the whole file over this, we label the single detected
    speaker as PARTICIPANT -- the more common case when only one voice
    comes through clearly is a quiet/absent interviewer channel, not a
    quiet participant. This is a reasonable default, not a confident
    diarization claim; worth spot-checking these specific files manually
    given the label is a guess."""
    counts = {}
    for t in turns:
        speaker = t["speaker"]
        text = "".join(t["words"]).strip().lower()
        counts.setdefault(speaker, {"question_turns": 0, "total_turns": 0})
        counts[speaker]["total_turns"] += 1
        is_question = text.endswith("?") or any(text.startswith(o) for o in QUESTION_OPENERS)
        if is_question:
            counts[speaker]["question_turns"] += 1

    if len(counts) == 1:
        only_speaker = next(iter(counts))
        print(f"WARNING: only 1 speaker detected ({only_speaker}) -- "
              f"labeling as PARTICIPANT by default. Review this file manually.")
        return {only_speaker: "PARTICIPANT"}, counts

    rates = {ch: (c["question_turns"] / c["total_turns"] if c["total_turns"] else 0)
             for ch, c in counts.items()}
    ranked = sorted(rates.items(), key=lambda x: x[1], reverse=True)
    interviewer_channel, participant_channel = ranked[0][0], ranked[1][0]

    mapping = {interviewer_channel: "INTERVIEWER", participant_channel: "PARTICIPANT"}
    return mapping, counts


def apply_role_mapping(turns, chunks, mapping):
    for t in turns:
        t["speaker"] = mapping.get(t["speaker"], t["speaker"])
    for c in chunks:
        c["speaker"] = mapping.get(c["speaker"], c["speaker"])
    return turns, chunks


# ============================================================
# MAIN ENTRY POINT: process ONE interview file
# ============================================================

def process_interview(audio_path, output_dir, asr_model,
                      work_dir="/tmp/transcription_core_work",
                      file_prefix="", llm_model=None, llm_tokenizer=None,
                      language="en", token=None, speculative_decoding=True):
    """
    Runs the full pipeline on ONE audio file and writes outputs to
    output_dir:
      {file_prefix}_transcript.txt
      {file_prefix}_captions.vtt
      {file_prefix}_captions.srt
      {file_prefix}_llm_suggestion.json   (only if llm_model/llm_tokenizer given)

    asr_model: an already-loaded CrisperWhisper 2.0 model (from
    load_crisperwhisper()) -- load ONCE in the runner and pass it in here.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    channels = load_audio_and_check_channels(audio_path)

    if len(channels) == 2:
        left_audio, right_audio = channels[0], channels[1]

        left_ranges = get_vad_ranges(left_audio, os.path.join(work_dir, "_vad_left.wav"))
        right_ranges = get_vad_ranges(right_audio, os.path.join(work_dir, "_vad_right.wav"))

        if has_real_channel_separation(left_ranges, right_ranges, left_audio, right_audio):
            segments = build_dominant_timeline(left_ranges, right_ranges, left_audio, right_audio)
            source_channels = {"LEFT": left_audio, "RIGHT": right_audio}
            diarization_method = "channel_dominance"
        else:
            downmixed_audio = (
                AudioSegment.from_file(audio_path).set_frame_rate(SAMPLE_RATE).set_channels(1)
            )
            segments = run_pyannote_diarization(audio_path, NUM_SPEAKERS, token=token)
            source_channels = {spk: downmixed_audio for _, _, spk in segments}
            diarization_method = "pyannote_fallback_no_real_stereo_separation"

    elif len(channels) == 1:
        mono_audio = channels[0]
        segments = run_pyannote_diarization(audio_path, NUM_SPEAKERS, token=token)
        source_channels = {spk: mono_audio for _, _, spk in segments}
        diarization_method = "pyannote"

    else:
        raise RuntimeError(f"Unexpected channel count: {len(channels)}")

    chunks = transcribe_segments(
        asr_model, source_channels, segments, work_dir,
        language=language, speculative_decoding=speculative_decoding,
    )
    turns = group_into_turns(chunks)

    role_mapping, role_evidence = heuristic_assign_roles(turns)
    turns, chunks = apply_role_mapping(turns, chunks, role_mapping)

    transcript_lines = []
    for t in turns:
        start_str = f"{t['start']:.2f}" if t["start"] is not None else "?"
        end_str = f"{t['end']:.2f}" if t["end"] is not None else "?"
        text = "".join(t["words"]).strip()
        transcript_lines.append(f"[{start_str} - {end_str}] {t['speaker']}: {text}")
    transcript_text = "\n".join(transcript_lines)

    with open(os.path.join(output_dir, f"{file_prefix}_transcript.txt"), "w") as f:
        f.write(transcript_text)

    caption_cues = build_caption_cues(chunks)
    write_vtt(caption_cues, os.path.join(output_dir, f"{file_prefix}_captions.vtt"))
    write_srt(caption_cues, os.path.join(output_dir, f"{file_prefix}_captions.srt"))

    # Word-level JSON is not part of the original pipeline, but the Chirp-3
    # comparison scores per-word timing, so keep the raw words too.
    with open(os.path.join(output_dir, f"{file_prefix}_words.json"), "w") as f:
        json.dump(
            [
                {
                    "word": c["text"].strip(),
                    "start": c["timestamp"][0],
                    "end": c["timestamp"][1],
                    "speaker": c["speaker"],
                }
                for c in chunks
            ],
            f, indent=2,
        )

    llm_result = None
    if llm_model is not None and llm_tokenizer is not None:
        from llm_review import run_llm_verification

        llm_result = run_llm_verification(llm_model, llm_tokenizer, transcript_text)
        with open(os.path.join(output_dir, f"{file_prefix}_llm_suggestion.json"), "w") as f:
            json.dump(llm_result, f, indent=2)

    result = {
        "diarization_method": diarization_method,
        "role_mapping": role_mapping,
        "role_evidence": role_evidence,
        "num_words": len(chunks),
        "num_turns": len(turns),
        "num_caption_cues": len(caption_cues),
        "num_segments": len(segments),
    }
    if llm_result is not None:
        result["num_word_corrections"] = len(llm_result.get("word_corrections", []))
        result["num_speaker_flags"] = len(llm_result.get("speaker_flags", []))
        result["role_mapping_check"] = llm_result.get("role_mapping_check")

    return result
