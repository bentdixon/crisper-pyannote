"""Optional LLM review of a finished transcript (unchanged from the original).

Produces suggestions only -- word_corrections, speaker_flags and a single
role_mapping_check -- and never edits the transcript or caption files. Kept in
its own module so the heavyweight causal-LM imports only happen when a review
is actually requested.
"""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


LLM_SYSTEM_PROMPT = """You are reviewing a clinical interview transcript for THREE NARROW,
specific kinds of possible issues. You must be conservative — when in doubt, do NOT flag.

===========================================================
CATEGORY 1: WORD_CORRECTIONS
===========================================================

ONLY flag a SINGLE WORD (or at most a 2-word phrase) that was likely
MISHEARD by the speech-to-text model — meaning a different word that
SOUNDS similar was probably said instead. This is a narrow phonetic
substitution, nothing else.

VALID example:
  Transcript: "Test, test, test... it's so weird being on camera"
  Correction: original="task", suggested="test"
  Reasoning: "task" and "test" are phonetically similar (same syllable
  count, similar consonant sounds) and "test, test, test" while checking
  a camera/mic makes sense; "task" does not fit that context.

INVALID — DO NOT DO THIS:
  Transcript: "Not in the last seven days in my life, yes."
  This is NOT a word correction, even if a phrase sounds redundant,
  hesitant, or grammatically unusual. Real spoken language is full of
  self-corrections, hesitations, and restarts — these are NOT
  transcription errors and must NEVER be flagged or removed. If you
  cannot point to a SPECIFIC single word that sounds like a plausible
  mishearing of a DIFFERENT specific word, do not flag anything in that
  sentence at all.

Rules for word_corrections:
- "original" must be an EXACT single word (or 2-word phrase) copied
  verbatim from the transcript — never a full sentence.
- "suggested" must be phonetically similar to "original" — if you cannot
  explain the phonetic similarity, do not flag it.
- NEVER flag: filler words ([UM], [UH]), repeated words, hesitations,
  self-corrections, restarts, or grammatically atypical but plausible
  spoken phrasing. These are normal speech, not errors.
- NEVER delete, shorten, or rewrite a phrase. Only substitute one
  specific word for another specific word.

===========================================================
CATEGORY 2: SPEAKER_FLAGS
===========================================================

This is a clinical interview between an INTERVIEWER (asks structured
questions) and a PARTICIPANT (describes personal experience). Only flag
a turn if you can point to SPECIFIC, CONCRETE textual evidence that it's
attributed to the wrong role — not a general impression.

Valid evidence includes ONLY:
- The turn contains a question mark, AND is attributed to PARTICIPANT.
- The turn starts with a clear interrogative word/phrase ("how often",
  "have you", "in the past week, did you...", "what", "when"), AND is
  attributed to PARTICIPANT.
- The turn is a first-person description of a symptom/experience
  ("I hear voices", "I feel like..."), AND is attributed to INTERVIEWER.

INVALID — DO NOT DO THIS:
  Flagging a turn just because it "sounds more formal" or "sounds more
  like a question" without an actual question mark or interrogative
  opener present in the text. Vague impressions are not evidence.

Rules for speaker_flags:
- "reasoning" MUST quote the specific word/punctuation that is your
  evidence (e.g. "contains '?'" or "starts with 'how often'").
- If you cannot quote specific textual evidence, do not flag the turn.

===========================================================
CATEGORY 3: ROLE_MAPPING_CHECK
===========================================================

The labels INTERVIEWER and PARTICIPANT were already assigned to each
speaker throughout this transcript by a separate process, based on which
speaker asks more questions overall. This is a single, whole-transcript
sanity check: does that overall assignment look correct?

- If INTERVIEWER clearly asks most of the questions and PARTICIPANT
  clearly describes experience/answers — the assignment looks correct.
- If the pattern looks backwards (PARTICIPANT is asking most of the
  questions, INTERVIEWER is mostly describing personal experience) —
  flag it.

This is ONE overall judgment for the whole transcript, not per-turn.

===========================================================
OUTPUT FORMAT
===========================================================

Respond with ONLY valid JSON, no other text:

{
  "word_corrections": [
    {"original": "task", "suggested": "test", "context": "...task, task, task...", "reasoning": "phonetically similar; fits camera-check context"}
  ],
  "speaker_flags": [
    {"turn_text": "...", "current_speaker": "PARTICIPANT", "reasoning": "contains '?' and starts with 'how often'"}
  ],
  "role_mapping_check": {
    "looks_correct": true,
    "reasoning": "INTERVIEWER asks most questions ('how often...', 'in the past week...'); PARTICIPANT describes experience"
  }
}

If nothing meets the narrow criteria for word_corrections or speaker_flags,
return empty lists for those categories. Empty lists are the CORRECT and
EXPECTED outcome when no qualifying issue is present — do not force
suggestions to appear useful. role_mapping_check must always be filled in.
"""


def load_llm(model_id=LLM_MODEL_ID, device=None):
    """Load the review model. `device` overrides the default cuda:0 so the
    review can run on a different GPU than an in-flight transcription job."""
    device = device or DEVICE
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=TORCH_DTYPE, device_map=device,
    )
    return model, tokenizer


def _strip_markdown_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.lstrip("`")
        text = text[4:] if text.lower().startswith("json") else text
    text = text.rstrip("`").strip()
    return text


def _validate_llm_output(result, transcript_text):
    valid_corrections = []
    for correction in result.get("word_corrections", []):
        original = correction.get("original", "")
        word_count = len(original.split())
        if not original or word_count > 2 or original not in transcript_text:
            continue
        valid_corrections.append(correction)

    valid_flags = []
    for flag in result.get("speaker_flags", []):
        reasoning = flag.get("reasoning", "")
        has_quote = "'" in reasoning or '"' in reasoning
        if not has_quote:
            continue
        valid_flags.append(flag)

    return {
        "word_corrections": valid_corrections,
        "speaker_flags": valid_flags,
        "role_mapping_check": result.get("role_mapping_check"),
    }


def run_llm_verification(model, tokenizer, transcript_text):
    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": f"Transcript:\n\n{transcript_text}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    cleaned_response = _strip_markdown_fences(response)

    try:
        raw_result = json.loads(cleaned_response)
    except json.JSONDecodeError:
        return {
            "word_corrections": [], "speaker_flags": [], "role_mapping_check": None,
            "raw_response": response,
        }

    return _validate_llm_output(raw_result, transcript_text)
