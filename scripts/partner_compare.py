#!/usr/bin/env python3
import sys
import re
from difflib import SequenceMatcher


FILLER_RE = re.compile(
    r"^(u+h+|u+m+|e+r+m?|h+m+|o+h+|a+h+|m{2,}|uh-huh|mm-hm|mhm|huh|hmm?)$",
    re.IGNORECASE,
)


def tokenize(text, normalize=False):
    text = re.sub(r"\[[^\]]*\](\S*)", r"\1", text)
    text = re.sub(r"\{[^\}]*\}(\S*)", r"\1", text)
    if normalize:
        text = re.sub(r"[.,?!\-—;:\"'()]", "", text)
        text = text.lower()
    return text.split()


def replace_fillers(tokens):
    return ["[filler]" if FILLER_RE.match(t) else t for t in tokens]


def analyze(ref, hyp):
    errors = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        if tag == "replace":
            errors += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            errors += i2 - i1
        elif tag == "insert":
            errors += j2 - j1
    wer = errors / len(ref) * 100 if ref else 0.0
    return wer


def compare_transcripts(path1, path2):
    with open(path1) as f1, open(path2) as f2:
        text1 = f1.read()
        text2 = f2.read()

    print(f"\nGround truth:  {path1}")
    print(f"AI transcript: {path2}\n")

    ref_raw  = tokenize(text1)
    hyp_raw  = tokenize(text2)
    ref_norm = tokenize(text1, normalize=True)
    hyp_norm = tokenize(text2, normalize=True)
    ref_fill = replace_fillers(ref_norm)
    hyp_fill = replace_fillers(hyp_norm)

    wer_raw  = analyze(ref_raw,  hyp_raw)
    wer_norm = analyze(ref_norm, hyp_norm)
    wer_fill = analyze(ref_fill, hyp_fill)

    print(f"  {'':45}  WER")
    print(f"  {'-'*45}  {'-'*6}")
    print(f"  {'1. Raw':45}  {wer_raw:.2f}%")
    print(f"  {'2. Normalized (no punctuation / capitalization)':45}  {wer_norm:.2f}%")
    print(f"  {'3. Filler-normalized':45}  {wer_fill:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compareFiles.py <ground_truth> <ai_transcript>")
        sys.exit(1)
    compare_transcripts(sys.argv[1], sys.argv[2])
