#!/usr/bin/env python3
"""
Word Error Rate (WER) Assessment Script for NVIDIA TXT Transcripts

Compares target TXT transcripts against ground truth transcripts using the jiwer library.
Excludes words in square brackets and special characters/punctuation from WER calculations.
"""

import os
import re
import csv
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import jiwer


def preprocess_transcript(text: str) -> str:
    """
    Preprocess transcript text by removing:
    1. All words within square brackets []
    2. All special characters and punctuation

    Args:
        text: Raw transcript text

    Returns:
        Preprocessed text with only alphanumeric characters and spaces
    """
    # Remove all content within square brackets (including the brackets)
    text = re.sub(r'\[.*?\]', '', text)

    # Remove all special characters and punctuation, keep only alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Normalize whitespace (replace multiple spaces with single space)
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Convert to lowercase for case-insensitive comparison
    text = text.lower()

    return text


def load_transcript_from_txt(file_path: str) -> str:
    """
    Load transcript text from TXT file (NVIDIA format).
    Removes speaker tags (S1:, S2:, etc.) and timestamps (HH:MM:SS.mmm).

    Args:
        file_path: Path to text transcript file

    Returns:
        Transcript text with speaker tags and timestamps removed

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    transcript_text = []

    # Pattern to match speaker tag and timestamp: "S1: 00:02:18.449 "
    pattern = re.compile(r'^S\d+:\s+\d{2}:\d{2}:\d{2}\.\d{3}\s+')

    for line in lines:
        # Skip empty lines
        if line.strip() == '':
            continue

        # Remove speaker tag and timestamp if present
        cleaned_line = pattern.sub('', line)

        # Add the cleaned text if there's content
        if cleaned_line.strip():
            transcript_text.append(cleaned_line.strip())

    return ' '.join(transcript_text)


def load_ground_truth_from_txt(file_path: str) -> str:
    """
    Load ground truth transcript from text file.
    Removes speaker tags (S1:, S2:, etc.) and timestamps (HH:MM:SS.mmm).

    Args:
        file_path: Path to text transcript file

    Returns:
        Transcript text with speaker tags and timestamps removed

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    transcript_text = []

    # Pattern to match speaker tag and timestamp: "S1: 00:02:18.449 "
    pattern = re.compile(r'^S\d+:\s+\d{2}:\d{2}:\d{2}\.\d{3}\s+')

    for line in lines:
        # Skip empty lines
        if line.strip() == '':
            continue

        # Remove speaker tag and timestamp if present
        cleaned_line = pattern.sub('', line)

        # Add the cleaned text if there's content
        if cleaned_line.strip():
            transcript_text.append(cleaned_line.strip())

    return ' '.join(transcript_text)


def parse_speaker_transcript_txt(file_path: str) -> List[Tuple[str, str]]:
    """
    Parse TXT transcript and return list of (speaker, word) tuples.

    Args:
        file_path: Path to text transcript file

    Returns:
        List of (speaker_tag, word) tuples preserving order
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    speaker_words = []
    current_speaker = None

    # Pattern to match speaker tag and timestamp: "S1: 00:02:18.449 "
    pattern = re.compile(r'^(S\d+):\s+\d{2}:\d{2}:\d{2}\.\d{3}\s+')

    for line in lines:
        # Skip empty lines
        if line.strip() == '':
            continue

        # Check for speaker tag
        match = pattern.match(line)
        if match:
            current_speaker = match.group(1)
            # Remove speaker tag and timestamp
            text = pattern.sub('', line).strip()
        else:
            # Continuation of previous speaker
            text = line.strip()

        if text and current_speaker:
            # Split into words and create (speaker, word) pairs
            words = text.split()
            for word in words:
                speaker_words.append((current_speaker, word))

    return speaker_words


def preprocess_word(word: str) -> str:
    """
    Preprocess a single word the same way as the full transcript.

    Args:
        word: Raw word

    Returns:
        Preprocessed word
    """
    # Remove content within square brackets
    word = re.sub(r'\[.*?\]', '', word)
    # Remove special characters and punctuation
    word = re.sub(r'[^a-zA-Z0-9\s]', '', word)
    # Convert to lowercase
    word = word.lower().strip()
    return word


def calculate_der_with_mapping(gt_processed: List[Tuple[str, str]],
                                hyp_processed: List[Tuple[str, str]],
                                speaker_mapping: Dict[str, str],
                                output: object) -> Tuple[int, int]:
    """
    Calculate speaker errors with a given speaker mapping.

    Args:
        gt_processed: Ground truth (speaker, word) tuples
        hyp_processed: Hypothesis (speaker, word) tuples
        speaker_mapping: Mapping from hypothesis speakers to ground truth speakers
        output: jiwer alignment output

    Returns:
        Tuple of (speaker_errors, total_correct_words)
    """
    speaker_errors = 0
    total_correct_words = 0
    gt_idx = 0
    hyp_idx = 0

    for op in output.alignments[0]:
        if op.type == 'equal':  # Words match (hit)
            if gt_idx < len(gt_processed) and hyp_idx < len(hyp_processed):
                gt_speaker = gt_processed[gt_idx][0]
                hyp_speaker = hyp_processed[hyp_idx][0]

                # Apply speaker mapping
                mapped_hyp_speaker = speaker_mapping.get(hyp_speaker, hyp_speaker)

                if gt_speaker != mapped_hyp_speaker:
                    speaker_errors += 1

                total_correct_words += 1

            gt_idx += 1
            hyp_idx += 1
        elif op.type == 'delete':
            gt_idx += 1
        elif op.type == 'insert':
            hyp_idx += 1
        elif op.type == 'substitute':
            gt_idx += 1
            hyp_idx += 1

    return speaker_errors, total_correct_words


def calculate_der(gt_speaker_words: List[Tuple[str, str]],
                  hyp_speaker_words: List[Tuple[str, str]],
                  wer_output: object) -> Dict[str, float]:
    """
    Calculate Diarization Error Rate (DER) by comparing speaker labels
    for correctly matched words. Automatically tries different speaker mappings
    to find the best alignment.

    Args:
        gt_speaker_words: Ground truth list of (speaker, word) tuples
        hyp_speaker_words: Hypothesis list of (speaker, word) tuples
        wer_output: WER alignment output from jiwer

    Returns:
        Dictionary with DER metrics and the best speaker mapping
    """
    # Build full text and preprocess the same way WER does
    gt_text = ' '.join(word for _, word in gt_speaker_words)
    hyp_text = ' '.join(word for _, word in hyp_speaker_words)

    gt_clean = preprocess_transcript(gt_text)
    hyp_clean = preprocess_transcript(hyp_text)

    # Split into words the same way jiwer does
    gt_words_split = gt_clean.split()
    hyp_words_split = hyp_clean.split()

    # Now map back to speakers by tracking which original word each preprocessed word came from
    # This is complex, so let's use a simpler approach: for each word in the split,
    # find the corresponding speaker from the original list

    gt_with_speakers = []
    hyp_with_speakers = []

    # For ground truth: map preprocessed words back to speakers
    gt_word_idx = 0
    for spk, orig_word in gt_speaker_words:
        # Preprocess this word
        cleaned = preprocess_word(orig_word)
        if cleaned:  # If word survives preprocessing
            # This word contributes to the cleaned text
            # Split by space in case one "word" became multiple
            parts = cleaned.split()
            for part in parts:
                if gt_word_idx < len(gt_words_split) and part == gt_words_split[gt_word_idx]:
                    gt_with_speakers.append(spk)
                    gt_word_idx += 1

    # For hypothesis: map preprocessed words back to speakers
    hyp_word_idx = 0
    for spk, orig_word in hyp_speaker_words:
        cleaned = preprocess_word(orig_word)
        if cleaned:
            parts = cleaned.split()
            for part in parts:
                if hyp_word_idx < len(hyp_words_split) and part == hyp_words_split[hyp_word_idx]:
                    hyp_with_speakers.append(spk)
                    hyp_word_idx += 1

    print(f"  DEBUG: GT words after WER split: {len(gt_words_split)}, with speakers: {len(gt_with_speakers)}")
    print(f"  DEBUG: HYP words after WER split: {len(hyp_words_split)}, with speakers: {len(hyp_with_speakers)}")
    print(f"  DEBUG: WER alignment ops: {len(wer_output.alignments[0])}")
    print(f"  DEBUG: WER hits: {wer_output.hits}, total words for WER: {len(gt_words_split)}")

    # Check what the alignment operation objects look like
    if len(wer_output.alignments[0]) > 0:
        first_op = wer_output.alignments[0][0]
        print(f"  DEBUG: First alignment op: type={first_op.type}, ref_start_idx={getattr(first_op, 'ref_start_idx', 'N/A')}, ref_end_idx={getattr(first_op, 'ref_end_idx', 'N/A')}")

    # Get unique speakers from hypothesis
    hyp_speakers = sorted(set(hyp_with_speakers))
    num_speakers = len(hyp_speakers)

    # Try all possible speaker permutations
    best_der = 1.0
    best_mapping = {spk: spk for spk in hyp_speakers}
    best_errors = 0
    best_total = 0

    # Generate all cyclic permutations
    for shift in range(num_speakers):
        # Create mapping: shift speakers in a cycle
        # e.g., shift=1: S1->S2, S2->S3, S3->S1
        speaker_mapping = {}
        for i, hyp_spk in enumerate(hyp_speakers):
            # Extract speaker number, shift it, wrap around
            spk_num = int(hyp_spk[1:]) if hyp_spk.startswith('S') else int(hyp_spk)
            new_num = ((spk_num - 1 + shift) % num_speakers) + 1
            speaker_mapping[hyp_spk] = f"S{new_num}"

        # Calculate errors with this mapping
        speaker_errors = 0
        total_correct_words = 0
        gt_idx = 0
        hyp_idx = 0

        for op in wer_output.alignments[0]:
            # Each operation covers a span of words
            ref_start = op.ref_start_idx
            ref_end = op.ref_end_idx
            hyp_start = op.hyp_start_idx
            hyp_end = op.hyp_end_idx

            if op.type == 'equal':  # Words match (hits)
                # Process all words in this span
                for i in range(ref_end - ref_start):
                    gt_word_idx = ref_start + i
                    hyp_word_idx = hyp_start + i

                    if gt_word_idx < len(gt_with_speakers) and hyp_word_idx < len(hyp_with_speakers):
                        gt_speaker = gt_with_speakers[gt_word_idx]
                        hyp_speaker = hyp_with_speakers[hyp_word_idx]

                        # Apply speaker mapping
                        mapped_hyp_speaker = speaker_mapping.get(hyp_speaker, hyp_speaker)

                        if gt_speaker != mapped_hyp_speaker:
                            speaker_errors += 1

                        total_correct_words += 1
            elif op.type == 'delete':
                # Deletion: words exist in reference but not in hypothesis
                pass
            elif op.type == 'insert':
                # Insertion: words exist in hypothesis but not in reference
                pass
            elif op.type == 'substitute':
                # Substitution: words don't match (don't count for DER)
                pass

        der = speaker_errors / total_correct_words if total_correct_words > 0 else 0.0

        if der < best_der:
            best_der = der
            best_mapping = speaker_mapping
            best_errors = speaker_errors
            best_total = total_correct_words

    return {
        'der': best_der,
        'speaker_errors': best_errors,
        'total_correct_words': best_total,
        'speaker_mapping': best_mapping
    }


def calculate_wer(ground_truth: str, target: str) -> Tuple[Dict[str, float], object]:
    """
    Calculate Word Error Rate and related metrics.

    Args:
        ground_truth: Ground truth transcript text
        target: Target transcript text to evaluate

    Returns:
        Tuple of (metrics dictionary, jiwer output object for visualization)
    """
    # Preprocess both transcripts
    ground_truth_clean = preprocess_transcript(ground_truth)
    target_clean = preprocess_transcript(target)

    # Process words and get detailed output for visualization
    output = jiwer.process_words(ground_truth_clean, target_clean)

    metrics = {
        'wer': output.wer,
        'mer': output.mer,  # Match Error Rate
        'wil': output.wil,  # Word Information Lost
        'wip': output.wip,  # Word Information Preserved
        'hits': output.hits,
        'substitutions': output.substitutions,
        'deletions': output.deletions,
        'insertions': output.insertions,
        'total_words': len(ground_truth_clean.split())
    }

    return metrics, output


def find_matching_files(ground_truth_dir: str, target_dir: str, verbose: bool = True) -> List[Tuple[str, str]]:
    """
    Find matching transcript files in ground truth and target directories.
    Matches target TXT files with ground truth .txt files by finding the base name
    within the ground truth filename.

    Args:
        ground_truth_dir: Directory containing ground truth .txt transcripts
        target_dir: Directory containing target TXT transcripts to evaluate
        verbose: Whether to print debug information

    Returns:
        List of tuples containing (ground_truth_path, target_path) for matching files
    """
    ground_truth_path = Path(ground_truth_dir)
    target_path = Path(target_dir)

    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {ground_truth_dir}")

    if not target_path.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    # Find all .txt files in ground truth directory
    ground_truth_files = list(ground_truth_path.rglob('*.txt'))

    # Find all .txt files in target directory
    target_files = list(target_path.rglob('*.txt'))

    if verbose:
        print(f"Found {len(ground_truth_files)} ground truth .txt files")
        print(f"Found {len(target_files)} target .txt files")
        print()

    matching_pairs = []
    unmatched_targets = []

    for target_file in target_files:
        # Extract base name
        base_name = target_file.stem

        if verbose:
            print(f"Looking for match for: {target_file.name}")
            print(f"  Base name: {base_name}")

        # Find ground truth file containing this base name
        matched = False
        for gt_file in ground_truth_files:
            if base_name in gt_file.stem:
                matching_pairs.append((str(gt_file), str(target_file)))
                if verbose:
                    print(f"  MATCHED with: {gt_file.name}")
                matched = True
                break

        if not matched:
            unmatched_targets.append((base_name, target_file.name))
            if verbose:
                print(f"  NO MATCH FOUND")

        if verbose:
            print()

    if verbose and unmatched_targets:
        print(f"\n{len(unmatched_targets)} unmatched target files:")
        for base_name, filename in unmatched_targets:
            print(f"  - {filename} (looking for '{base_name}' in ground truth)")
        print()

        if ground_truth_files:
            print("Available ground truth files:")
            for gt_file in ground_truth_files[:10]:  # Show first 10
                print(f"  - {gt_file.name}")
            if len(ground_truth_files) > 10:
                print(f"  ... and {len(ground_truth_files) - 10} more")
            print()

    return matching_pairs


def assess_wer_batch(ground_truth_dir: str, target_dir: str, output_csv: str, show_errors: bool = True):
    """
    Batch process and assess WER for all matching transcript pairs.

    Args:
        ground_truth_dir: Directory containing ground truth transcripts
        target_dir: Directory containing target transcripts
        output_csv: Path to output CSV file for results
        show_errors: Whether to display error visualization for each file
    """
    print(f"Searching for matching files...")
    matching_pairs = find_matching_files(ground_truth_dir, target_dir)

    if not matching_pairs:
        print("No matching transcript files found.")
        return

    print(f"Found {len(matching_pairs)} matching file pairs.")
    print(f"\nProcessing transcripts...\n")

    results = []

    for ground_truth_path, target_path in matching_pairs:
        filename = Path(ground_truth_path).name
        print(f"{'='*70}")
        print(f"Processing: {filename}")
        print(f"{'='*70}")

        try:
            # Load transcripts - both from TXT format
            ground_truth_text = load_ground_truth_from_txt(ground_truth_path)
            target_text = load_transcript_from_txt(target_path)

            # Calculate WER and get visualization output
            metrics, wer_output = calculate_wer(ground_truth_text, target_text)

            # Parse speaker-tagged transcripts for DER calculation
            gt_speaker_words = parse_speaker_transcript_txt(ground_truth_path)
            hyp_speaker_words = parse_speaker_transcript_txt(target_path)

            # Calculate DER (reuses WER alignment)
            der_metrics = calculate_der(gt_speaker_words, hyp_speaker_words, wer_output)

            # Store results
            result = {
                'filename': filename,
                'ground_truth_path': ground_truth_path,
                'target_path': target_path,
                'wer': metrics['wer'],
                'mer': metrics['mer'],
                'wil': metrics['wil'],
                'wip': metrics['wip'],
                'hits': metrics['hits'],
                'substitutions': metrics['substitutions'],
                'deletions': metrics['deletions'],
                'insertions': metrics['insertions'],
                'total_words': metrics['total_words'],
                'der': der_metrics['der'],
                'speaker_errors': der_metrics['speaker_errors'],
                'correct_words_for_der': der_metrics['total_correct_words']
            }
            results.append(result)

            print(f"\nWER: {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
            print(f"Total words: {metrics['total_words']}")
            print(f"Hits: {metrics['hits']} | Substitutions: {metrics['substitutions']} | "
                  f"Deletions: {metrics['deletions']} | Insertions: {metrics['insertions']}")

            print(f"\nDER: {der_metrics['der']:.4f} ({der_metrics['der']*100:.2f}%)")
            print(f"Speaker errors: {der_metrics['speaker_errors']} out of {der_metrics['total_correct_words']} correctly recognized words")

            # Show speaker mapping if it's not identity
            identity_mapping = all(k == v for k, v in der_metrics['speaker_mapping'].items())
            if not identity_mapping:
                mapping_str = ', '.join([f"{k}->{v}" for k, v in sorted(der_metrics['speaker_mapping'].items())])
                print(f"Best speaker mapping: {mapping_str}")

        except Exception as e:
            import traceback
            error_msg = str(e) if str(e) else type(e).__name__
            print(f"ERROR: {error_msg}")
            print(f"Traceback:")
            traceback.print_exc()
            results.append({
                'filename': filename,
                'ground_truth_path': ground_truth_path,
                'target_path': target_path,
                'error': error_msg
            })

        print()  # Empty line between files

    # Write results to CSV
    if results:
        fieldnames = ['filename', 'wer', 'der', 'mer', 'wil', 'wip', 'hits', 'substitutions',
                      'deletions', 'insertions', 'total_words', 'speaker_errors',
                      'correct_words_for_der', 'ground_truth_path', 'target_path', 'error']

        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n{'='*70}")
        print(f"Results saved to: {output_csv}")
        print(f"{'='*70}")

        # Print summary statistics
        successful_results = [r for r in results if 'error' not in r]
        if successful_results:
            avg_wer = sum(r['wer'] for r in successful_results) / len(successful_results)
            avg_der = sum(r['der'] for r in successful_results) / len(successful_results)
            print(f"\nSummary Statistics:")
            print(f"  Total files processed: {len(matching_pairs)}")
            print(f"  Successful: {len(successful_results)}")
            print(f"  Failed: {len(results) - len(successful_results)}")
            print(f"\n  Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
            print(f"  Best WER: {min(r['wer'] for r in successful_results):.4f}")
            print(f"  Worst WER: {max(r['wer'] for r in successful_results):.4f}")
            print(f"\n  Average DER: {avg_der:.4f} ({avg_der*100:.2f}%)")
            print(f"  Best DER: {min(r['der'] for r in successful_results):.4f}")
            print(f"  Worst DER: {max(r['der'] for r in successful_results):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Assess Word Error Rate (WER) between ground truth and NVIDIA TXT transcripts.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python assess_wer_nvidia.py ground_truth/ target/ -o results.csv
  python assess_wer_nvidia.py /path/to/ground_truth /path/to/target --output wer_results_nvidia.csv
        """
    )

    parser.add_argument(
        'ground_truth_dir',
        help='Directory containing ground truth .txt transcript files'
    )

    parser.add_argument(
        'target_dir',
        help='Directory containing target TXT transcript files to evaluate'
    )

    parser.add_argument(
        '-o', '--output',
        default='wer_results_nvidia.csv',
        help='Output CSV file path (default: wer_results_nvidia.csv)'
    )

    parser.add_argument(
        '--no-errors',
        action='store_true',
        help='Disable error visualization output (only show summary metrics)'
    )

    args = parser.parse_args()

    assess_wer_batch(args.ground_truth_dir, args.target_dir, args.output,
                     show_errors=not args.no_errors)


if __name__ == '__main__':
    main()
