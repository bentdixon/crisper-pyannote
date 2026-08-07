#!/usr/bin/env bash
# End-to-end evaluation chain, run detached on the cluster.
#
# Waits for the in-flight transcription sweeps, then produces the two LLM
# variants and scores all five systems with the DIALOG-DeID metrics. Written
# to run server-side so the chain survives a dropped session: every stage is
# idempotent, so re-running picks up wherever it stopped.
#
# Stages:
#   1. wait for run_cohort (ours, verbatimize) and run_baseline to exit
#   2. role-label our output so the role-based LLM review can consume it
#   3. LLM review + apply on the baseline tree      (GPU $LLM_GPU_A)
#   4. LLM review + apply on our role-labelled tree (GPU $LLM_GPU_B)
#   5. score chirp3 / ours / verbatimize / baseline / baseline_llm / ours_llm
#
# Usage: nohup bash scripts/orchestrate_eval.sh > /tmp/orchestrate.log 2>&1 &

set -uo pipefail

REPO="/data/data/wolfflab/btdixon/Dixon/crisper-whisper-2"
COHORT="/data/data/wolfflab/Data/Chirp3_PSYCHS_NDA4"
LLM_GPU_A="cuda:1"
LLM_GPU_B="cuda:2"
RESULTS="$REPO/outputs/results.json"

cd "$REPO" || exit 1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] orchestrate: $*"; }

log "waiting for transcription sweeps to finish"
while pgrep -f "mode ours" >/dev/null \
   || pgrep -f "mode verbatimize" >/dev/null \
   || pgrep -f "run_baseline.py" >/dev/null; do
  sleep 120
done
log "sweeps done"
log "  ours:        $(find outputs/ours -name transcript.json 2>/dev/null | wc -l) visit(s)"
log "  verbatimize: $(find outputs/verbatimize -name transcript.json 2>/dev/null | wc -l) visit(s)"
log "  baseline:    $(find baseline/outputs -name '*_transcript.txt' 2>/dev/null | wc -l) visit(s)"

log "stage 2: role-labelling our output"
uv run python scripts/to_role_transcript.py \
  --inputs outputs/ours --output-dir outputs/ours_roles 2>&1 | tail -3

log "stage 3: LLM review on the baseline tree"
uv run python baseline/apply_llm_corrections.py \
  --outputs baseline/outputs --device "$LLM_GPU_A" 2>&1 | tail -12

log "stage 4: LLM review on our role-labelled tree"
uv run python baseline/apply_llm_corrections.py \
  --outputs outputs/ours_roles --device "$LLM_GPU_B" 2>&1 | tail -12

log "stage 5: scoring all systems"
mkdir -p "$(dirname "$RESULTS")"
uv run python scripts/evaluate_systems.py \
  --cohort "$COHORT" \
  --system chirp3 \
  --system ours=outputs/ours \
  --system verbatimize=outputs/verbatimize \
  --system baseline=baseline/outputs \
  --system baseline_llm=baseline/outputs \
  --system ours_llm=outputs/ours_roles \
  --output "$RESULTS" 2>&1 | tail -20

log "COMPLETE -- results at $RESULTS"
