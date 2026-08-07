#!/usr/bin/env bash
# Local watcher: when the cohort transfer finishes, top up every sweep over
# the visits that landed late, then re-run the scoring chain on the server.
#
# The transcription sweeps snapshot the cohort when they start, so visits that
# arrive mid-sweep are missed. Every runner skips finished work, so a second
# pass is cheap and only picks up stragglers.
#
# Runs on the laptop because only it can see the transfer process. Launch with
# nohup + disown so it outlives the session that started it:
#   nohup bash scripts/finalize_after_transfer.sh > /tmp/finalize.log 2>&1 &
#   disown

set -uo pipefail

REMOTE="gpu2"
REPO="/data/data/wolfflab/btdixon/Dixon/crisper-whisper-2"
COHORT="/data/data/wolfflab/Data/Chirp3_PSYCHS_NDA4"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] finalize: $*"; }

remote() { ssh -o ConnectTimeout=60 -o ServerAliveInterval=30 "$REMOTE" "cd $REPO && $1"; }

log "waiting for the cohort transfer to finish"
while pgrep -f fetch_cohort >/dev/null; do sleep 120; done
log "transfer finished"

log "waiting for in-flight sweeps and orchestrator to finish"
while remote 'pgrep -f "run_cohort|run_baseline.py|orchestrate_eval" >/dev/null'; do sleep 120; done
log "cluster idle; starting top-up pass"

log "top-up: ours"
remote "nohup uv run python scripts/run_cohort.py --cohort $COHORT --mode ours \
  --output-dir outputs/ours --device-index 1 > /tmp/ours_topup.log 2>&1"
log "top-up: verbatimize"
remote "nohup uv run python scripts/run_cohort.py --cohort $COHORT --mode verbatimize \
  --output-dir outputs/verbatimize --device-index 2 > /tmp/verb_topup.log 2>&1"
log "top-up: baseline"
remote "nohup uv run python baseline/run_baseline.py --cohort $COHORT \
  --output-dir baseline/outputs --device-index 3 > /tmp/baseline_topup.log 2>&1"

log "re-running the scoring chain"
remote "nohup bash scripts/orchestrate_eval.sh > /tmp/orchestrate_final.log 2>&1"

log "COMPLETE"
