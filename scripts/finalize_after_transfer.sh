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
#
# Two traps this script exists to avoid, both learned the hard way:
#   - pgrep -f PATTERN run over ssh matches the ssh command line itself, so a
#     plain "wait until no matches" loop never terminates. Every pattern here
#     uses the [x]yz bracket form so it cannot match its own invocation.
#   - uv is not on PATH in a non-login ssh shell, so it is called by absolute
#     path; without this the top-up silently does nothing.

set -uo pipefail

REMOTE="gpu2"
REPO="/data/data/wolfflab/btdixon/Dixon/crisper-whisper-2"
COHORT="/data/data/wolfflab/Data/Chirp3_PSYCHS_NDA4"
UV="/home/btdixon/.local/bin/uv"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] finalize: $*"; }

remote() { ssh -o ConnectTimeout=60 -o ServerAliveInterval=30 "$REMOTE" "cd $REPO && $1"; }

busy() {
  # Bracket form so the pattern cannot match this ssh invocation's own argv.
  remote 'pgrep -f "[r]un_cohort.py|[r]un_baseline.py|[o]rchestrate_eval|[a]pply_llm_corrections|[e]valuate_systems" >/dev/null'
}

log "waiting for the cohort transfer to finish"
while pgrep -f "[f]etch_cohort" >/dev/null; do sleep 120; done
log "transfer finished"

log "waiting for in-flight work to finish"
while busy; do sleep 120; done
log "cluster idle; starting sharded top-up pass"

# Two shards per sweep, each pinned to its own GPU. CUDA_VISIBLE_DEVICES is
# used rather than --device-index because the latter only steers the CT2 ASR
# model; pyannote and the ported core pin to cuda:0 regardless.
remote "for s in 1 2; do CUDA_VISIBLE_DEVICES=\$s nohup $UV run python scripts/run_cohort.py \
  --cohort $COHORT --mode ours --output-dir outputs/ours --shard \$s/2 --device-index 0 \
  > /tmp/ours_topup\$s.log 2>&1 & done"
remote "for s in 1 2; do CUDA_VISIBLE_DEVICES=\$((s+2)) nohup $UV run python scripts/run_cohort.py \
  --cohort $COHORT --mode verbatimize --output-dir outputs/verbatimize --shard \$s/2 --device-index 0 \
  > /tmp/verb_topup\$s.log 2>&1 & done"
remote "for s in 1 2; do CUDA_VISIBLE_DEVICES=\$((s+4)) nohup $UV run python baseline/run_baseline.py \
  --cohort $COHORT --output-dir baseline/outputs --shard \$s/2 --device-index 0 \
  > /tmp/baseline_topup\$s.log 2>&1 & done"
log "top-up sweeps launched (2 shards each)"

sleep 60
while busy; do sleep 120; done
log "top-up sweeps finished"
remote "printf 'ours %s | verb %s | base %s\n' \
  \$(find outputs/ours -name transcript.json | wc -l) \
  \$(find outputs/verbatimize -name transcript.json | wc -l) \
  \$(find baseline/outputs -name '*_transcript.txt' | wc -l)"

log "re-running the scoring chain"
remote "setsid nohup bash scripts/orchestrate_eval.sh > /tmp/orchestrate_final.log 2>&1 < /dev/null &"
sleep 60
while busy; do sleep 120; done

log "COMPLETE"
remote "tail -25 /tmp/orchestrate_final.log"
