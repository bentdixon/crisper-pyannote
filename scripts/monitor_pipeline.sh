#!/usr/bin/env bash
# Event stream for the five-system evaluation run.
#
# Emits a line only when something actionable happens, plus a heartbeat every
# half hour so a quiet stretch can be told apart from a dead monitor:
#   - orchestrator stage transitions (its log's last line changing)
#   - the transfer finishing
#   - new relay/transcription failures
#   - every worker gone with no results written (crash or stall)
#   - results.json appearing (done)
#
# Coverage note: greping only for success would stay silent through a
# crashloop, so worker-death and error-count checks are part of the loop.

REPO="/data/data/wolfflab/btdixon/Dixon/crisper-whisper-2"
COHORT="/data/data/wolfflab/Data/Chirp3_PSYCHS_NDA4"
RELAY_LOG="/private/tmp/claude-501/-Users-benjamindixon-crisper-whisper-2/00b6cdaa-fa7c-434d-a0a6-b053bd23f2e1/scratchpad/relay_full.log"

prev_stage=""
prev_errors=-1
prev_transfer=""
prev_stalled=""
ticks=0

remote() {
  ssh -o ConnectTimeout=45 -o ServerAliveInterval=20 gpu2 "$1" 2>/dev/null
}

while true; do
  ticks=$((ticks + 1))

  state=$(remote "cd $REPO 2>/dev/null && printf '%s %s %s %s %s %s %s' \
    \$(find $COHORT -mindepth 3 -maxdepth 3 -type d 2>/dev/null | wc -l) \
    \$(find outputs/ours -name transcript.json 2>/dev/null | wc -l) \
    \$(find outputs/verbatimize -name transcript.json 2>/dev/null | wc -l) \
    \$(find baseline/outputs -name '*_transcript.txt' 2>/dev/null | wc -l) \
    \$(find . -name '*_transcript_corrected.txt' 2>/dev/null | wc -l) \
    \$((\$(pgrep -cf '[r]un_cohort.py') + \$(pgrep -cf '[r]un_baseline.py') + \$(pgrep -cf '[a]pply_llm_corrections') + \$(pgrep -cf '[e]valuate_systems.py'))) \
    \$(test -f outputs/results.json && echo yes || echo no)")

  if [ -z "$state" ]; then
    echo "WARN: cannot reach gpu2 (ssh failed)"
    sleep 300
    continue
  fi

  read -r visits ours verb base llm workers done_flag <<<"$state"
  stage=$(remote "tail -1 /tmp/orchestrate.log 2>/dev/null")

  # 1. results landed -> report and stop
  if [ "$done_flag" = "yes" ]; then
    echo "DONE: outputs/results.json written (visits=$visits ours=$ours verb=$verb base=$base llm=$llm)"
    exit 0
  fi

  # 2. orchestrator moved to a new stage
  if [ -n "$stage" ] && [ "$stage" != "$prev_stage" ]; then
    echo "STAGE: ${stage#*orchestrate: }"
    prev_stage="$stage"
  fi

  # 3. transfer state change
  if pgrep -f fetch_cohort >/dev/null; then transfer="running"; else transfer="finished"; fi
  if [ "$transfer" != "$prev_transfer" ]; then
    [ -n "$prev_transfer" ] && echo "TRANSFER: $transfer (cohort now $visits visits)"
    prev_transfer="$transfer"
  fi

  # 4. new failures anywhere
  relay_errors=$(grep -c "Relay failed" "$RELAY_LOG" 2>/dev/null || echo 0)
  run_errors=$(remote "cat /tmp/ours_s*.log /tmp/verb_s*.log /tmp/baseline_shard*.log /tmp/baseline_run2.log 2>/dev/null | grep -cE 'Failed on|Traceback|out of memory'" || echo 0)
  errors=$((relay_errors + run_errors))
  if [ "$prev_errors" -ge 0 ] && [ "$errors" -gt "$prev_errors" ]; then
    echo "ERRORS: $((errors - prev_errors)) new failure(s); total relay=$relay_errors run=$run_errors"
  fi
  prev_errors=$errors

  # 5. everything stopped but nothing produced -> stalled.
  # Reported once per transition, not every tick, so a genuine stall does not
  # bury the events that follow it. The worker count above must include every
  # stage that can legitimately be the only thing running -- scoring included,
  # or a long scoring pass reads as a stall.
  if [ "$workers" -eq 0 ] && [ "$transfer" = "finished" ] && [ "$done_flag" = "no" ]; then
    stalled="yes"
  else
    stalled="no"
  fi
  if [ "$stalled" = "yes" ] && [ "$stalled" != "$prev_stalled" ]; then
    echo "STALLED: no workers running, transfer finished, no results.json (ours=$ours verb=$verb base=$base llm=$llm)"
  fi
  prev_stalled="$stalled"

  # 6. heartbeat every 30 min
  if [ $((ticks % 6)) -eq 0 ]; then
    echo "PROGRESS: cohort=$visits/269 ours=$ours verb=$verb base=$base llm_corrected=$llm workers=$workers transfer=$transfer"
  fi

  sleep 300
done
