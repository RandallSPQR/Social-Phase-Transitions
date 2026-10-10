#!/usr/bin/env bash
# RunPod bootstrap for Amendment 4 (run inside the pod). Requires HF_TOKEN (Gemma weights + results) and the repo at
# $REPO (default /root/Social-Phase-Transitions, container disk; no network volume is mounted).
#
# Tracking (see llm/pod_tracker.py): a heartbeat uploads runs/$RUN/status.json + pod.log to the private dataset every
# 90 s; each model's outputs (smoke and full) are uploaded and verified the moment that model finishes. Everything is
# also printed to stdout, so the RunPod console's Logs tab shows it live.
# End: all outputs uploaded and verified, retrying until the deadline; the pod removes itself ONLY after a verified
# upload. If uploading never succeeds the pod stays up and the session watchdog ends it at its time cap.
#
# Env overrides (local end-to-end tests): RUN, MODELS ("model-id key;..."), DEVICE, MAX_PROMPTS, JOB_HOURS, NO_REMOVE.
set -uo pipefail
REPO=${REPO:-/root/Social-Phase-Transitions}; export REPO
RUN=${RUN:-${RUNPOD_POD_ID:-local-$(date +%s)}}; export RUN
export POD_T_START=$(date +%s)
JOB_HOURS=${JOB_HOURS:-3.6}        # job deadline; the watchdog's cap (4.25 h) leaves time for the final uploads
DEADLINE=$(python3 -c "print(int($POD_T_START + float('$JOB_HOURS') * 3600))")
LOGDIR=$REPO/results/llm/logs; mkdir -p "$LOGDIR"
LOG=$LOGDIR/pod_${RUN}.log
# Weights go to the pod's OWN 200 GB volume at /workspace (created with the pod, deleted with it; never the owner's
# shared network volume). Rerun 2 showed that writes under /root were capped near 20 GB despite a 200 GB container disk.
export HF_HOME=${HF_HOME:-$([ -d /workspace ] && echo /workspace/hf || echo /root/hf)}
# The xet download backend fails on RunPod storage with "Disk quota exceeded" even with ~200 GB free (first launch and
# the 2026-10-09 21:23 rerun, both at the first weight shard). Use the classic HTTP download path instead.
export HF_HUB_DISABLE_XET=1
DEVICE=${DEVICE:-cuda}
MODELS=${MODELS:-"google/gemma-4-26b-a4b-it gemma-26b-local;google/gemma-4-31b-it gemma-31b-local"}
cd "$REPO"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
phase() { python - "$@" <<'EOF'
import json, os, sys, time
p = os.path.join(os.environ["REPO"], "results/llm/logs/phase.json")
d = {"phase": sys.argv[1], "detail": " ".join(sys.argv[2:]), "t": time.time()}
open(p + ".tmp", "w").write(json.dumps(d)); os.replace(p + ".tmp", p)
EOF
  log "PHASE $*"; }

finish() {
  code=$1; trap - EXIT; trap '' TERM             # nothing may interrupt the final uploads
  kill "$TIMER" 2>/dev/null
  pkill -f "llm/local_gemma.py" 2>/dev/null
  phase finishing "exit $code"
  ok=0
  while [ "$(date +%s)" -lt $(( DEADLINE + ${FINAL_RETRY_S:-1800} )) ]; do
    if python llm/pod_tracker.py upload-all --run "$RUN" 2>&1 | tee -a "$LOG" | grep -q "UPLOADED all OK"; then ok=1; break; fi
    log "final upload failed; retrying in 60 s"; sleep 60
  done
  if [ $ok = 1 ]; then
    phase done "exit $code; all outputs uploaded and verified"
    sleep 100                                       # one more heartbeat carries the final status
    if [ -z "${NO_REMOVE:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then log "removing pod $RUNPOD_POD_ID"; runpodctl remove pod "$RUNPOD_POD_ID"; fi
  else
    phase upload_failed "exit $code; NOT removing the pod (watchdog will end it)"
  fi
  kill "$HB" 2>/dev/null
  exit "$code"
}
trap 'finish $?' EXIT
trap 'log "DEADLINE/TERM received"; exit 5' TERM
( sleep $(( DEADLINE - POD_T_START )); log "JOB DEADLINE (${JOB_HOURS} h) reached"; kill -TERM $$ ) &
TIMER=$!

phase setup "installing packages"
pip install -q "transformers==5.19.0" accelerate huggingface_hub pandas numpy 2>&1 | tail -2 | tee -a "$LOG"
python llm/pod_tracker.py heartbeat --run "$RUN" --log "$LOG" --interval "${HB_INTERVAL:-90}" >> "$LOG" 2>&1 &
HB=$!
log "run $RUN; deadline $(date -u -d @$DEADLINE +%H:%M:%S) UTC; device $DEVICE"
[ "$DEVICE" = cuda ] && nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$LOG"

IFS=';' read -ra SPECS <<< "$MODELS"
for spec in "${SPECS[@]}"; do
  set -- $spec; MID=$1; KEY=$2
  phase download "$MID"
  python llm/pod_tracker.py download --run "$RUN" --model "$MID" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "download failed for $MID"; exit 2; }
  phase smoke "$KEY"
  python llm/local_gemma.py --model-id "$MID" --key "${KEY}-smoke" --max-prompts 40 --device "$DEVICE" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "smoke test failed for $KEY"; exit 3; }
  python llm/pod_tracker.py upload --run "$RUN" --key "${KEY}-smoke" 2>&1 | tee -a "$LOG"
  phase main "$KEY"
  python llm/local_gemma.py --model-id "$MID" --key "$KEY" --device "$DEVICE" ${MAX_PROMPTS:+--max-prompts $MAX_PROMPTS} 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "main run failed for $KEY"; exit 4; }
  phase upload "$KEY"
  python llm/pod_tracker.py upload --run "$RUN" --key "$KEY" 2>&1 | tee -a "$LOG"
done
exit 0
