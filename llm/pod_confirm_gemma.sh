#!/usr/bin/env bash
# RunPod GPU bootstrap for the confirmatory amplification phase (plan: llm/AMPLIFICATION_CONFIRM_PLAN.md §3), Gemma-4 31B.
# Same tracking chain as llm/pod_run.sh: heartbeat to HF every 90 s (runs/$RUN/status.json + log), every unit uploaded
# and verified the moment it exists (llm/local_gemma_confirm.py), everything to stdout, job deadline JOB_HOURS, final
# upload retried, and the pod removes itself ONLY after a verified final upload (otherwise the watchdog ends it).
# Env: HF_TOKEN (secret), RUN, JOB_HOURS (default 3.6), SMOKE_ONLY (stop after the smoke run), NO_REMOVE.
set -uo pipefail
REPO=${REPO:-/root/Social-Phase-Transitions}; export REPO
RUN=${RUN:-${RUNPOD_POD_ID:-local-$(date +%s)}}; export RUN
export POD_T_START=$(date +%s)
JOB_HOURS=${JOB_HOURS:-3.6}
DEADLINE=$(python3 -c "print(int($POD_T_START + float('$JOB_HOURS') * 3600))")
LOGDIR=$REPO/results/llm/logs; mkdir -p "$LOGDIR"; LOG=$LOGDIR/pod_${RUN}.log
export HF_HOME=${HF_HOME:-$([ -d /workspace ] && echo /workspace/hf || echo /root/hf)}
export HF_HUB_DISABLE_XET=1
MID=${MID:-google/gemma-4-31b-it}
cd "$REPO"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
phase() { python - "$@" <<'PY'
import json, os, sys, time
p = os.path.join(os.environ["REPO"], "results/llm/logs/phase.json")
open(p + ".tmp", "w").write(json.dumps({"phase": sys.argv[1], "detail": " ".join(sys.argv[2:]), "t": time.time()})); os.replace(p + ".tmp", p)
PY
  log "PHASE $*"; }
finish() {
  code=$1; trap - EXIT; trap '' TERM
  kill "$TIMER" 2>/dev/null
  [ -f "$LOGDIR/local_gemma.pid" ] && kill "$(cat "$LOGDIR/local_gemma.pid")" 2>/dev/null   # by PID, never by pattern
  phase finishing "exit $code"; ok=0
  while [ "$(date +%s)" -lt $(( DEADLINE + ${FINAL_RETRY_S:-1800} )) ]; do
    if python llm/pod_tracker.py upload --run "$RUN" --key gemma-confirm-ALL 2>&1 | tee -a "$LOG" | grep -q "UPLOADED gemma-confirm-ALL OK"; then ok=1; break; fi
    log "final upload failed; retrying in 60 s"; sleep 60
  done
  if [ $ok = 1 ]; then
    phase done "exit $code; all outputs uploaded and verified"; sleep 100
    if [ -z "${NO_REMOVE:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then log "removing pod $RUNPOD_POD_ID"; runpodctl remove pod "$RUNPOD_POD_ID"; fi
  else
    phase upload_failed "exit $code; NOT removing the pod (watchdog will end it)"
  fi
  kill "$HB" 2>/dev/null; exit "$code"
}
trap 'finish $?' EXIT
trap 'log "DEADLINE/TERM received"; exit 5' TERM
( sleep $(( DEADLINE - POD_T_START )); log "JOB DEADLINE (${JOB_HOURS} h) reached"; kill -TERM $$ ) &
TIMER=$!
phase setup "installing packages"
pip install -q "transformers==5.19.0" accelerate huggingface_hub pandas numpy 2>&1 | tail -2 | tee -a "$LOG"
python llm/pod_tracker.py heartbeat --run "$RUN" --log "$LOG" --interval "${HB_INTERVAL:-90}" >> "$LOG" 2>&1 &
HB=$!
log "run $RUN; deadline $(date -u -d @$DEADLINE +%H:%M:%S) UTC"
[ "${DEVICE:-cuda}" = cuda ] && nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1 | tee -a "$LOG"
phase download "$MID"
if [ -z "${SKIP_DOWNLOAD:-}" ]; then
  python llm/pod_tracker.py download --run "$RUN" --model "$MID" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "download failed"; exit 2; }
fi
phase smoke "$MID"
python llm/local_gemma_confirm.py --model-id "$MID" --device "${DEVICE:-cuda}" ${CONFIRM_ARGS:-} --smoke 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = 0 ] || { log "smoke run failed"; exit 3; }
[ -n "${SMOKE_ONLY:-}" ] && { log "SMOKE_ONLY: stopping after smoke"; exit 0; }
phase main "$MID"
python llm/local_gemma_confirm.py --model-id "$MID" --device "${DEVICE:-cuda}" ${CONFIRM_ARGS:-} 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = 0 ] || { log "main run failed"; exit 4; }
exit 0
