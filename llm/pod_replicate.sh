#!/usr/bin/env bash
# RunPod CPU pod: continue the option B replicate (llm/stage3_replicate.py) until done. Run inside the pod.
# Credentials: ONLY the owner's two RunPod secrets, passed as env HF_TOKEN and OPENROUTER_API_KEY (dedicated key,
# $10 cap). Nothing else is needed (self-removal uses runpodctl's pod-scoped key, which RunPod provides in the image).
#
# 1. Restore the run state from the private HF dataset runs/stage3-rep1/: the five cache files (full copies, every
#    call so far), saved chains, sentinel prompts and log, interruptions.jsonl, run_t0.txt. Resume = replay from cache.
# 2. Run the replicate (5 streams, one call in flight per endpoint, sentinels every 5 min, $7 budget stop). The runner
#    uploads status + cache copies + chains to runs/stage3-rep1/ every 15 min and verifies each upload by remote size.
# 3. Final upload, retried until verified; only then the pod removes itself. If it never verifies, the pod stays up and
#    the session's check-ins end it after checking HF. Job deadline JOB_HOURS (default 40 h).
# Everything goes to stdout (RunPod Logs tab) and /root/pod_replicate.log.
set -uo pipefail
REPO=${REPO:-/root/Social-Phase-Transitions}; cd "$REPO"
LOG=/root/pod_replicate.log
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
: "${HF_TOKEN:?HF_TOKEN secret missing}" "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY secret missing}"
JOB_HOURS=${JOB_HOURS:-40}
log "pod ${RUNPOD_POD_ID:-?} starting; installing deps"
pip install -q numpy pandas scipy numba networkx requests huggingface_hub >> "$LOG" 2>&1
log "restoring run state from HF runs/stage3-rep1/"
python - <<'EOF' 2>&1 | tee -a "$LOG"
import gzip, os, shutil, sys
sys.path.insert(0, "llm"); import fetch_cache
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=os.environ["HF_TOKEN"]); repo = fetch_cache.manifest()["dataset"]["repo_id"]
files = [x.path for x in api.list_repo_tree(repo, repo_type="dataset", path_in_repo="runs/stage3-rep1", recursive=True) if hasattr(x, "size")]
out = "results/llm/stage3_rep1"; tmp = "/root/restore"
for f in files:
    if os.path.basename(f).startswith("boot_"):     # boot logs are records, not run state
        continue
    p = hf_hub_download(repo, f, repo_type="dataset", local_dir=tmp, token=os.environ["HF_TOKEN"])
    rel = f[len("runs/stage3-rep1/"):]
    if rel.startswith("cache/"):
        dst = os.path.join("results/llm/cache", os.path.basename(rel)[:-3]); os.makedirs(os.path.dirname(dst), exist_ok=True)
        with gzip.open(p, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    else:
        dst = os.path.join(out, rel); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copyfile(p, dst)
n = {os.path.basename(f): sum(1 for _ in open(os.path.join("results/llm/cache", f))) for f in os.listdir("results/llm/cache") if f.endswith(".jsonl")}
print("restored", len(files), "files; cache records", n, flush=True)
assert len(n) == 5 and os.path.exists(os.path.join(out, "run_t0.txt")) and os.path.exists(os.path.join(out, "sentinel_prompts.json")), "restore incomplete"
EOF
[ "${PIPESTATUS[0]}" = 0 ] || { log "RESTORE FAILED; idling (nothing was run)"; sleep infinity; }
[ -n "${RESTORE_ONLY:-}" ] && { log "RESTORE_ONLY set: stopping after restore (test)"; exit 0; }
log "starting replicate (deadline ${JOB_HOURS} h)"
CLIENT_SLIM=1 OMP_NUM_THREADS=1 timeout "${JOB_HOURS}h" python llm/stage3_replicate.py run 2>&1 | tee -a "$LOG"
log "runner exited (code ${PIPESTATUS[0]}); final upload"
ok=0
for i in $(seq 1 60); do
  res=$(CLIENT_SLIM=1 python -c "
import sys; sys.path.insert(0,'llm'); import stage3_replicate as SR
r = SR.Run.__new__(SR.Run); r.cfg = SR.Cfg(False); import threading; r.lock = threading.Lock()
import json; r.upload(json.load(open('results/llm/stage3_rep1/status.json')))" 2>&1 | grep "UPLOAD verified" | tail -1)
  log "final upload attempt $i: $res"
  if echo "$res" | grep -qE "verified ([0-9]+)/\1 files$"; then ok=1; break; fi
  sleep 60
done
done_flag=$(python -c "import json; print(json.load(open('results/llm/stage3_rep1/status.json')).get('done'))")
if [ $ok = 1 ]; then
  log "FINAL UPLOAD VERIFIED (run done: $done_flag)"
  if { [ "$done_flag" = "True" ] || [ -n "${E2E_TEST:-}" ]; } && [ -z "${NO_REMOVE:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    sleep 120; log "removing pod $RUNPOD_POD_ID"; runpodctl remove pod "$RUNPOD_POD_ID" 2>&1 | tee -a "$LOG"
  fi
else
  log "FINAL UPLOAD NOT VERIFIED; NOT removing the pod (session check-in will decide)"
fi
sleep infinity
