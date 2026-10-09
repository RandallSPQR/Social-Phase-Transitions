#!/usr/bin/env bash
# RunPod bootstrap for Amendment 4 (run inside the pod). Requires: HF_TOKEN (Gemma weights + results upload) and
# the repo checked out at $REPO (default /root/Social-Phase-Transitions, on the container disk).
# The owner's network volume (mounted at /workspace) is SHARED with other experiments. This script writes on it
# only inside its own directory $NS (/workspace/social-phase-transitions, HF cache under $NS/hf), refuses to
# start if that directory exists without our marker file, and never deletes anything on the volume. If the
# volume lacks room for the weights, they go to the container disk instead. Self-terminates the pod at the end
# (success or failure) via runpodctl, so billing stops; it only ever removes its own pod ($RUNPOD_POD_ID).
set -uo pipefail
# hard backstop: whole job (both models) must finish within 4 h or the pod terminates (trap below)
( sleep 14400; echo 'HARD TIME LIMIT' ; kill -TERM $$ ) &
REPO=${REPO:-/root/Social-Phase-Transitions}; export REPO
LOG=$REPO/results/llm/logs/pod_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$(dirname "$LOG")"
NS=/workspace/social-phase-transitions; MARK=social-phase-transitions-amendment4
NEED_GB=130   # both Gemma-4 checkpoints (~115 GB) + margin
if [ -d /workspace ]; then
  if [ -e "$NS" ] && [ "$(cat "$NS/.owner" 2>/dev/null)" != "$MARK" ]; then
    echo "ABORT: $NS exists on the shared volume but is not ours; not touching it" | tee -a "$LOG"; exit 2
  fi
  have_gb=$(du -s -BG "$NS/hf" 2>/dev/null | cut -f1 | tr -dc 0-9); have_gb=${have_gb:-0}
  free_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
  echo "volume: free ${free_gb} GB, already cached ${have_gb} GB" | tee -a "$LOG"
  if [ $((free_gb + have_gb)) -ge $NEED_GB ]; then
    mkdir -p "$NS/hf"; [ -e "$NS/.owner" ] || echo "$MARK" > "$NS/.owner"
    export HF_HOME=$NS/hf
  else
    echo "volume too full for the weights; using container disk" | tee -a "$LOG"; export HF_HOME=/root/hf
  fi
else
  export HF_HOME=/root/hf
fi
echo "HF_HOME=$HF_HOME" | tee -a "$LOG"
finish() {
  echo "pod finishing (exit $1) $(date)" | tee -a "$LOG"
  # ship results before terminating: tarball to the HF dataset (private) under pod_results/
  tar czf /root/pod_results.tgz -C "$REPO" results/llm/stage1 results/llm/activations results/llm/logs 2>/dev/null
  python - <<'EOF' 2>&1 | tee -a "$LOG"
import json, os
from huggingface_hub import HfApi
m = json.load(open(os.path.join(os.environ["REPO"], "results/llm/cache_manifest.json")))
HfApi(token=os.environ["HF_TOKEN"]).upload_file(repo_id=m["dataset"]["repo_id"], repo_type="dataset",
    path_or_fileobj="/root/pod_results.tgz", path_in_repo=f"pod_results/pod_results_{os.environ.get('RUNPOD_POD_ID','x')}.tgz")
print("uploaded pod results")
EOF
  [ -n "${RUNPOD_POD_ID:-}" ] && runpodctl remove pod "$RUNPOD_POD_ID"
}
trap 'finish $?' EXIT
pip install -q "transformers==5.19.0" accelerate huggingface_hub pandas numpy 2>&1 | tail -2 | tee -a "$LOG"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$LOG"
for spec in "google/gemma-4-26b-a4b-it gemma-26b-local" "google/gemma-4-31b-it gemma-31b-local"; do
  set -- $spec
  # smoke test first: 40 prompts; abort the whole run if it fails
  timeout 30m python llm/local_gemma.py --model-id "$1" --key "${2}-smoke" --max-prompts 40 2>&1 | tee -a "$LOG" || exit 3
  timeout 150m python llm/local_gemma.py --model-id "$1" --key "$2" 2>&1 | tee -a "$LOG" || exit 4
done
