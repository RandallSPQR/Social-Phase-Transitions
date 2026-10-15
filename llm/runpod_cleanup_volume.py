"""One-off (owner-approved 2026-10-09): remove OUR leftover directory from the owner's shared network volume.

The first Amendment 4 launch left a partial Gemma download in /workspace/social-phase-transitions/ (marked by our
.owner file) and hit the volume's 150 GB quota. This starts a short pod in the volume's data centre that deletes
exactly that directory, only if the marker matches, logs volume usage before/after, uploads the log to the private
dataset (cleanup/<pod id>.log) and removes itself. This script terminates the pod after --max-minutes regardless.
Nothing else on the volume is read beyond `du` totals, and nothing else is touched.
  python llm/runpod_cleanup_volume.py
"""
import argparse, os, sys, time
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runpod_launch import H, REST, IMAGE, log, pod, terminate

DC, VOLUME = "EUR-IS-1", "u0isne6ams"
GPUS = ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation"]
NS, MARK = "/workspace/social-phase-transitions", "social-phase-transitions-amendment4"

SCRIPT = f"""
L=/root/cleanup.log
echo "start $(date -u)" > $L
echo "volume used before: $(timeout 600 du -sh /workspace 2>/dev/null | cut -f1)" >> $L
if [ -e {NS} ]; then
  if [ "$(cat {NS}/.owner 2>/dev/null)" = "{MARK}" ]; then
    echo "ours ({MARK}); size $(timeout 600 du -sh {NS} 2>/dev/null | cut -f1); removing" >> $L
    rm -rf {NS}; echo "removed: $([ -e {NS} ] && echo NO || echo yes)" >> $L
  else
    echo "ABORT: {NS} exists but marker does not match; not touched" >> $L
  fi
else
  echo "{NS} not present; nothing to do" >> $L
fi
echo "volume used after: $(timeout 600 du -sh /workspace 2>/dev/null | cut -f1)" >> $L
pip install -q huggingface_hub >/dev/null 2>&1
python -c "import os; from huggingface_hub import HfApi; HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj='/root/cleanup.log', path_in_repo='cleanup/'+os.environ.get('RUNPOD_POD_ID','x')+'.log', repo_id='RandallSPQR/social-phase-transitions-llm-cache', repo_type='dataset')" >> $L 2>&1
[ -n "${{RUNPOD_POD_ID:-}}" ] && runpodctl remove pod "$RUNPOD_POD_ID"
sleep infinity
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-minutes", type=float, default=20); a = ap.parse_args()
    body = {"name": "spt-volume-cleanup", "imageName": IMAGE, "gpuTypeIds": GPUS, "gpuCount": 1, "cloudType": "SECURE",
            "dataCenterIds": [DC], "networkVolumeId": VOLUME, "volumeMountPath": "/workspace", "containerDiskInGb": 20,
            "env": {"HF_TOKEN": os.environ["HF_TOKEN"]}, "dockerEntrypoint": ["bash", "-c"], "dockerStartCmd": [SCRIPT]}
    r = requests.post(f"{REST}/pods", headers=H(), json=body, timeout=60)
    if r.status_code >= 300:
        log(f"cleanup pod create failed: http {r.status_code} {r.text[:300]}"); sys.exit(1)
    p = r.json(); pid, t0 = p["id"], time.time()
    log(f"CLEANUP pod {pid} created ({(p.get('machine') or {}).get('gpuTypeId')}, ${p.get('costPerHr')}/h)")
    while True:
        time.sleep(30)
        if pod(pid) is None:
            log(f"cleanup pod {pid} gone"); return
        if (time.time() - t0) / 60 > a.max_minutes:
            terminate(pid, f"cleanup pod over {a.max_minutes} min"); return


if __name__ == "__main__":
    main()
