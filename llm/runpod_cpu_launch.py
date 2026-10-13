"""Launch / police the CPU pod for the option B replicate (llm/pod_replicate.sh).

  python llm/runpod_cpu_launch.py bundle                      # git archive HEAD -> HF code/pod_code_<sha>.tgz (verified)
  python llm/runpod_cpu_launch.py launch --hf-secret NAME --or-secret NAME [--dry-run]
  python llm/runpod_cpu_launch.py check                       # session check-in: pod state, uptime, spend, HF heartbeat age;
                                                              # terminates THIS pod only on a cap or a verified finish

The pod receives only references to the owner's RunPod secrets ({{ RUNPOD_SECRET_<name> }}), never a credential value.
CPU: general-purpose, 2 vCPU (8 GB RAM; the runner peaked at 2.6 GB). Never mounts a network volume; never touches any
other pod. Caps (pod side): JOB_HOURS 40, API budget $7 (runner), OpenRouter key cap $10 (owner). Caps (check-ins):
uptime > 44 h or pod spend (rate x uptime) > $8 -> terminate after a final HF check. The pod id goes to
results/llm/logs/rep1_pod.json.
"""
import argparse, json, os, subprocess, sys, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "llm"))
REST, GQL = "https://rest.runpod.io/v1", "https://api.runpod.io/graphql"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"   # has python, pip and runpodctl (pod self-removal)
NAME = "spt-stage3-rep1"
PODF = os.path.join(ROOT, "results/llm/logs/rep1_pod.json")
MAX_H, MAX_POD_USD = 44.0, 8.0
H = lambda: {"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"], "Content-Type": "application/json"}


def hf():
    import fetch_cache
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"]), fetch_cache.manifest()["dataset"]["repo_id"]


def bundle():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert not subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip(), "commit first"
    tgz = f"/tmp/pod_code_{sha}.tgz"
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", tgz, "HEAD"], cwd=ROOT, check=True)
    api, repo = hf()
    api.upload_file(path_or_fileobj=tgz, path_in_repo=f"code/pod_code_{sha}.tgz", repo_id=repo, repo_type="dataset",
                    commit_message=f"pod code {sha} (option B replicate)")
    rs = {x.path: x.size for x in api.list_repo_tree(repo, repo_type="dataset", path_in_repo="code") if hasattr(x, "size")}
    assert rs.get(f"code/pod_code_{sha}.tgz") == os.path.getsize(tgz), "bundle upload not verified"
    print(f"bundle code/pod_code_{sha}.tgz uploaded and verified ({os.path.getsize(tgz) / 1e6:.1f} MB)")
    return sha


def start_cmd(sha, boot="/root"):
    """Pod boot. All output goes to stdout (RunPod Logs tab) and $BOOT/boot.log; a boot heartbeat uploads that log to
    runs/stage3-rep1/boot_<pod>.log every 60 s, so a failure before the runner starts still leaves a record. Secrets
    are reported by length only (and whether the {{ RUNPOD_SECRET_ }} reference was left unresolved), never by value."""
    repo = "RandallSPQR/social-phase-transitions-llm-cache"
    chk = ("import os\nfor k in ('HF_TOKEN','OPENROUTER_API_KEY'):\n v=os.environ.get(k,'')\n"
           " print('[boot] secret', k, 'len', len(v), 'UNRESOLVED' if v.startswith('{{') else ('MISSING' if not v else 'ok'), flush=True)")
    up = ("import os,time\nfrom huggingface_hub import HfApi\n"
          "HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj=os.environ['BOOT']+'/boot.log', path_in_repo="
          "'runs/stage3-rep1/boot_'+os.environ.get('RUNPOD_POD_ID','local')+'.log', repo_id='%s', repo_type='dataset', "
          "commit_message='boot log')" % repo)
    dl = ("import os\nfrom huggingface_hub import hf_hub_download as d\n"
          "print(d('%s', 'code/pod_code_%s.tgz', repo_type='dataset', local_dir=os.environ['BOOT']+'/code', "
          "token=os.environ['HF_TOKEN']), flush=True)" % (repo, sha))
    import base64
    py = lambda code: "python3 -c \"$(echo %s | base64 -d)\"" % base64.b64encode(code.encode()).decode()
    return (f"export BOOT={boot}; mkdir -p $BOOT; exec > >(tee -a $BOOT/boot.log) 2>&1; "
            "echo \"[boot] $(date -u +%FT%TZ) pod ${RUNPOD_POD_ID:-local}\"; "
            + py(chk) + "; pip install -q huggingface_hub 2>&1 | tail -2; "
            "( while true; do " + py(up) + " > /dev/null 2>&1 || echo '[boot] boot-log upload failed'; sleep 60; done ) & "
            + py(dl) + f" && mkdir -p $BOOT/Social-Phase-Transitions && "
            f"tar xzf $BOOT/code/code/pod_code_{sha}.tgz -C $BOOT/Social-Phase-Transitions && echo '[boot] code ready' && "
            "REPO=$BOOT/Social-Phase-Transitions bash $BOOT/Social-Phase-Transitions/llm/pod_replicate.sh; "
            "echo \"[boot] $(date -u +%FT%TZ) boot sequence ended; idling until removed\"; ${BOOT_EXIT:+exit 0}; sleep infinity")


def launch(a):
    sha = bundle()
    body = {"name": NAME, "imageName": IMAGE, "computeType": "CPU", "cpuFlavorIds": ["cpu3g"], "vcpuCount": 2,
            "containerDiskInGb": 20, "cloudType": "SECURE",
            "env": {"HF_TOKEN": "{{ RUNPOD_SECRET_%s }}" % a.hf_secret, "OPENROUTER_API_KEY": "{{ RUNPOD_SECRET_%s }}" % a.or_secret,
                    "PYTHONUNBUFFERED": "1", "JOB_HOURS": a.job_hours, **({"E2E_TEST": "1"} if a.e2e else {})},
            "dockerEntrypoint": ["bash", "-c"], "dockerStartCmd": [start_cmd(sha)]}
    if a.dry_run:
        print(json.dumps({k: v for k, v in body.items() if k != "dockerStartCmd"}, indent=1)); return
    r = requests.post(f"{REST}/pods", headers=H(), json=body, timeout=60)
    print("create ->", r.status_code, r.text[:500])
    r.raise_for_status()
    p = r.json()
    rec = dict(id=p["id"], created=time.time(), utc=time.strftime("%FT%TZ", time.gmtime()), costPerHr=p.get("costPerHr"), code=sha)
    os.makedirs(os.path.dirname(PODF), exist_ok=True); json.dump(rec, open(PODF, "w"), indent=1)
    print("pod", rec)


def check(a):
    rec = json.load(open(PODF)); pid = rec["id"]
    r = requests.get(f"{REST}/pods/{pid}", headers=H(), timeout=30)
    pod = None if r.status_code == 404 else r.json()
    api, repo = hf()
    try:
        st = json.load(open(api.hf_hub_download(repo, "runs/stage3-rep1/status.json", repo_type="dataset", force_download=True)))
    except Exception as e:
        st = {"error": repr(e)[:200]}
    hrs = (time.time() - rec["created"]) / 3600; rate = rec.get("costPerHr") or (pod or {}).get("costPerHr") or 0
    age = (time.time() - st.get("t", 0)) / 60 if "t" in st else None
    out = dict(pod=pid, pod_state=(pod or {}).get("desiredStatus", "GONE"), uptime_h=round(hrs, 2), pod_spend=round(rate * hrs, 2),
               heartbeat_age_min=None if age is None else round(age, 1), api_spend=st.get("spend_usd"), done=st.get("done"),
               stopped=st.get("stopped"), streams={k: f"{v['chains_done']}/{v['chains_total']}" for k, v in st.get("streams", {}).items()},
               rss_mb=st.get("rss_mb"))
    print(json.dumps(out, indent=1))
    if pod is not None and not a.no_terminate:
        why = None
        if hrs > MAX_H: why = f"uptime {hrs:.1f} h > {MAX_H} h"
        elif rate * hrs > MAX_POD_USD: why = f"pod spend ${rate * hrs:.2f} > ${MAX_POD_USD}"
        elif st.get("done") and age is not None and age > 30: why = "run done and uploaded, pod still up 30+ min"
        if why:
            print("TERMINATING", pid, why)
            print(requests.delete(f"{REST}/pods/{pid}", headers=H(), timeout=30).status_code)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["bundle", "launch", "check"])
    ap.add_argument("--hf-secret"); ap.add_argument("--or-secret"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-terminate", action="store_true"); ap.add_argument("--job-hours", default="40")
    ap.add_argument("--e2e", action="store_true", help="end-to-end test pod: short JOB_HOURS, removes itself after a verified upload even if not done")
    a = ap.parse_args()
    {"bundle": bundle, "launch": lambda: launch(a), "check": lambda: check(a)}[a.what]()
