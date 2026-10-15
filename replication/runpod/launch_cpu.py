"""RunPod CPU pod for the full-mode steps not yet measured in full mode (prepared; nothing is launched by default).

  python replication/runpod/launch_cpu.py plan                          # print the pod spec and the steps; no API call
  python replication/runpod/launch_cpu.py volume-check --volume-id ID   # read-only: confirm the volume is the dedicated one
  python replication/runpod/launch_cpu.py launch --volume-id ID --hf-secret NAME [--e2e] --confirm
  python replication/runpod/launch_cpu.py check [--no-terminate]        # check-in: state, uptime, spend, heartbeat age,
                                                                        # steps; removes the pod only on a cap or after
                                                                        # a verified final upload

Rules (CLAUDE.md):
- Only a DEDICATED network volume named "spt-replication…" is mounted. The owner's shared volume is never mounted:
  `launch` reads the volume's name and refuses any other.
- The pod receives only a reference to the owner's RunPod secret ({{ RUNPOD_SECRET_<name> }}) for HF_TOKEN, never a value.
- Heartbeat and per-step uploads go to the HF dataset runs/<RUN>/ (readable from a session) as well as to the volume.
- The pod never removes itself. `check` removes it only after runs/<RUN>/status.json reports final_verified (and 30 min
  have passed), or when a cap is hit (uptime > MAX_H, or pod spend > MAX_POD_USD).
- `--e2e` runs a 4-step end-to-end test of the whole chain (heartbeat, per-step upload, final upload, removal) before
  any paid full run, as CLAUDE.md requires.
- Without --confirm, `launch` prints the request body and exits.
"""
import argparse, json, os, subprocess, sys, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REST = "https://rest.runpod.io/v1"
REPO_ID = "RandallSPQR/social-phase-transitions-llm-cache"
# python:3.13-slim pinned by digest (the same base as the Dockerfile); the pod installs replication/requirements.lock
IMAGE = "python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f"
VCPU, MEM_NOTE = 8, "8 vCPU compute-optimised (the scripts use up to 4 processes; peak RSS 4.6 GB in fast mode)"
STEPS_FULL = ("qc.canary s1.probes s1.figures s2.surrogates s2.sims s3.replay s3.audit k.test k.avg k.followups "
              "r.tests r.replicate amp.stageB conf.api conf.locate conf.step45 conf.gemma substack")
STEPS_E2E = "qc.canary base.tables s1.figures conf.gemma"          # about 2 min of work: proves the chain end to end
JOB_HOURS, JOB_HOURS_E2E = 20, 1
MAX_H, MAX_POD_USD = 24.0, 12.0                                     # check-in caps (the expected run is about 13 h)
PODF = os.path.join(ROOT, "replication", "runpod", "pod.json")      # pod id + launch record (git-ignored)
VOL_PREFIX = "spt-replication"
H = lambda: {"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"], "Content-Type": "application/json"}


def sha():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def bundle(s):
    """git archive HEAD -> HF code/pod_code_<sha>.tgz, verified by size (as llm/runpod_cpu_launch.py does)."""
    from huggingface_hub import HfApi
    assert not subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip(), "commit first"
    tgz = f"/tmp/pod_code_{s}.tgz"
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", tgz, "HEAD"], cwd=ROOT, check=True)
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(path_or_fileobj=tgz, path_in_repo=f"code/pod_code_{s}.tgz", repo_id=REPO_ID, repo_type="dataset",
                    commit_message=f"pod code {s} (replication full-mode remaining steps)")
    sizes = {x.path: getattr(x, "size", None) for x in api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo="code")}
    assert sizes.get(f"code/pod_code_{s}.tgz") == os.path.getsize(tgz), "bundle upload not verified"
    print(f"bundle code/pod_code_{s}.tgz uploaded and verified")


def start_cmd(s, run, steps, job_hours):
    """Pod boot: fetch the code bundle from HF, unpack, run pod_full_remaining.sh. All output to stdout and the volume."""
    dl = ("import os\nfrom huggingface_hub import hf_hub_download as d\n"
          f"print(d('{REPO_ID}', 'code/pod_code_{s}.tgz', repo_type='dataset', local_dir='/root/code', "
          "token=os.environ['HF_TOKEN']), flush=True)")
    import base64
    py = "python3 -c \"$(echo %s | base64 -d)\"" % base64.b64encode(dl.encode()).decode()
    return (f"echo \"[boot] $(date -u +%FT%TZ) pod ${{RUNPOD_POD_ID:-local}}\"; "
            "pip install -q huggingface_hub==1.33.0 && " + py + " && mkdir -p /root/repo && "
            f"tar xzf /root/code/code/pod_code_{s}.tgz -C /root/repo && echo '[boot] code ready' && "
            f"REPO=/root/repo VOL=/workspace RUN={run} JOB_HOURS={job_hours} STEPS='{steps}' "
            "bash /root/repo/replication/runpod/pod_full_remaining.sh; "
            "echo \"[boot] $(date -u +%FT%TZ) ended; idling until removed\"; sleep infinity")


def body(a, s):
    run = f"{'e2e' if a.e2e else 'full-remaining'}-{s}"
    steps, jh = (STEPS_E2E, JOB_HOURS_E2E) if a.e2e else (STEPS_FULL, JOB_HOURS)
    return run, {"name": f"spt-{run}", "imageName": IMAGE, "computeType": "CPU", "cpuFlavorIds": ["cpu3c"],
                 "vcpuCount": VCPU, "containerDiskInGb": 20, "cloudType": "SECURE",
                 "networkVolumeId": a.volume_id, "volumeMountPath": "/workspace",
                 "env": {"HF_TOKEN": "{{ RUNPOD_SECRET_%s }}" % a.hf_secret, "PYTHONUNBUFFERED": "1",
                         "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0", "SPT_OFFLINE": "1"},
                 "dockerEntrypoint": ["bash", "-c"], "dockerStartCmd": [start_cmd(s, run, steps, jh)]}


def volume_check(vid):
    r = requests.get(f"{REST}/networkvolumes/{vid}", headers=H(), timeout=30); r.raise_for_status()
    v = r.json(); name = v.get("name", "")
    print(json.dumps({k: v.get(k) for k in ("id", "name", "size", "dataCenterId")}, indent=1))
    if not name.startswith(VOL_PREFIX):
        sys.exit(f"REFUSED: volume '{name}' is not a dedicated '{VOL_PREFIX}…' volume (the shared volume is off-limits)")
    return v


def plan(a):
    print(f"image   {IMAGE}\ncpu     {MEM_NOTE}\nsteps   {STEPS_FULL}\nsteps (e2e) {STEPS_E2E}\n"
          f"caps    job {JOB_HOURS} h (e2e {JOB_HOURS_E2E} h); check-in removes the pod at uptime > {MAX_H} h or spend > ${MAX_POD_USD}\n"
          f"volume  dedicated '{VOL_PREFIX}…' network volume mounted at /workspace; logs at /workspace/spt/<RUN>/\n"
          f"record  HF {REPO_ID} runs/<RUN>/ (status.json every 90 s, pod.log, steps/*.log, records.json, final.tar.gz)")


def launch(a):
    s = sha(); run, b = body(a, s)
    if not a.confirm:
        print(json.dumps({k: v for k, v in b.items() if k != "dockerStartCmd"}, indent=1))
        print(f"run {run}\n(dry run: nothing created; add --confirm to bundle the code and create the pod)"); return
    volume_check(a.volume_id)
    bundle(s)
    r = requests.post(f"{REST}/pods", headers=H(), json=b, timeout=60); print("create ->", r.status_code, r.text[:400])
    r.raise_for_status(); p = r.json()
    rec = dict(id=p["id"], run=run, created=time.time(), utc=time.strftime("%FT%TZ", time.gmtime()),
               costPerHr=p.get("costPerHr"), code=s, e2e=a.e2e)
    json.dump(rec, open(PODF, "w"), indent=1); print("pod", rec)


def check(a):
    from huggingface_hub import HfApi
    rec = json.load(open(PODF)); pid = rec["id"]
    r = requests.get(f"{REST}/pods/{pid}", headers=H(), timeout=30); pod = None if r.status_code == 404 else r.json()
    api = HfApi(token=os.environ["HF_TOKEN"])
    try:
        st = json.load(open(api.hf_hub_download(REPO_ID, f"runs/{rec['run']}/status.json", repo_type="dataset",
                                                force_download=True)))
    except Exception as e:
        st = {"error": repr(e)[:200]}
    hrs = (time.time() - rec["created"]) / 3600; rate = rec.get("costPerHr") or (pod or {}).get("costPerHr") or 0
    age = (time.time() - st["t"]) / 60 if "t" in st else None
    out = dict(pod=pid, run=rec["run"], state=(pod or {}).get("desiredStatus", "GONE"), uptime_h=round(hrs, 2),
               spend=round(rate * hrs, 2), heartbeat_age_min=None if age is None else round(age, 1),
               steps_ok=st.get("steps_ok"), steps_failed=st.get("steps_failed"), uploaded=st.get("steps_uploaded"),
               done=st.get("done"), final_verified=st.get("final_verified"), vol_free_gb=st.get("vol_free_gb"))
    print(json.dumps(out, indent=1))
    if pod is not None and not a.no_terminate:
        why = None
        if hrs > MAX_H: why = f"uptime {hrs:.1f} h > {MAX_H} h"
        elif rate * hrs > MAX_POD_USD: why = f"pod spend ${rate * hrs:.2f} > ${MAX_POD_USD}"
        elif st.get("final_verified") and age is not None and age > 30: why = "final upload verified; pod idle 30+ min"
        if why:
            print("REMOVING", pid, "-", why)
            print(requests.delete(f"{REST}/pods/{pid}", headers=H(), timeout=30).status_code)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["plan", "volume-check", "launch", "check"])
    ap.add_argument("--volume-id"); ap.add_argument("--hf-secret", default="HF_TOKEN")
    ap.add_argument("--e2e", action="store_true"); ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--no-terminate", action="store_true")
    a = ap.parse_args()
    if a.what == "plan":
        plan(a)
    elif a.what == "volume-check":
        volume_check(a.volume_id)
    elif a.what == "launch":
        assert a.volume_id, "--volume-id of the dedicated spt-replication volume is required"
        launch(a)
    else:
        check(a)
