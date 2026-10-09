"""Wait for an A100 in the owner's network-volume data centre, launch the Amendment 4 pod, and police it.

  RUNPOD_API_KEY=... HF_TOKEN=... python llm/runpod_launch.py --code-rev <git sha>     # runs until the pod is gone
  python llm/runpod_launch.py --dry-run                                                  # one poll, never creates

Phase 1 (poll): every --poll s, read A100 80GB stock in EUR-IS-1 (secure cloud). When stock is listed, or every
  --force-try s regardless, try to create ONE pod (A100 PCIe or SXM, 1 GPU) with the owner's volume mounted at
  /workspace. A failed create costs nothing.
Phase 2 (watchdog): every --check s, terminate THIS pod (by its id; nothing else is ever touched) if it has run longer
  than --max-hours, or if this pod's estimated spend reaches --max-spend: spend = balance drop since launch minus the
  account's other burn rate measured just before launch (the owner's other pods keep running). Exits when the pod is
  gone. The pod itself has a 4 h hard limit and removes itself (llm/pod_run.sh); this is the outside backstop.
Everything is logged to results/llm/logs/runpod_launch.log; the pod id is written to results/llm/logs/pod_id.txt.
"""
import argparse, json, os, sys, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "results/llm/logs/runpod_launch.log")
GQL, REST = "https://api.runpod.io/graphql", "https://rest.runpod.io/v1"
DC, VOLUME = "EUR-IS-1", "u0isne6ams"
A100 = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"     # the image the owner's A100 pods already use
NAME = "spt-amendment4-gemma"


def H():
    return {"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"], "Content-Type": "application/json"}


def log(*a):
    msg = time.strftime("%Y-%m-%d %H:%M:%S UTC ", time.gmtime()) + " ".join(map(str, a))
    print(msg, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    open(LOG, "a").write(msg + "\n")


def gql(q):
    r = requests.post(GQL, headers=H(), json={"query": q}, timeout=30); r.raise_for_status()
    j = r.json()
    if j.get("errors"):
        raise RuntimeError(str(j["errors"])[:300])
    return j["data"]


def account():
    return gql("{ myself { clientBalance currentSpendPerHr } }")["myself"]


def a100_stock():
    q = "{ gpuTypes { id lowestPrice(input:{gpuCount:1, dataCenterId:\"%s\", secureCloud:true}) { uninterruptablePrice stockStatus } } }" % DC
    return {g["id"]: g["lowestPrice"] for g in gql(q)["gpuTypes"] if g["id"] in A100}


def start_cmd(code_rev):
    """Pod boot: fetch the code tarball from the private HF dataset, run pod_run.sh, then idle (never exit: an exited
    container would be restarted by RunPod and restart the job)."""
    py = ("from huggingface_hub import hf_hub_download as d; import os; "
          "print(d('RandallSPQR/social-phase-transitions-llm-cache', 'code/pod_code_%s.tgz', repo_type='dataset', "
          "local_dir='/root/code', token=os.environ['HF_TOKEN']))" % code_rev)
    return ("pip install -q huggingface_hub > /root/boot.log 2>&1; "
            f"python -c \"{py}\" >> /root/boot.log 2>&1 && mkdir -p /root/Social-Phase-Transitions && "
            f"tar xzf /root/code/code/pod_code_{code_rev}.tgz -C /root/Social-Phase-Transitions >> /root/boot.log 2>&1 && "
            "bash /root/Social-Phase-Transitions/llm/pod_run.sh >> /root/boot.log 2>&1; "
            "echo 'job finished; idling until removed' >> /root/boot.log; sleep infinity")


def create(code_rev):
    body = {"name": NAME, "imageName": IMAGE, "gpuTypeIds": A100, "gpuCount": 1, "cloudType": "SECURE",
            "dataCenterIds": [DC], "networkVolumeId": VOLUME, "volumeMountPath": "/workspace",
            "containerDiskInGb": 200, "supportPublicIp": False,
            "env": {"HF_TOKEN": os.environ["HF_TOKEN"], "PYTHONUNBUFFERED": "1"},
            "dockerEntrypoint": ["bash", "-c"], "dockerStartCmd": [start_cmd(code_rev)]}
    r = requests.post(f"{REST}/pods", headers=H(), json=body, timeout=60)
    if r.status_code >= 300:
        return None, f"http {r.status_code}: {r.text[:300]}"
    return r.json(), None


def pod(pid):
    r = requests.get(f"{REST}/pods/{pid}", headers=H(), timeout=30)
    return None if r.status_code == 404 else r.json()


def terminate(pid, why):
    log(f"TERMINATING pod {pid}: {why}")
    r = requests.delete(f"{REST}/pods/{pid}", headers=H(), timeout=30)
    log(f"terminate -> http {r.status_code} {r.text[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-rev"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=int, default=180); ap.add_argument("--force-try", type=int, default=900)
    ap.add_argument("--check", type=int, default=120); ap.add_argument("--max-hours", type=float, default=4.25)
    ap.add_argument("--max-spend", type=float, default=12.0); ap.add_argument("--max-wait-hours", type=float, default=72)
    a = ap.parse_args()
    if a.dry_run:
        log("dry run: stock", a100_stock(), "account", account()); return
    assert a.code_rev, "--code-rev required"
    log(f"poller start: A100 80GB in {DC} with volume {VOLUME}; code {a.code_rev}")
    t_start, last_try, created = time.time(), 0.0, None
    while created is None:
        if time.time() - t_start > a.max_wait_hours * 3600:
            log("gave up waiting for an A100"); return
        try:
            st = a100_stock()
            listed = {k: v for k, v in st.items() if v and v.get("stockStatus")}
            if listed or time.time() - last_try > a.force_try:
                acc = account()
                last_try = time.time()
                res, err = create(a.code_rev)
                if res:
                    created, base = res, acc
                    break
                log(f"create failed (stock {listed or 'none listed'}): {err}")
        except Exception as e:
            log("poll error:", repr(e)[:300])
        time.sleep(a.poll)
    pid, t0 = created["id"], time.time()
    open(os.path.join(ROOT, "results/llm/logs/pod_id.txt"), "w").write(pid + "\n")
    log(f"CREATED pod {pid} ({created.get('gpu', {}) or created.get('machine', {})}, ${created.get('costPerHr')}/h); "
        f"baseline balance ${base['clientBalance']:.2f}, other burn ${base['currentSpendPerHr']:.3f}/h")
    while True:
        time.sleep(a.check)
        try:
            p = pod(pid)
            if p is None:
                log(f"pod {pid} is gone (self-terminated or removed); watchdog done"); return
            hrs = (time.time() - t0) / 3600
            acc = account()
            spent = (base["clientBalance"] - acc["clientBalance"]) - base["currentSpendPerHr"] * hrs
            log(f"pod {pid} {p.get('desiredStatus')} {hrs:.2f} h, est. spend ${spent:.2f}")
            if hrs > a.max_hours:
                terminate(pid, f"ran {hrs:.2f} h > {a.max_hours} h")
            elif spent > a.max_spend:
                terminate(pid, f"estimated spend ${spent:.2f} > ${a.max_spend}")
        except Exception as e:
            log("watchdog error:", repr(e)[:300])


if __name__ == "__main__":
    main()
