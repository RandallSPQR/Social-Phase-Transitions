"""Real-time tracking for the GPU pod: everything is recorded in the private HF dataset while the job runs, so nothing
depends on the pod surviving.

  python llm/pod_tracker.py heartbeat --run RUN --log LOG [--interval 90]   # background loop on the pod
  python llm/pod_tracker.py upload --run RUN --key KEY                        # one model's outputs, verified; exit 1 on failure
  python llm/pod_tracker.py upload-all --run RUN                              # every output on disk, verified

Heartbeat (every --interval s, one commit): runs/RUN/status.json (phase and progress from progress.json written by
local_gemma.py, ETA, GPU utilisation/memory/power, container-disk use, uptime, files uploaded so far) and
runs/RUN/pod.log (the full log). Uploads: runs/RUN/results/<path relative to the repo>, then re-listed and size-checked;
results/llm/logs/uploads.json records what has been verified, and the heartbeat reports it.
"""
import argparse, glob, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ID = "RandallSPQR/social-phase-transitions-llm-cache"
LOGS = os.path.join(ROOT, "results/llm/logs")
PROGRESS = os.path.join(LOGS, "progress.json")
UPLOADS = os.path.join(LOGS, "uploads.json")
PHASE = os.path.join(LOGS, "phase.json")
T_START = float(os.environ.get("POD_T_START", time.time()))


def api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"])


def read_json(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def write_json(p, obj):
    tmp = p + ".tmp"; json.dump(obj, open(tmp, "w")); os.replace(tmp, p)


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip()
        name, util, mu, mt, pw = [x.strip() for x in out.split("\n")[0].split(",")]
        return {"name": name, "util_pct": float(util), "mem_used_gb": float(mu) / 1024, "mem_total_gb": float(mt) / 1024,
                "power_w": float(pw)}
    except Exception as e:
        return {"error": repr(e)[:120]}


def disk():
    try:
        u = shutil.disk_usage("/root" if os.path.isdir("/root") else ROOT)
        return {"used_gb": u.used / 1e9, "total_gb": u.total / 1e9}
    except Exception:
        return {}


def hf_cache_gb():
    root = os.environ.get("HF_HOME", "")
    try:
        return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(root) for f in fs) / 1e9 if root else None
    except Exception:
        return None


def download(mid, tries=3):
    """Fetch a model's weights as its own phase (classic HTTP path; xet is disabled in pod_run.sh)."""
    from huggingface_hub import snapshot_download
    if os.path.isdir(mid):
        print(f"[tracker] local model {mid}", flush=True); return True
    for a in range(1, tries + 1):
        try:
            t = time.time()
            p = snapshot_download(mid, allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*"],
                                  token=os.environ["HF_TOKEN"])
            gb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) / 1e9
            print(f"[tracker] DOWNLOADED {mid}: {gb:.1f} GB in {time.time() - t:.0f} s", flush=True); return True
        except Exception as e:
            print(f"[tracker] download {mid} attempt {a} failed: {repr(e)[:300]}", flush=True); time.sleep(20)
    return False


def status(run, log):
    raw = open(log, errors="replace").read().replace("\r", "\n").splitlines() if os.path.exists(log) else []
    lines = [l for l in raw if l.strip() and "%|" not in l and "it/s]" not in l]     # drop progress-bar noise
    return {"run": run, "t": time.time(), "uptime_s": time.time() - T_START, "phase": read_json(PHASE, {}),
            "progress": read_json(PROGRESS, {}), "gpu": gpu(), "disk": {**disk(), "hf_cache_gb": hf_cache_gb()},
            "uploads": read_json(UPLOADS, {}), "log_tail": lines[-60:], "log_lines": len(lines)}


def heartbeat(run, log, interval):
    from huggingface_hub import CommitOperationAdd
    n = 0
    while True:
        n += 1
        try:
            st = status(run, log); st["beat"] = n
            ops = [CommitOperationAdd(f"runs/{run}/status.json", json.dumps(st, indent=1).encode())]
            if os.path.exists(log):
                ops.append(CommitOperationAdd(f"runs/{run}/pod.log", log))
            api().create_commit(REPO_ID, ops, commit_message=f"heartbeat {run} #{n}", repo_type="dataset")
        except Exception as e:
            print(f"[tracker] heartbeat {n} failed: {repr(e)[:200]}", flush=True)
        time.sleep(interval)


def files_for(key=None):
    pats = ["results/llm/stage1/*/{k}.csv", "results/llm/activations/{k}/*"] if key else \
           ["results/llm/stage1/*/*.csv", "results/llm/activations/*/*", "results/llm/logs/*"]
    out = []
    for p in pats:
        out += [f for f in glob.glob(os.path.join(ROOT, p.format(k=key))) if os.path.isfile(f)]
    return sorted(set(out))


def upload(run, key=None, tries=5):
    """Upload (one key's | all) outputs in one commit and verify every file is listed with the right size."""
    from huggingface_hub import CommitOperationAdd
    files = files_for(key)
    if not files:
        print(f"[tracker] upload {key or 'all'}: no files", flush=True); return False
    for a in range(1, tries + 1):
        try:
            A = api()
            ops = [CommitOperationAdd(f"runs/{run}/results/{os.path.relpath(f, ROOT)}", f) for f in files]
            A.create_commit(REPO_ID, ops, commit_message=f"results {run} {key or 'all'}", repo_type="dataset")
            remote = {x.path: x.size for x in A.list_repo_tree(REPO_ID, path_in_repo=f"runs/{run}/results", recursive=True,
                                                             repo_type="dataset") if hasattr(x, "size")}
            bad = [f for f in files if remote.get(f"runs/{run}/results/{os.path.relpath(f, ROOT)}") != os.path.getsize(f)
                   and not f.endswith(".log")]          # logs keep growing between commit and check
            if bad:
                raise RuntimeError(f"{len(bad)} files missing/size mismatch, e.g. {os.path.relpath(bad[0], ROOT)}")
            u = read_json(UPLOADS, {}); u[key or "all"] = {"t": time.time(), "files": len(files),
                                                          "bytes": sum(os.path.getsize(f) for f in files)}
            write_json(UPLOADS, u)
            print(f"[tracker] UPLOADED {key or 'all'} OK: {len(files)} files, verified", flush=True)
            return True
        except Exception as e:
            print(f"[tracker] upload {key or 'all'} attempt {a} failed: {repr(e)[:200]}", flush=True)
            time.sleep(min(30 * a, 120))
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["heartbeat", "upload", "upload-all", "download"])
    ap.add_argument("--run", required=True); ap.add_argument("--log"); ap.add_argument("--key"); ap.add_argument("--model")
    ap.add_argument("--interval", type=int, default=90)
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    if a.what == "heartbeat":
        heartbeat(a.run, a.log, a.interval)
    elif a.what == "download":
        sys.exit(0 if download(a.model) else 1)
    else:
        sys.exit(0 if upload(a.run, a.key if a.what == "upload" else None) else 1)
