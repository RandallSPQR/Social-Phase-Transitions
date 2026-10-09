"""Session-side relay for a tracked run: reads the pod's heartbeat from the private HF dataset and the pod's state from
RunPod, and writes one dashboard document per update to results/llm/logs/live/<run>.json.

  python llm/run_relay.py --run RUN [--pod POD_ID] [--every 60] [--emit 180] [--cap-hours 4.25] [--cap-spend 12]

Prints one line ("LIVE ...") whenever the document should be pushed to the dashboard: every --emit seconds, and at
once on a phase change, a new verified upload, an alert, or the end of the run. Alerts: heartbeat older than 5 min,
pod gone without a 'done' phase, upload failure, spend or time near a cap.
"""
import argparse, json, os, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ID = "RandallSPQR/social-phase-transitions-llm-cache"
REST = "https://rest.runpod.io/v1"
MODELS = ["gemma-26b-local", "gemma-31b-local"]


def hf_status(run):
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/runs/{run}/status.json"
    r = requests.get(url, headers={"Authorization": "Bearer " + os.environ["HF_TOKEN"]}, timeout=30,
                     params={"t": int(time.time())})
    return r.json() if r.status_code == 200 else None


def pod_state(pod):
    if not pod:
        return None
    r = requests.get(f"{REST}/pods/{pod}", headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"]}, timeout=30)
    return {"gone": True} if r.status_code == 404 else r.json()


def build(run, pod, st, ps, hist, a, t_start):
    now = time.time()
    prog = (st or {}).get("progress") or {}
    phase = (st or {}).get("phase") or {}
    uploads = (st or {}).get("uploads") or {}
    hb_age = now - st["t"] if st else None
    if ps and not ps.get("gone") and ps.get("lastStartedAt"):
        import calendar
        t_start = calendar.timegm(time.strptime(ps["lastStartedAt"][:19], "%Y-%m-%d %H:%M:%S"))
    up_h = (now - t_start) / 3600 if t_start else (st or {}).get("uptime_s", 0) / 3600
    rate = float((ps or {}).get("costPerHr") or a.rate)
    spend = rate * up_h
    # per-model state and an overall fraction (main runs dominate; smoke counted as 5% of a model)
    models, frac = [], 0.0
    for k in MODELS:
        m = {"key": k, "smoke": "done" if f"{k}-smoke" in uploads else "pending",
             "main": "uploaded" if k in uploads else "pending", "frac": 1.0 if k in uploads else 0.0}
        if prog.get("key") in (k, f"{k}-smoke") and prog.get("stage") not in ("done",):
            which = "smoke" if prog["key"].endswith("-smoke") else "main"
            m[which] = prog.get("stage", "running")
            if which == "main" and prog.get("total"):
                m["frac"] = prog.get("done", 0) / prog["total"]
        frac += (0.05 if m["smoke"] == "done" else 0) + 0.95 * m["frac"]
        models.append(m)
    frac /= len(MODELS)
    alerts = []
    if hb_age is not None and hb_age > 300 and phase.get("phase") != "done":
        alerts.append(f"no heartbeat for {hb_age/60:.0f} min")
    if ps and ps.get("gone") and phase.get("phase") != "done":
        alerts.append("pod is gone but the run did not report 'done'")
    if phase.get("phase") == "upload_failed":
        alerts.append("final upload failed; pod left up for the watchdog")
    if up_h > 0.85 * a.cap_hours:
        alerts.append(f"time {up_h:.2f} h of {a.cap_hours} h cap")
    if spend > 0.85 * a.cap_spend:
        alerts.append(f"spend ${spend:.2f} of ${a.cap_spend} cap")
    g = (st or {}).get("gpu") or {}
    if st:
        hist.append({"t": now, "frac": round(frac, 4), "util": g.get("util_pct"), "mem": g.get("mem_used_gb"),
                     "spend": round(spend, 3)})
        del hist[:-400]
    return {
        "run": run, "updated_at": now,
        "pod": {"id": pod, "status": "gone" if (ps or {}).get("gone") else (ps or {}).get("desiredStatus"),
                "gpu_type": ((ps or {}).get("machine") or {}).get("gpuTypeId") or g.get("name"),
                "cost_per_hr": rate, "uptime_h": round(up_h, 3), "spend": round(spend, 2),
                "cap_hours": a.cap_hours, "cap_spend": a.cap_spend},
        "phase": phase.get("phase"), "phase_detail": phase.get("detail"),
        "progress": {k: prog.get(k) for k in ("key", "stage", "rep", "batch", "n_batches", "done", "total", "eta_s",
                                              "elapsed_s", "n_prompts")},
        "overall_frac": round(frac, 4), "models": models,
        "gpu": g, "disk": (st or {}).get("disk"), "uploads": uploads,
        "heartbeat": {"beat": (st or {}).get("beat"), "age_s": round(hb_age) if hb_age is not None else None},
        "log_tail": ((st or {}).get("log_tail") or [])[-30:], "alerts": alerts, "history": hist,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--pod")
    ap.add_argument("--every", type=int, default=60); ap.add_argument("--emit", type=int, default=180)
    ap.add_argument("--cap-hours", type=float, default=4.25); ap.add_argument("--cap-spend", type=float, default=12.0)
    ap.add_argument("--rate", type=float, default=0.0); ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    out = os.path.join(ROOT, "results/llm/logs/live", f"{a.run}.json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    hist, last_emit, last_key, t_start = [], 0.0, None, None
    while True:
        try:
            st, ps = hf_status(a.run), pod_state(a.pod or a.run)
            if t_start is None and st:
                t_start = st["t"] - st.get("uptime_s", 0)
            doc = build(a.run, a.pod or a.run, st, ps, hist, a, t_start)
            json.dump(doc, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out)
            key = (doc["phase"], doc["progress"].get("key"), tuple(sorted(doc["uploads"])), tuple(doc["alerts"]))
            if key != last_key or time.time() - last_emit >= a.emit or a.once:
                p = doc["progress"]
                print(f"LIVE {a.run} phase={doc['phase']} {p.get('key')} {p.get('stage')} "
                      f"{p.get('done')}/{p.get('total')} overall={doc['overall_frac']:.0%} "
                      f"eta={(p.get('eta_s') or 0)/60:.0f}m gpu={doc['gpu'].get('util_pct')}% "
                      f"spend=${doc['pod']['spend']:.2f} hb={doc['heartbeat']['age_s']}s "
                      f"uploads={sorted(doc['uploads'])} alerts={doc['alerts']}", flush=True)
                last_emit, last_key = time.time(), key
            if a.once or ((ps or {}).get("gone") and doc["phase"] in ("done", "upload_failed")):
                print(f"LIVE-END {a.run} phase={doc['phase']}", flush=True); return
        except Exception as e:
            print(f"LIVE-ERROR {repr(e)[:200]}", flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
