"""Orchestrate Stage 1 main runs for a list of endpoints, sequentially, with the owner's budget rule:
spend = max(sum of per-call costs in the cache, OpenRouter key usage); stop before starting an endpoint if
spend > --cap. Each endpoint: comprehension battery (workhorses + bridge), then the main battery, then fits.

  python ../llm/run_main.py --keys llama-8b --cap 12
  python ../llm/run_main.py --keys llama-70b llama-70b-pa qwen-9b ... --cap 12
"""
import argparse, glob, json, os, subprocess, sys, time
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client
from run_stage1 import run

ARMS_ALL = ["neutral_own", "neutral_noown", "political", "workplace", "neutral_own_w2"]
ARMS_FOR = {"llama-70b-pa": ["neutral_own", "neutral_noown"]}     # provider-robustness replication
NO_COMPREHENSION = {"llama-70b-pa"}


def spend_now():
    cache = 0.0
    for f in glob.glob(os.path.join(client.CACHE_DIR, "*.jsonl")):
        for line in open(f):
            r = json.loads(line); cache += ((r["resp"].get("usage") or {}).get("cost") or 0.0)
    try:
        key = float(requests.get("https://openrouter.ai/api/v1/key", headers=client._headers(), timeout=20).json()["data"]["usage"])
    except Exception:
        key = float("nan")
    return max(cache, key if key == key else 0.0), cache, key


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--cap", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    for key in a.keys:
        s, sc, sk = spend_now()
        log(f"spend check: max(cache ${sc:.3f}, key ${sk:.3f}) = ${s:.3f}  (cap ${a.cap})")
        if s > a.cap:
            log(f"STOP: spend above cap before {key}"); break
        if key not in NO_COMPREHENSION:
            run(key, "comprehension", ["comprehension"], [0], workers=a.workers, budget=a.cap - s, order_subset=False)
        path = run(key, "main", ARMS_FOR.get(key, ARMS_ALL), range(2), workers=a.workers, budget=a.cap - s)
        out = os.path.join(client.ROOT, "results", "llm", "fits", "main")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "analysis.py"), path, "--B", str(a.B), "--out", out],
                       stdout=open(os.path.join(client.ROOT, "results", "llm", "fits", f"log_{key}.txt"), "w"), stderr=subprocess.STDOUT)
        log(f"done {key}")
    s, sc, sk = spend_now(); log(f"final spend: cache ${sc:.3f}, key ${sk:.3f}")
