"""Download the response-cache snapshots listed in results/llm/cache_manifest.json from Hugging Face and verify
their SHA-256. With the cache in place every Stage 1 rerun is free (no API calls).
  python llm/fetch_cache.py [OUT_DIR]  # public dataset; default OUT_DIR = results/llm/cache
  HF_TOKEN=... python llm/fetch_cache.py   # private/gated dataset"""
import hashlib, json, os, sys, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = json.load(open(os.path.join(ROOT, "results/llm/cache_manifest.json")))
d = m["dataset"]; out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results/llm/cache"); os.makedirs(out, exist_ok=True)
bad = 0
for f in m["files"]:
    p = os.path.join(out, f["file"])
    if os.path.exists(p) and hashlib.sha256(open(p, "rb").read()).hexdigest() == f["sha256"]:
        print("ok (present)", f["file"]); continue
    url = d["url_template"].format(repo_id=d["repo_id"], revision=d["revision"], file=f["file"])
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {})
    with urllib.request.urlopen(req) as r, open(p, "wb") as fo:
        fo.write(r.read())
    ok = hashlib.sha256(open(p, "rb").read()).hexdigest() == f["sha256"]
    bad += not ok; print(("ok " if ok else "SHA-256 MISMATCH ") + f["file"])
sys.exit(1 if bad else 0)
