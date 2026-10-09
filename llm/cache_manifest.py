"""Build results/llm/cache_manifest.json: SHA-256, size and record count of every local cache snapshot
(results/llm/cache/*.jsonl.gz), plus the private Hugging Face dataset they are published to (snapshots are not in git).
Run after llm/snapshot_cache.py.   python llm/cache_manifest.py [--repo OWNER/NAME] [--revision main]"""
import argparse, glob, gzip, hashlib, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="RandallSPQR/social-phase-transitions-llm-cache")
ap.add_argument("--revision", default="main"); a = ap.parse_args()
files = []
for p in sorted(glob.glob(os.path.join(ROOT, "results/llm/cache/*.jsonl.gz"))):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    n = sum(1 for _ in gzip.open(p, "rt"))
    files.append({"file": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": h, "records": n})
m = {"dataset": {"host": "huggingface.co", "repo_type": "dataset", "repo_id": a.repo, "revision": a.revision,
                 "url_template": "https://huggingface.co/datasets/{repo_id}/resolve/{revision}/cache/{file}"},
     "record_schema": {"key": "sha256 of request body + replicate index", "rep": "replicate index", "body": "OpenRouter request body (no headers, no credentials)",
                       "t": "unix time", "resp": "compact response: id, provider, model, content, reasoning, finish, first-token top logprobs, usage"},
     "files": files, "total_records": sum(f["records"] for f in files), "total_bytes": sum(f["bytes"] for f in files)}
json.dump(m, open(os.path.join(ROOT, "results/llm/cache_manifest.json"), "w"), indent=1)
print(f"{len(files)} files, {m['total_records']} records, {m['total_bytes']/1e6:.1f} MB -> results/llm/cache_manifest.json")
