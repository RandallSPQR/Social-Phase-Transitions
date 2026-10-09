"""Upload the cache snapshots + manifest to the Hugging Face dataset named in the manifest (needs HF_TOKEN with
write access and network access to huggingface.co). Verifies remote SHA-256 by re-download afterwards.
  HF_TOKEN=... python llm/upload_cache_hf.py          # private dataset (owner decision 2026-10-09)"""
import json, os, subprocess, sys
from huggingface_hub import HfApi
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = json.load(open(os.path.join(ROOT, "results/llm/cache_manifest.json"))); repo = m["dataset"]["repo_id"]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
assert api.repo_info(repo, repo_type="dataset").private, "dataset must be private"
api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=os.path.join(ROOT, "results/llm/cache"),
                  path_in_repo="cache", allow_patterns=["*.jsonl.gz"])
api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=os.path.join(ROOT, "results/llm/cache_manifest.json"),
                path_in_repo="cache_manifest.json")
# verify: re-download every file into a fresh temp dir and compare SHA-256 with the manifest
import tempfile
tmp = tempfile.mkdtemp()
rc = subprocess.run([sys.executable, os.path.join(ROOT, "llm/fetch_cache.py"), tmp], env=dict(os.environ)).returncode
print("remote verification", "PASSED" if rc == 0 else "FAILED"); sys.exit(rc)
