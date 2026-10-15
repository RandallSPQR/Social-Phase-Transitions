"""Fetch the Gemma-4 activations used by the exploratory probes (full mode only, about 820 MB) into
replication/out/activations/<key>/ and verify each file's SHA-256 against replication/activations_manifest.json.

  python replication/fetch_activations.py
"""
import hashlib, json, os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "replication", "out", "activations")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch():
    m = json.load(open(os.path.join(ROOT, "replication", "activations_manifest.json")))
    tok = os.environ.get("HF_TOKEN"); bad = 0
    for path, v in m["files"].items():
        key, name = path.rstrip("/").split("/")[-2:]
        p = os.path.join(DEST, key, name); os.makedirs(os.path.dirname(p), exist_ok=True)
        if not (os.path.exists(p) and sha256(p) == v["sha256"]):
            url = f"https://huggingface.co/datasets/{m['dataset']}/resolve/main/{path}"
            req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok} if tok else {})
            with urllib.request.urlopen(req) as r, open(p + ".part", "wb") as fo:
                for b in iter(lambda: r.read(1 << 20), b""):
                    fo.write(b)
            os.replace(p + ".part", p)
        ok = sha256(p) == v["sha256"]
        bad += not ok
        print(("ok " if ok else "SHA-256 MISMATCH ") + f"{key}/{name}", flush=True)
    return bad


if __name__ == "__main__":
    sys.exit(1 if fetch() else 0)
