"""Fetch the simulation unit bundles (per-unit outputs of the long simulations) from the Hugging Face dataset, verify
their SHA-256 against replication/sim_units_manifest.json and unpack them into the repository.

  python replication/fetch_units.py

Fast mode needs these: the simulation scripts skip every unit whose output file exists, so with the bundles in place
they rerun only the analysis. Spot checks recompute a sample of units and require them to be bit-identical to the
bundled ones (replication/replicate.py). Full mode does not need the bundles: it recomputes every unit.
The dataset is public once released; before that, set HF_TOKEN.
"""
import hashlib, json, os, sys, tarfile, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "replication", "sim_units_manifest.json")
DL = os.path.join(ROOT, "replication", "out", "downloads")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch():
    m = json.load(open(MANIFEST)); d = m["dataset"]; os.makedirs(DL, exist_ok=True); bad = 0
    for f in m["files"]:
        p = os.path.join(DL, os.path.basename(f["file"]))
        if not (os.path.exists(p) and sha256(p) == f["sha256"]):
            url = d["url_template"].format(repo_id=d["repo_id"], revision=d["revision"], file=f["file"])
            tok = os.environ.get("HF_TOKEN")
            req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok} if tok else {})
            with urllib.request.urlopen(req) as r, open(p + ".part", "wb") as fo:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fo.write(b)
            os.replace(p + ".part", p)
        if sha256(p) != f["sha256"]:
            print("SHA-256 MISMATCH", f["file"], flush=True); os.remove(p); bad += 1; continue
        with tarfile.open(p) as t:
            members = t.getmembers()
            for x in members:   # only regular files under the declared prefixes, never absolute or parent paths
                if not x.isfile() or x.name.startswith("/") or ".." in x.name.split("/") or \
                        not any(x.name.startswith(q) for q in f["prefixes"]):
                    raise RuntimeError(f"unexpected member {x.name} in {f['file']}")
            t.extractall(ROOT, members=members, filter="data")
        print(f"units: {f['file']} verified, {f['n_files']} files unpacked", flush=True)
    return bad


if __name__ == "__main__":
    sys.exit(1 if fetch() else 0)
