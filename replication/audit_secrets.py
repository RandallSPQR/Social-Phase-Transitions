"""Scan for credential-like strings before anything is made public. Prints only file names and pattern NAMES, never
the matched text.

  python replication/audit_secrets.py git        # every blob in the git history (all refs)
  HF_TOKEN=... python replication/audit_secrets.py hf [--skip-npy]   # every file of the Hugging Face dataset
                                                                      # (archives are opened, .gz decompressed)
"""
import gzip, io, json, os, re, subprocess, sys, tarfile, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "RandallSPQR/social-phase-transitions-llm-cache"
PATTERNS = {
    "hf_token": rb"hf_[A-Za-z0-9]{30,}",
    "openrouter_key": rb"sk-or-v1-[0-9a-f]{20,}",
    "openai_like_key": rb"sk-[A-Za-z0-9_-]{30,}",
    "anthropic_key": rb"sk-ant-[A-Za-z0-9_-]{20,}",
    "runpod_key": rb"rpa_[A-Za-z0-9]{20,}",
    "aws_key": rb"AKIA[0-9A-Z]{16}",
    "github_token": rb"gh[pousr]_[A-Za-z0-9]{30,}",
    "bearer_header": rb"Bearer [A-Za-z0-9_.-]{20,}",
    "private_key": rb"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "key_assignment": rb"(OPENROUTER_API_KEY|HF_TOKEN|RUNPOD_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY)\s*[=:]\s*['\"]?[A-Za-z0-9_-]{16,}",
    "openrouter_generation_id": rb"gen-[0-9]{10}-[A-Za-z0-9]{10,}",
    "email_address": rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[a-z]{2,}",
}
# addresses that are expected and harmless in this repository (commit trailers, licence texts)
EMAIL_OK = {b"noreply@anthropic.com"}


def scan(data):
    out = {}
    for k, p in PATTERNS.items():
        m = re.findall(p, data)
        if k == "email_address":
            m = [x for x in re.findall(p, data) if x not in EMAIL_OK]
        if m:
            out[k] = len(m)
    return out


def git():
    objs = subprocess.run(["git", "rev-list", "--objects", "--all"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
    hits, n = {}, 0
    for line in objs:
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        sha, path = parts
        t = subprocess.run(["git", "cat-file", "-t", sha], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        if t != "blob":
            continue
        data = subprocess.run(["git", "cat-file", "-p", sha], cwd=ROOT, capture_output=True).stdout
        if path.endswith(".gz"):
            try:
                data = gzip.decompress(data)
            except Exception:
                pass
        n += 1
        h = scan(data)
        if h:
            hits[f"{path} ({sha[:10]})"] = h
    msgs = subprocess.run(["git", "log", "--all", "--format=%H%n%B"], cwd=ROOT, capture_output=True).stdout
    h = scan(msgs)
    if h:
        hits["<commit messages>"] = h
    return n, hits


def get(url, hdr, tries=5):
    import time
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=300).read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def hf(skip_npy):
    tok = os.environ.get("HF_TOKEN")
    hdr = {"Authorization": "Bearer " + tok} if tok else {}
    tree = json.load(urllib.request.urlopen(urllib.request.Request(
        f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true", headers=hdr)))
    files = [x["path"] for x in tree if x["type"] == "file" and not (skip_npy and x["path"].endswith(".npy"))]
    hits = {}
    for p in files:
        data = get(f"https://huggingface.co/datasets/{REPO}/resolve/main/{p}", hdr)
        try:
            if p.endswith((".tgz", ".tar.gz", ".tar")):
                with tarfile.open(fileobj=io.BytesIO(data)) as t:
                    names = []
                    for m in t.getmembers():
                        names.append(m.name)
                        if m.isfile():
                            d = t.extractfile(m).read()
                            if m.name.endswith(".gz"):
                                try:
                                    d = gzip.decompress(d)
                                except Exception:
                                    pass
                            h = scan(d)
                            if h:
                                hits[f"{p}::{m.name}"] = h
                    h = scan("\n".join(names).encode())
                    if h:
                        hits[f"{p}::<member names>"] = h
                continue
            if p.endswith(".gz"):
                data = gzip.decompress(data)
        except Exception as e:
            hits[p] = {"unreadable_" + type(e).__name__: 1}; continue
        h = scan(data)
        if h:
            hits[p] = h
    return len(files), hits


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "git"
    n, hits = git() if what == "git" else hf("--skip-npy" in sys.argv)
    print(json.dumps(hits, indent=1))
    print(f"{what}: {n} files scanned, {len(hits)} with matches (pattern names and counts only)")
