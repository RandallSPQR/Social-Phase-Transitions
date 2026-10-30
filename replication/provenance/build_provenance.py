"""Pre-registration provenance: tie every pre-registration document to its git commit and to independent, server-side
timestamps, so the order "plan committed → plan on GitHub → data collected" can be checked without trusting the
author's clock.

  python replication/provenance/build_provenance.py logs      # export GitHub's push log and HF's commit log (read-only)
  python replication/provenance/build_provenance.py table     # build PREREG_PROVENANCE.md + provenance.json
  python replication/provenance/build_provenance.py pack OUT  # verification pack of the ORIGINAL history (see below)

Evidence used, and who attests it:
- git commit hashes (content-addressed). Each commit hash fixes the exact text of every file in it and the whole history
  before it. Author/committer dates inside a commit are set by the author's machine and are NOT independent.
- GitHub's push log (repos/{owner}/{repo}/activity): GitHub's server time of every push and the commit the branch then
  pointed to. Independent of the author's clock. Exported to github_push_log.json; anyone with read access to the
  repository can re-query it live.
- Hugging Face's commit log of the dataset: HF's server time of every upload (code bundles named by commit, GPU/CPU run
  heartbeats). Exported to hf_commit_log.json.
- Call timestamps in the response cache (`t`): the client's clock, NOT independent; listed for completeness. OpenRouter's
  own activity log (owner's account) can corroborate them.

The verification pack (`pack`): every commit and tree object of the original history plus every blob except the
third-party PDFs (papers/) and the old cache files (results/llm/cache/, which contain account-linked generation ids).
Commit and tree hashes do not need the omitted blobs, so anyone can (1) recompute and confirm every original commit
hash, (2) read every pre-registration document exactly as committed, and (3) match those hashes against the exported
push log.
"""
import datetime as dt, glob, gzip, hashlib, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "replication", "provenance")
GH_REPO = "RandallSPQR/Social-Phase-Transitions"
HF_REPO = "RandallSPQR/social-phase-transitions-llm-cache"
EXCLUDE = ("papers/", "results/llm/cache/")

# (commit, document, what it fixed, data it governs: list of evidence keys)
PREREGS = [
    ("e9c7e47", "llm/PREREG.md", "Stage 1 battery, hypotheses H1–H5", ["s1_pilot", "s1_main"]),
    ("8c0ab35", "llm/PREREG.md (Amendment 1)", "H3 test, estimand", ["s1_pilot", "s1_main"]),
    ("9bf6c28", "llm/PREREG.md (Amendment 2)", "answer instruction v2 after the pilot", ["s1_main"]),
    ("483decc", "llm/PREREG.md (Amendment 3)", "comprehension gate, temperature check, Stage 3 option order",
     ["s1_comprehension", "s1_main", "s1_temp"]),
    ("ee18890", "llm/PREREG.md (Amendment 4)", "self-hosted bf16 Gemma-4", ["hf:e3zan9ygnmqe45"]),
    ("7705d12", "llm/PREREG.md (Amendment 5)", "rescue rerun of Gemma-26B at batch size 1", ["hf:4p4a463q78clp0"]),
    ("878724e", "llm/STAGE3_PREDICTIONS.md", "Stage 3 predictions", ["stage3"]),
    ("de0ff08", "llm/STAGE3_PREDICTIONS.md (addendum)", "Stage 3 pre-data addendum", ["stage3"]),
    ("4c3924a", "llm/STAGE3_AUDIT_PLAN.md", "harness audit", ["file:results/llm/STAGE3_AUDIT_RESULTS.md"]),
    ("fe65958", "llm/KERNEL_TEST_PLAN.md", "three-kernel test, outcome map", ["kavg"]),
    ("f6b1922", "llm/KERNEL_TEST_PLAN.md (amendment)", "K_avg options", ["kavg"]),
    ("9c759f7", "llm/RESIDUAL_TEST_PLAN.md", "heterogeneity test, replicate design", ["stage3-rep1"]),
    ("b27d83a", "llm/RESIDUAL_TEST_PLAN.md (amendment 1)", "power, sentinels", ["stage3-rep1"]),
    ("93bd950", "llm/RESIDUAL_TEST_PLAN.md (amendment 2)", "no-contacts predictions", ["stage3-rep1"]),
    ("f06db97", "llm/REPLAY_AUDIT_PLAN.md", "replay audit, combined effect", ["file:results/llm/stage3_rep1/persistence_effect_combined.json"]),
    ("18f619e", "llm/AMPLIFICATION_PLAN.md", "Stage B amplification test", ["file:results/llm/amplification/verdict.json"]),
    ("eef27e6", "llm/AMPLIFICATION_PLAN.md (amendment 1)", "T_c relocation rule", ["file:results/llm/amplification/verdict.json"]),
    ("76aad78", "llm/AMPLIFICATION_CONFIRM_PLAN.md", "corrected confirmatory test", ["confirm", "hf:yor638ho767zok"]),
    ("32daeac", "llm/AMPLIFICATION_CONFIRM_PLAN.md (amendment 1)", "model selection", ["confirm", "hf:yor638ho767zok"]),
    ("0b69998", "llm/AMPLIFICATION_CONFIRM_PLAN.md (amendment 2)", "guardrail re-pin of candidates 1 and 3",
     ["model:deepseek/deepseek-v4.1-flash", "model:deepseek/deepseek-v4-pro-0813"]),
    ("ef5674d", "llm/AMPLIFICATION_CONFIRM_PLAN.md (amendment 3)", "negative control, labels", ["file:results/llm/confirm/results/step4.json"]),
    ("f9d5e0c", "llm/GEMMA_MECHANISM_PLAN.md", "Gemma mechanism test (exploratory)", ["file:results/llm/confirm/gemma_mechanism/summary.json"]),
]


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def iso(t):
    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- logs (read-only export)
def export_logs():
    import re, urllib.parse
    rows, seen, q = [], set(), "per_page=100&direction=asc"
    for _ in range(50):
        out = subprocess.run(["gh", "api", "-i", f"repos/{GH_REPO}/activity?{q}"], capture_output=True, text=True,
                             timeout=60, check=True).stdout
        sep = "\r\n\r\n" if "\r\n\r\n" in out else "\n\n"
        head, _, body = out.partition(sep); data = json.loads(body)
        for d in data:
            if d["id"] not in seen:
                seen.add(d["id"]); rows.append({k: d.get(k) for k in ("id", "timestamp", "activity_type", "ref", "before", "after")})
        m = re.search(r'<([^>]+)>;\s*rel="next"', head)
        if not m or not data:
            break
        after = urllib.parse.parse_qs(urllib.parse.urlparse(m.group(1)).query).get("after")
        if not after:
            break
        q = "per_page=100&direction=asc&after=" + after[0]
    json.dump(dict(source=f"GET https://api.github.com/repos/{GH_REPO}/activity (server timestamps)",
                   exported_utc=iso(dt.datetime.now(dt.UTC).timestamp()), events=rows),
              open(os.path.join(HERE, "github_push_log.json"), "w"), indent=1)
    from huggingface_hub import HfApi
    cs = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_commits(HF_REPO, repo_type="dataset")
    json.dump(dict(source=f"Hugging Face commit log of datasets/{HF_REPO} (server timestamps)",
                   exported_utc=iso(dt.datetime.now(dt.UTC).timestamp()),
                   commits=[dict(id=c.commit_id, created_at=c.created_at.isoformat(), title=c.title) for c in reversed(cs)]),
              open(os.path.join(HERE, "hf_commit_log.json"), "w"), indent=1)
    print(f"github_push_log.json: {len(rows)} events; hf_commit_log.json: {len(cs)} commits")


# ---------------------------------------------------------------- evidence
def first_push(commit, events):
    """Earliest GitHub push (server time) after which a branch contained `commit`."""
    for e in sorted(events, key=lambda e: e["timestamp"]):
        if not e.get("after") or set(e["after"]) == {"0"}:
            continue
        if subprocess.run(["git", "merge-base", "--is-ancestor", commit, e["after"]], cwd=ROOT).returncode == 0:
            return e
    return None


def stage1_first_calls():
    """Exact earliest call of the Stage 1 pilot, comprehension battery and main battery: every committed Stage 1 CSV
    from an API endpoint is rebuilt request by request (replication/stage1_rematerialise.py) and each request is looked
    up in the cache, whose record carries the call time (client clock)."""
    sys.path.insert(0, os.path.join(ROOT, "replication")); sys.path.insert(0, os.path.join(ROOT, "llm"))
    import tempfile, client, run_stage1, stage1_rematerialise as SR
    run_stage1.OUT = tempfile.mkdtemp(prefix="prov_s1_")
    first = {}; real = client.chat
    def chat(body, rep=0, **kw):
        rec = real(body, rep=rep, **kw); ph = chat.phase
        first[ph] = rec["t"] if ph not in first else min(first[ph], rec["t"])
        return rec
    client.chat = chat
    try:
        for j in SR.jobs():
            j.pop("n_committed"); reps = j.pop("reps")
            chat.phase = {"main": "s1_main", "comprehension": "s1_comprehension", "pilot": "s1_pilot"}[j["phase"]]
            run_stage1.run(j["key"], j["phase"], j["arms"], reps, j["samples"], j["temperature"], j["kmax"], 8, None,
                           tag_override=j["tag_override"], out_name=j["out_name"])
    finally:
        client.chat = real
    return first


def cache_first_calls():
    """Earliest cache call per data family (client clock)."""
    marks = {"stage3": lambda r: isinstance(r, str) and r.startswith("stage3:"),
             "stage3-rep1": lambda r: isinstance(r, str) and r.startswith("stage3-rep1"),
             "kavg": lambda r: isinstance(r, str) and (r.startswith("kavg-") or r.startswith("kavg:")),
             "confirm": lambda r: isinstance(r, str) and r.startswith("confirm"),
             "s1_temp": lambda r: isinstance(r, int) and r >= 3000}
    models = {d[6:] for _, _, _, ds in PREREGS for d in ds if d.startswith("model:")}
    first = {}
    for f in glob.glob(os.path.join(ROOT, "results", "llm", "cache", "*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            r = json.loads(line); rep, t = r.get("rep"), r["t"]
            keys = [k for k, fn in marks.items() if fn(rep)]
            m = (r.get("body") or {}).get("model")
            if m in models and isinstance(rep, str) and rep.startswith("confirm"):
                keys.append("model:" + m)
            for k in keys:
                first[k] = t if k not in first else min(first[k], t)
    os.environ["SPT_OFFLINE"] = "1"
    first.update(stage1_first_calls())
    return first


def table():
    gh = json.load(open(os.path.join(HERE, "github_push_log.json")))["events"]
    hf = json.load(open(os.path.join(HERE, "hf_commit_log.json")))["commits"]
    calls = cache_first_calls()
    rows = []
    for sha, doc, what, data in PREREGS:
        full = git("rev-parse", sha)
        path = doc.split(" ")[0]
        blob = git("rev-parse", f"{sha}:{path}")
        text = subprocess.run(["git", "show", f"{sha}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
        p = first_push(full, gh)
        ev = []
        for d in data:
            if d.startswith("hf:"):
                run = d[3:]; m = [c for c in hf if run in c["title"]]
                if m:
                    ev.append(dict(kind="HF run upload (server)", what=f"first upload of run {run}",
                                   time=min(c["created_at"] for c in m)[:19] + "Z"))
            elif d.startswith("file:"):
                fpath = d[5:]
                adds = git("log", "--diff-filter=A", "--format=%H", "--", fpath).split()
                if adds:
                    q = first_push(adds[-1], gh)
                    ev.append(dict(kind="GitHub push (server)", what=f"first push of {fpath} ({adds[-1][:7]})",
                                   time=q["timestamp"] if q else None))
            else:
                if calls.get(d):
                    ev.append(dict(kind="first API call (client clock)", what=d.replace("model:", "calls to "),
                                   time=iso(calls[d])))
        pushed = p["timestamp"] if p else None
        ok = pushed is not None and all(e["time"] and e["time"] > pushed for e in ev)
        rows.append(dict(commit=full, short=sha, document=doc, fixes=what, committed=git("show", "-s", "--format=%cI", sha),
                         github_first_push=pushed, github_push_event=p and p["id"], file_git_blob=blob,
                         file_sha256=hashlib.sha256(text).hexdigest(), data=ev, plan_pushed_before_data=ok))
    json.dump(dict(generated_utc=iso(dt.datetime.now(dt.UTC).timestamp()), rows=rows),
              open(os.path.join(HERE, "provenance.json"), "w"), indent=1)
    L = ["# Pre-registration provenance", "",
         "Generated by `replication/provenance/build_provenance.py`. For every pre-registration document: the commit that",
         "fixed it, the first time GitHub's servers received that commit, and the earliest evidence of the data it",
         "governs. Server timestamps (GitHub, Hugging Face) are independent of the author's clock; API-call times are",
         "the client's clock and are marked as such. `file sha256` is the SHA-256 of the document exactly as committed.", "",
         "| commit | document | fixes | on GitHub (server) | earliest data | plan first? | file sha256 |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        data = "; ".join(f"{e['time']} ({e['kind']}: {e['what']})" for e in r["data"]) or "—"
        L.append(f"| `{r['short']}` | `{r['document']}` | {r['fixes']} | {r['github_first_push']} | {data} | "
                 f"{'yes' if r['plan_pushed_before_data'] else '**check**'} | `{r['file_sha256'][:16]}…` |")
    L += ["", "Full hashes, push-event ids and blob ids: `provenance.json`. Logs: `github_push_log.json`, `hf_commit_log.json`.",
          "",
          "## Which clocks are independent",
          "",
          "- **GitHub push times** come from GitHub's repository activity log (server side). Each event names the commit",
          "  the branch pointed to after the push; a pre-registration counts as pushed at the first event whose commit has",
          "  it as an ancestor. Commit dates inside git are the author's and are not used.",
          "- **Hugging Face upload times** come from the dataset's commit log (server side).",
          "- **API-call times** (Stage 1, Stage 3, the kernel test, the replicate, the confirmatory calls) are the client's",
          "  clock, recorded in the response cache. They agree with the server logs wherever both exist, but they are not",
          "  independent. Four rows have margins under a minute between the GitHub push and the first call:",
          "  `483decc` (17 s), `de0ff08` (31 s), `93bd950` (33 s) and `0b69998` (7 s). In each case the run was started",
          "  right after the push; a client clock more than that far fast would reverse the order.",
          "",
          "## How to verify",
          "",
          "The original repository and the original Hugging Face dataset stay private: their history holds third-party",
          "PDFs (and, in git, account-linked generation ids) that are not redistributed. The release carries instead:",
          "",
          "1. `original-history-<id>.pack`: every commit, tree and blob of the original repository except the blobs under",
          "   `papers/` and `results/llm/cache/`. Commit and tree ids are hashes of their content, so every commit id cited",
          "   in the paper and the plans can be checked from the pack without the omitted blobs.",
          "2. `python replication/provenance/verify_pack.py DIR` unpacks it into an empty repository (git recomputes every",
          "   object hash), checks that no PDF or generation id is in it, and checks every row above: the document's",
          "   SHA-256 at that commit, and that the commit is contained in the GitHub push recorded for it.",
          "3. The exported logs are the author's copies. Their server origin can be checked by a reviewer given read access",
          "   to the private originals (GitHub: `GET /repos/randallspqr/social-phase-transitions/activity`; Hugging Face:",
          "   the dataset's commit history), on request."]
    open(os.path.join(HERE, "PREREG_PROVENANCE.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- verification pack
# The repository's root commit was made by GitHub's "create repository" page and carries the owner's personal email
# address as author; it holds only an empty README. It is left out of the pack, and its children are marked as
# shallow boundaries in the verifying repository (verify_pack.py). Every other commit id stays verifiable.
OMIT_COMMITS = {"5e990d78f2363933163b604dc0a85d7ff07b3a23": "root commit created by GitHub; carries a personal email"}


def pack(out):
    os.makedirs(out, exist_ok=True)
    objs = git("rev-list", "--objects", "--all").splitlines()
    keep, dropped = [], 0
    for line in objs:
        sha, _, path = line.partition(" ")
        if sha in OMIT_COMMITS:
            continue
        if path and any(path.startswith(x) for x in EXCLUDE) and git("cat-file", "-t", sha) == "blob":
            dropped += 1; continue
        keep.append(sha)
    base = os.path.join(out, "original-history")
    p = subprocess.run(["git", "pack-objects", base], cwd=ROOT, input="\n".join(keep) + "\n", capture_output=True,
                       text=True, check=True)
    name = p.stdout.strip()
    refs = {r.split()[1]: r.split()[0] for r in git("show-ref").splitlines()}
    children = sorted(l.split()[0] for l in git("rev-list", "--all", "--parents").splitlines()
                      if set(l.split()[1:]) & set(OMIT_COMMITS))
    json.dump(dict(pack=f"original-history-{name}.pack", objects=len(keep), blobs_omitted=dropped,
                   omitted_paths=list(EXCLUDE), commits_omitted=OMIT_COMMITS, shallow=children, refs=refs), open(os.path.join(out, "pack_manifest.json"), "w"), indent=1)
    print(f"pack {name}: {len(keep)} objects, {dropped} blobs omitted ({', '.join(EXCLUDE)})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "table"
    {"logs": export_logs, "table": table, "pack": lambda: pack(sys.argv[2])}[cmd]()
