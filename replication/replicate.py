"""Replication driver: fetch -> verify -> rerun every analysis from the cache -> regenerate figures -> compare with
the committed results -> pass/fail table per claim.

  python replication/replicate.py --mode fast          # make replicate-fast
  python replication/replicate.py --mode full          # make replicate
  python replication/replicate.py --mode fast --only s1.fits k.avg   # selected steps (dependencies are not resolved)
  python replication/replicate.py --report-only        # rebuild the report from the last run's records

No API key and no GPU are used. llm/client.py runs with SPT_OFFLINE=1, so any request missing from the cache raises
instead of calling the API. Outputs go to replication/out/: one log per step (logs/), the reference copies of the
committed results (reference/), the comparison records (compare.json), and the report (REPORT.md, report.json).

Run it on a clean checkout. Steps overwrite result files in place, and the comparison reference is the snapshot taken
at the start of the first run (replication/out/reference/). Delete replication/out/ to start over.
"""
import argparse, glob, hashlib, json, os, platform, random, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "replication"))
import compare, steps as S  # noqa: E402

OUT = os.path.join(ROOT, "replication", "out")
REF = os.path.join(OUT, "reference")
LOGS = os.path.join(OUT, "logs")
RECORDS = os.path.join(OUT, "records.json")


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rel(p):
    return os.path.relpath(p, ROOT)


def expand(pattern):
    return sorted(p for p in glob.glob(os.path.join(ROOT, pattern), recursive=True) if os.path.isfile(p))


# ---------------------------------------------------------------- environment
def check_env():
    """The pinned versions are in replication/requirements.lock. networkx must match exactly (graph construction
    depends on it); the others are reported."""
    want = {}
    for line in open(os.path.join(ROOT, "replication", "requirements.in")):
        line = line.split("#")[0].strip()
        if "==" in line:
            n, v = line.split("=="); want[n.strip()] = v.strip()
    from importlib.metadata import version, PackageNotFoundError
    rows, hard = [], []
    for n, v in want.items():
        try:
            have = version(n)
        except PackageNotFoundError:
            have = None
        rows.append((n, v, have))
        if have != v and n in ("networkx", "numpy", "numba", "llvmlite", "scipy", "pandas"):
            hard.append(f"{n} {have} (pinned {v})")
    info = dict(python=platform.python_version(), platform=platform.platform(), cpus=os.cpu_count(),
                packages={n: h for n, _, h in rows})
    if not platform.python_version().startswith("3.13"):
        hard.append(f"python {platform.python_version()} (pinned 3.13)")
    return info, hard


# ---------------------------------------------------------------- data
def fetch(mode):
    """Cache snapshots (results/llm/cache_manifest.json) and, for fast mode, the simulation unit bundles
    (replication/sim_units_manifest.json), all SHA-256-verified."""
    sys.path.insert(0, os.path.join(ROOT, "llm"))
    import fetch_cache
    bad = fetch_cache.fetch(verbose=False)
    say(f"cache: {fetch_cache.manifest()['files']} snapshots, {'all verified' if not bad else f'{bad} FAILED'}")
    if bad:
        return False
    if mode == "fast":
        import fetch_units
        if fetch_units.fetch():
            return False
    return True


# ---------------------------------------------------------------- reference snapshot
def snapshot(selected):
    """Copy every committed file that a selected step will compare into replication/out/reference/ (once)."""
    n = 0
    for st in selected:
        for c in st["compare"]:
            for p in expand(c[0]):
                q = os.path.join(REF, rel(p))
                if not os.path.exists(q):
                    os.makedirs(os.path.dirname(q), exist_ok=True); shutil.copy2(p, q); n += 1
    say(f"reference snapshot: {n} new files in {rel(REF)}")


# ---------------------------------------------------------------- spot checks (fast mode)
SPOT = {"amp.stageB": ["results/llm/amplification/locate2/*/*_N100_*.csv.gz", "results/llm/amplification/perturb/*/*_N100_*.csv.gz"],
        "conf.locate": ["results/llm/confirm/sim/coarse/*_N100_*.csv.gz", "results/llm/confirm/sim/fine/*_N100_*.csv.gz"],
        "conf.step45": ["results/llm/confirm/step45/step4/*_N100_*.csv.gz", "results/llm/confirm/step45/step5/*_N100_*.csv.gz"],
        "conf.gemma": ["results/llm/confirm/gemma_mechanism/len/*_N100_*.csv.gz"]}
SPOT_PER_PATTERN = 3


def spot_prepare(step_id):
    """Move a fixed sample of fetched unit files aside so the step recomputes exactly those units."""
    held = []
    for pat in SPOT.get(step_id, []):
        files = expand(pat)
        pick = sorted(files, key=lambda p: hashlib.sha256(rel(p).encode()).hexdigest())[:SPOT_PER_PATTERN]
        for p in pick:
            for q in [p] + ([p.replace(".csv.gz", "_visits.npy")] if os.path.exists(p.replace(".csv.gz", "_visits.npy")) else []):
                dest = os.path.join(OUT, "spot", rel(q))
                os.makedirs(os.path.dirname(dest), exist_ok=True); shutil.move(q, dest); held.append(q)
    return held


def spot_check(held):
    res = []
    for q in held:
        r = compare.compare_file(os.path.join(OUT, "spot", rel(q)), q, "exact")
        res.append(dict(file=rel(q), **r))
    return res


# ---------------------------------------------------------------- running
def remove(patterns):
    for pat in patterns:
        for p in glob.glob(os.path.join(ROOT, pat)):
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)


def ffmpeg_on_path(env):
    """Put the pinned ffmpeg (imageio-ffmpeg, in the lock file) first on PATH for the illustration step."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return env
    d = os.path.join(OUT, "bin"); os.makedirs(d, exist_ok=True)
    link = os.path.join(d, "ffmpeg")
    if not os.path.exists(link):
        os.symlink(exe, link)
    env["PATH"] = d + os.pathsep + env.get("PATH", "")
    return env


def run_step(st, mode, records):
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, st["id"] + ".log")
    env = ffmpeg_on_path(dict(os.environ, SPT_OFFLINE="1", MPLBACKEND="Agg", PYTHONHASHSEED="0", PYTHONUNBUFFERED="1"))
    remove(st["clean"])
    if mode == "full":
        remove(st["clean_full"])
    held = spot_prepare(st["id"]) if mode == "fast" else []
    t0, rc = time.time(), 0
    with open(log, "w") as f:
        for cmd in st["cmds"]:
            f.write(f"$ {cmd}\n"); f.flush()
            p = subprocess.run(cmd, shell=True, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            if p.returncode:
                rc = p.returncode; f.write(f"!! exit {rc}\n"); break
    dt = time.time() - t0
    cmp = []
    for c in st["compare"]:
        pat, prof = c[0], c[1]
        tol = c[2] if len(c) > 2 else None
        for refp in sorted(glob.glob(os.path.join(REF, pat), recursive=True)):
            newp = os.path.join(ROOT, os.path.relpath(refp, REF))
            try:
                r = compare.compare_file(refp, newp, prof, tol)
            except Exception as e:   # a file the comparator cannot read is a failed comparison, not a crash
                r = dict(ok=False, n=0, bad=1, worst=None, probs=[f"comparison error: {type(e).__name__}: {e}"[:200]],
                         profile=prof)
            cmp.append(dict(file=rel(newp), **r))
    spots = spot_check(held)
    ok = rc == 0 and all(c["ok"] for c in cmp) and all(s["ok"] for s in spots)
    records[st["id"]] = dict(id=st["id"], title=st["title"], claims=st["claims"], mode=mode, exit=rc,
                             seconds=round(dt, 1), compared=cmp, spot=spots, ok=ok, log=rel(log))
    json.dump(records, open(RECORDS, "w"), indent=1)
    nbad = sum(not c["ok"] for c in cmp)
    say(f"{'PASS' if ok else 'FAIL'} {st['id']:<14} {dt / 60:6.1f} min  {len(cmp)} files compared, {nbad} differ"
        + (f", {len(spots)} units recomputed ({sum(not s['ok'] for s in spots)} differ)" if spots else "")
        + (f", exit {rc} (see {rel(log)})" if rc else ""))
    for c in cmp:
        if not c["ok"]:
            say(f"      {c['file']}: {'; '.join(c['probs'][:2])}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    import claims
    if a.report_only:
        return claims.report(json.load(open(RECORDS)), a.mode, json.load(open(os.path.join(OUT, "env.json"))))
    info, hard = check_env()
    json.dump(info, open(os.path.join(OUT, "env.json"), "w"), indent=1)
    say(f"python {info['python']} on {info['platform']}, {info['cpus']} CPUs; mode {a.mode}")
    if hard:
        say("ENVIRONMENT MISMATCH: " + "; ".join(hard)); say("install replication/requirements.lock (or use the Dockerfile)")
        sys.exit(2)
    selected = [s for s in S.STEPS if (s["mode"] == "both" or s["mode"] == a.mode) and (a.only is None or s["id"] in a.only)]
    if not a.skip_fetch and not fetch(a.mode):
        say("data fetch/verification FAILED"); sys.exit(3)
    snapshot(selected)
    records = json.load(open(RECORDS)) if os.path.exists(RECORDS) and a.only else {}
    t0 = time.time()
    for st in selected:
        say(f"---- {st['id']}: {st['title']}")
        run_step(st, a.mode, records)
    say(f"all steps done in {(time.time() - t0) / 60:.1f} min")
    sys.exit(claims.report(records, a.mode, info))


if __name__ == "__main__":
    main()
