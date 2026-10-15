"""Re-materialise every API-served Stage 1 CSV from the response cache (no API calls; SPT_OFFLINE=1).

  SPT_OFFLINE=1 python replication/stage1_rematerialise.py [--workers 8]

For each committed CSV in results/llm/stage1/{main,comprehension,pilot}/ that came from an API endpoint, the
run_stage1.run() arguments (key, provider tag, arms, kmax, replicates or samples, temperature, output name) are read
off the committed file itself. run_stage1.run() then rebuilds every request body, looks it up in the cache and writes
the CSV to replication/out/stage1/<phase>/. The script then compares each row with the committed CSV.

Duplicate requests. Some batteries send the same request twice: two contact permutations can render the same
prompt, and the request then has the same body and replicate index. The original run made both live calls and
recorded both answers in the CSV. The cache is keyed by (body, replicate), so it keeps one of them. Rows of such
duplicate groups can therefore differ from the committed CSV. The check requires:
  (1) every row outside a duplicate group to be identical (pA, leak, censored, answer, content, served provider);
  (2) every row inside a duplicate group to equal one of the committed answers of its own group.
Downstream fits run on the committed CSVs (the record of the original answers), which this check ties to the cache.

Not re-materialised:
- *-local*.csv: self-hosted Gemma (GPU) outputs, which are inputs to the replication, not outputs of it.
- pilot/llama-8b_instruction-v1.csv: the pilot under the answer instruction that Amendment 2 replaced. The current
  battery code builds only the v2 instruction.
"""
import argparse, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "llm"))
import client, run_stage1  # noqa: E402
import json  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.join(ROOT, "replication", "out", "stage1")
run_stage1.OUT = OUT
GROUP = ["arm", "framing", "counts", "s0", "mapping", "order", "seq", "rep"]
FIELDS = ["pA", "leak", "censored", "answer", "content", "served"]

SKIP = {"pilot/llama-8b_instruction-v1.csv"}


def jobs():
    for phase in ("main", "comprehension", "pilot"):
        d = os.path.join(ROOT, "results", "llm", "stage1", phase)
        for f in sorted(os.listdir(d)):
            if not f.endswith(".csv") or "local" in f or f"{phase}/{f}" in SKIP:
                continue
            x = pd.read_csv(os.path.join(d, f), dtype={"counts": str})
            reps = sorted(int(r) for r in x.rep.unique())
            samples = len([r for r in reps if r >= 1000])
            kmax = int(x.counts.str.zfill(4).map(lambda s: sum(map(int, s))).max())
            yield dict(key=x.key.iloc[0], phase=phase, arms=sorted(x.arm.unique()), samples=samples,
                       reps=range(0 if samples else len([r for r in reps if r < 1000])),
                       temperature=float(x.temperature.iloc[0]), kmax=kmax if phase == "main" else 6,
                       tag_override=x.tag.iloc[0], out_name=f[:-4], n_committed=len(x))


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (np.isnan(a) and np.isnan(b)) or a == b
    return str(a) == str(b)


def check(ref_path, new_path):
    a = pd.read_csv(ref_path, dtype={"counts": str}, keep_default_na=False, na_values=[""])
    b = pd.read_csv(new_path, dtype={"counts": str}, keep_default_na=False, na_values=[""])
    f = [c for c in FIELDS if c in a.columns]
    k = GROUP + ["perm"] + (["question"] if "question" in a.columns else [])
    g = [c for c in GROUP + (["question"] if "question" in a.columns else []) if c in a.columns]
    a["_dup"] = a.groupby(g, dropna=False)["perm"].transform("size") > 1
    m = a.merge(b, on=k, how="outer", suffixes=("", "_new"), indicator=True)
    missing = int((m._merge != "both").sum())
    m = m[m._merge == "both"]
    diff = np.array([not all(same(r[c], r[c + "_new"]) for c in f) for _, r in m[f + [c + "_new" for c in f]].iterrows()])
    groups = {key: [tuple(x) for x in grp[f].values] for key, grp in a.groupby(g, dropna=False)}
    bad = 0
    for (_, r), d in zip(m.iterrows(), diff):
        if not d:
            continue
        if not r["_dup"]:
            bad += 1; continue
        new = tuple(r[c + "_new"] for c in f)
        if not any(all(same(x, y) for x, y in zip(o, new)) for o in groups[tuple(r[c] for c in g)]):
            bad += 1
    return dict(rows=len(a), rows_new=len(b), unmatched_rows=missing, duplicate_request_rows=int(a._dup.sum()),
                differing_rows=int(diff.sum()), unexplained=bad, ok=bool(missing == 0 and bad == 0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if not client.OFFLINE:
        sys.exit("refusing to run without SPT_OFFLINE=1 (a cache miss would become a paid API call)")
    res = {}
    for j in jobs():
        j.pop("n_committed")
        reps = j.pop("reps")
        p = run_stage1.run(j["key"], j["phase"], j["arms"], reps, j["samples"], j["temperature"], j["kmax"],
                           a.workers, None, tag_override=j["tag_override"], out_name=j["out_name"])
        ref = os.path.join(ROOT, "results", "llm", "stage1", j["phase"], j["out_name"] + ".csv")
        res[f"{j['phase']}/{j['out_name']}"] = r = check(ref, p)
        print(f"  {j['phase']}/{j['out_name']}: {r}", flush=True)
    json.dump(res, open(os.path.join(OUT, "cache_check.json"), "w"), indent=1)
    print("all ok" if all(r["ok"] for r in res.values()) else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
