"""Write gzip snapshots of the raw response cache (results/llm/cache/*.jsonl -> *.jsonl.gz) for committing /
publishing. Raw .jsonl files are git-ignored (60+ MB; gzip is ~15x smaller). Account-linked fields (the
OpenRouter generation id, resp.id) are stripped from both the raw cache and the snapshot."""
import glob, gzip, json, os
STRIP_RESP = ("id",)
d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "llm", "cache")
for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    recs = [json.loads(l) for l in open(p) if l.strip()]
    n_strip = 0
    for r in recs:
        for k in STRIP_RESP:
            if k in r["resp"]:
                del r["resp"][k]; n_strip += 1
    body = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in recs)
    tmp = p + ".tmp"; open(tmp, "w").write(body); os.replace(tmp, p)          # raw cache rewritten without ids
    with gzip.open(p + ".gz", "wt", compresslevel=9) as fo:
        fo.write(body)
    print(f"{os.path.basename(p)}: {len(recs)} records, stripped {n_strip} ids, {os.path.getsize(p + '.gz')/1e6:.1f} MB gz")
