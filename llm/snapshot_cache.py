"""Write gzip snapshots of the raw response cache (results/llm/cache/*.jsonl -> *.jsonl.gz) for committing.
Raw .jsonl files are git-ignored (they reach 60+ MB; gzip is ~15x smaller)."""
import glob, gzip, os, shutil
d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "llm", "cache")
for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    with open(p, "rb") as fi, gzip.open(p + ".gz", "wb", compresslevel=9) as fo:
        shutil.copyfileobj(fi, fo)
    print(f"{os.path.basename(p)}: {os.path.getsize(p)/1e6:.1f} MB -> {os.path.getsize(p + '.gz')/1e6:.1f} MB")
