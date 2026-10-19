"""Progress and ETA of the amplification stage B run (cost-weighted by unit size). python llm/amp_eta.py"""
import glob, json, os, re, time
OUT = "results/llm/amplification"
g = json.load(open(f"{OUT}/grid.json")); nT = len(g["grid"])
W = {("eq", 1600): 1.0, ("eq", 400): 0.42, ("eq", 100): 0.12, ("s3", 1600): 0.03, ("s3", 400): 0.012, ("s3", 100): 0.004}
total = sum(w * nT * 3 for w in W.values())
done = sum(W[(p.split("/")[-2], int(re.search(r"_N(\d+)_", p).group(1)))] for p in glob.glob(f"{OUT}/perturb/*/*.csv.gz"))
n = len(glob.glob(f"{OUT}/perturb/*/*.csv.gz"))
log = open("results/llm/logs/bg/amp_run.log").read()
t0 = time.mktime(time.strptime(re.findall(r"(\S+Z) start pid", log)[-1], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
el = time.time() - t0; frac = done / total
eta = el / frac * (1 - frac) if frac > 0 else float("nan")
print(f"{n}/360 units, {100 * frac:.0f}% of compute, elapsed {el / 3600:.1f} h, remaining ~{eta / 3600:.1f} h "
      f"(finish ~{time.strftime('%H:%M UTC', time.gmtime(time.time() + eta))})")
