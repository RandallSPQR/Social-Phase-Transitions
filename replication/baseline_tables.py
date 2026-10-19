"""Regenerate the baseline-physics summary tables from the committed raw simulation output (data/).

The original CSVs were transcribed from the printed output of sim/analyze.py and from data/asym/asym.npy; this
script makes that transcription a program.

  python replication/baseline_tables.py

- results/validation_crossings.csv: sim/analyze.summarize(tag) per tag, each tag in a FRESH process, so the
  module-level bootstrap generator (default_rng(0)) starts in the same state for every tag. Crossing point estimates
  do not depend on the generator; bootstrap CI bounds do, and the original call order is not recorded, so CI bounds
  are compared with a Monte-Carlo tolerance (replication/checks.py).
- results/asymmetry_sweep.csv: per (N, c, rho, eps, T) the mean over graphs of |m| (column 0), q2 (column 2) and
  persistence (column 4) of sim/run_asym.py's per-graph rows, with SE = SD(ddof=1)/sqrt(n_graphs).
"""
import csv, json, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = ["f_r00", "f_r10", "f_r20", "sg_c3", "sg_c4", "sg_c6"]

_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import io, contextlib, analyze
tag = sys.argv[2]
with contextlib.redirect_stdout(io.StringIO()):
    s = analyze.summarize(tag, "sg" if tag.startswith("sg") else "f")
print(json.dumps(dict(c=float(s["c"]), rho=float(s["rho"]), pred=float(s["pred"]),
                      res=[[int(a), int(b), [None if v is None else float(v) for v in bb[:3]],
                            [None if v is None else float(v) for v in ss[:3]]] for a, b, bb, ss in s["res"]])))
"""


def validation():
    out = os.path.join(ROOT, "results", "validation_crossings.csv")
    rows = []
    for tag in TAGS:
        r = json.loads(subprocess.run([sys.executable, "-c", _CHILD, os.path.join(ROOT, "sim"), tag],
                                      check=True, capture_output=True, text=True, cwd=os.path.join(ROOT, "sim")).stdout)
        for n1, n2, b, s in r["res"]:
            rows.append([tag, r["c"], r["rho"], "sg" if tag.startswith("sg") else "f", round(r["pred"], 4), n1, n2,
                         *[round(v, 4) for v in s], *[round(v, 4) for v in b]])
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "c", "rho", "transition", "theory_T", "N1", "N2", "scaledchi_T", "scaledchi_lo95",
                    "scaledchi_hi95", "binder_T", "binder_lo95", "binder_hi95"])
        w.writerows(rows)
    print("wrote", os.path.relpath(out, ROOT), len(rows), "rows")


def asymmetry():
    a = np.load(os.path.join(ROOT, "data", "asym", "asym.npy"), allow_pickle=True)
    out = os.path.join(ROOT, "results", "asymmetry_sweep.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "c", "rho", "eps", "T", "n_graphs", "abs_m", "abs_m_se", "q2", "persistence", "persistence_se"])
        for N, c, rho, eps, T, res in a:
            x = np.asarray(res, float); n = len(x); se = x.std(0, ddof=1) / np.sqrt(n)
            w.writerow([N, c, rho, eps, T, n, x[:, 0].mean(), se[0], x[:, 2].mean(), x[:, 4].mean(), se[4]])
    print("wrote", os.path.relpath(out, ROOT), len(a), "rows")


if __name__ == "__main__":
    validation()
    asymmetry()
