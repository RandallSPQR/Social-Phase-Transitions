"""Known-answer test for llm/compare_local.py (no real data used; writes only to a temp dir).

  python llm/test_compare_local.py [--B 200]

Two synthetic endpoints in the exact local CSV format (rows copied from gemma-31b-local.csv, P(A) replaced):
  S1  provider truth == local truth (independent prompt noise) -> slope CI contains 1; local estimates near truth
  S2  provider truth == 0.8 × local truth                       -> slope CI contains 0.8; verdict 'jitter biases estimates'
The S1 agreement fraction is printed, not asserted: the pre-registered verdict rule (≥ 90% of ~31 correlated
coefficients agreeing at 95% each) can fail under identical truth; its null operating characteristic is
estimated separately (llm/compare_local_null.py). Calibration of the per-coefficient CIs (bootstrap SE vs the SD
of the difference over 40 independent replications) was checked on 2026-10-10: ratios 0.75–1.53 (EXECUTION_LOG 33).
"""
import argparse, os, sys, tempfile
import numpy as np, pandas as pd
from scipy.special import expit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_local as C

ROOT = C.ROOT
TRUE = dict(b_a=1.0, b_r=0.6, gamma=0.8, h_C=0.4, h=0.3, h_L=0.3, h_O=-0.2)


def synth(src, scale, noise, seed):
    d = pd.read_csv(src, dtype={"counts": str, "seq": str})
    rng = np.random.default_rng(seed)
    c = np.array([[int(x) for x in s.zfill(4)] for s in d["counts"]])
    Ma, Mr = c[:, 0] - c[:, 1], c[:, 2] - c[:, 3]
    L = np.where(d["mapping"] == 0, 1, -1); O = np.where(d["mapping"] == d["order"], 1, -1)
    neutral = d["framing"] == "neutral"
    t = {k: v * scale for k, v in TRUE.items()}
    eta = t["b_a"] * Ma - t["b_r"] * Mr + t["gamma"] * d["s0"] + np.where(neutral, t["h"], t["h_C"] + t["h_L"] * L) + t["h_O"] * O
    # per-prompt noise shared by both replicates (cells differ), plus a tiny per-replicate jitter
    key = d[["arm", "counts", "s0", "mapping", "order", "perm"]].astype(str).agg("|".join, axis=1)
    pn = {k: rng.normal(0, noise) for k in key.unique()}
    eta = eta + key.map(pn).values + rng.normal(0, 0.02, len(d))
    y = expit(eta)
    d["pA"] = np.where(d["mapping"] == 0, y, 1 - y); d["leak"] = 0.0; d["censored"] = 0
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=200); a = ap.parse_args()
    src = os.path.join(ROOT, "results/llm/stage1/main/gemma-31b-local.csv")
    tmp = tempfile.mkdtemp()
    synth(src, 1.0, 0.3, 1).to_csv(os.path.join(tmp, "loc.csv"), index=False)
    synth(src, 1.0, 0.3, 2).to_csv(os.path.join(tmp, "prov_same.csv"), index=False)
    synth(src, 0.8, 0.3, 3).to_csv(os.path.join(tmp, "prov_att.csv"), index=False)
    ok = True
    for prov, want in (("prov_same", 1.0), ("prov_att", 0.8)):
        T, s = C.compare_pair("loc", prov, "synthetic", a.B, tmp, tmp, 4)
        print(f"{prov}: slope {s['slope']:.3f} [{s['slope_lo']:.3f}, {s['slope_hi']:.3f}] agree {s['frac_agree']:.0%} -> {s['verdict']}")
        ok &= s["slope_lo"] <= want <= s["slope_hi"]
        if want != 1.0:
            ok &= s["verdict"] == "jitter biases estimates"
        # every coefficient's local estimate should be near its truth in S1
        if prov == "prov_same":
            for r in T[T["arm"] == "political"].itertuples():
                if r.coef in TRUE:
                    print(f"   political {r.coef:6s} local {r.local:+.3f} (truth {TRUE[r.coef]:+.2f})")
                    ok &= abs(r.local - TRUE[r.coef]) < 0.05
    print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
