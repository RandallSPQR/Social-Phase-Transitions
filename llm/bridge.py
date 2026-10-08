"""Bridge check (PREREG §7): GPT-4o-mini response function from logprobs vs from 30-sample data, on the
identical reduced battery (neutral_own + political, k <= 4, perm 0). Writes results/llm/bridge_comparison.csv."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for src, f in [("logprobs", "gpt4o-mini.csv"), ("sampled", "gpt4o-mini_sampled.csv")]:
    d = analysis.load(os.path.join(ROOT, "results/llm/stage1/main", f))
    for arm in ("neutral_own", "political"):
        P, st = analysis.prompts(d[d.arm == arm]); P = P[(P.perm == 0) & (P.k <= 4)]
        X, names = analysis.design(P, "A0"); b = analysis.fit_ce(X, P.y.values, P.w.values.astype(float))
        est = analysis.derived(dict(zip(names, b))); est["alpha"] = analysis.fit_A3(P)[0]
        dr = analysis.bootstrap(P, 500, np.random.default_rng(1))
        for k in ("b_a", "b_r", "gamma", "beta", "b_r_minus_b_a", "alpha", "h", "h_C", "h_L", "h_O"):
            if k in dr:
                lo, hi = np.percentile(dr[k], [2.5, 97.5]); rows.append({"source": src, "arm": arm, "param": k, "est": est[k], "lo": lo, "hi": hi, "se": dr[k].std()})
R = pd.DataFrame(rows)
W = R.pivot_table(index=["arm", "param"], columns="source", values=["est", "se"])
W["diff"] = W[("est", "sampled")] - W[("est", "logprobs")]
W["z"] = W["diff"] / np.sqrt(W[("se", "sampled")] ** 2 + W[("se", "logprobs")] ** 2)
W.columns = ["_".join(c).strip("_") for c in W.columns]
W.round(3).to_csv(os.path.join(ROOT, "results/llm/bridge_comparison.csv"))
print(W.round(3).to_string())
