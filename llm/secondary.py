"""Secondary/exploratory contrasts (PREREG §7, Amendment 3): wording arm vs neutral_own, own-position-hidden
vs shown, and framing sensitivity. Differences of independent bootstrap draws (arms are fitted separately),
95% percentile CIs; flagged if the CI excludes 0. Writes results/llm/stage1_secondary.csv."""
import glob, json, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []; rng = np.random.default_rng(11)
CONTRASTS = [("wording: w2 − v2", "neutral_own_w2", "neutral_own"), ("own hidden − shown", "neutral_noown", "neutral_own"),
             ("political − neutral", "political", "neutral_own"), ("workplace − neutral", "workplace", "neutral_own"),
             ("workplace − political", "workplace", "political")]
for p in sorted(glob.glob(os.path.join(ROOT, "results/llm/fits/main/*.json"))):
    key = os.path.basename(p)[:-5]; A = {a["arm"]: a for a in json.load(open(p))["arms"]}
    for name, a1, a0 in CONTRASTS:
        if a1 not in A or a0 not in A: continue
        for q in ("beta", "b_a", "b_r", "gamma", "alpha"):
            if q in A[a1]["draws"] and q in A[a0]["draws"]:
                d = rng.permutation(np.array(A[a1]["draws"][q])) - rng.permutation(np.array(A[a0]["draws"][q]))
                lo, hi = np.percentile(d, [2.5, 97.5])
                rows.append({"key": key, "contrast": name, "param": q, "diff": A[a1]["coef"][q]["est"] - A[a0]["coef"][q]["est"],
                             "lo": lo, "hi": hi, "flag": bool(lo > 0 or hi < 0)})
R = pd.DataFrame(rows); R.round(3).to_csv(os.path.join(ROOT, "results/llm/stage1_secondary.csv"), index=False)
print(R.groupby(["contrast", "param"]).flag.agg(["sum", "count"]).to_string())
w = R[R.contrast.str.startswith("wording")]
print(w[w.flag].round(2).to_string(index=False))
