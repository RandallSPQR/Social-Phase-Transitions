"""Exploratory diagnostic (not pre-registered) of the failed temperature verification (Amendment 3b):
how much of the χ² excess is explained by the GPT-4o-mini reference-logit noise between identical calls?
Writes results/llm/temperature_verify_diagnostic.txt"""
import hashlib, os, sys
import numpy as np, pandas as pd
from scipy.stats import chi2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = pd.read_csv(os.path.join(ROOT, "results/llm/temperature_verify.csv"))
d = pd.read_csv(os.path.join(ROOT, "results/llm/stage1/main/gpt4o-mini.csv"), dtype={"counts": str})
d = d[(d.arm == "neutral_own") & (d.perm == 0)]
d["ident"] = [hashlib.sha256(f"{c}|{s}|{o}".encode()).hexdigest() for c, s, o in zip(d.counts, d.s0, d.order)]
L = lambda p: np.log(p / (1 - p))
ref = d.groupby("ident").pA.agg(list).to_dict(); ids = S.ident.unique()
r0 = np.array([L(ref[i][0]) for i in ids]); r1 = np.array([L(ref[i][1]) for i in ids]); s = np.std(r0 - r1) / np.sqrt(2)
out = [f"reference repeat SD on the 20 prompts: {s:.3f} nats (max |diff| {np.abs(r0 - r1).max():.2f})"]
lm = {i: (L(ref[i][0]) + L(ref[i][1])) / 2 for i in ids}; rng = np.random.default_rng(0)
for T, g in S.groupby("T"):
    a = g.groupby("ident").agg(n=("answer", "size"), nA=("answer", lambda x: (x == "A").sum())).reset_index()
    l = np.array([lm[i] for i in a.ident]); p = 1 / (1 + np.exp(-l / T)); x2 = (((a.nA - a.n * p) ** 2) / (a.n * p * (1 - p))).sum()
    sims = []
    for _ in range(2000):
        k = rng.binomial(a.n.values, 1 / (1 + np.exp(-(l + rng.normal(0, s / np.sqrt(2), len(a))) / T)))
        sims.append((((k - a.n * p) ** 2) / (a.n * p * (1 - p))).sum())
    out.append(f"T={T}: chi2 vs mean-of-2-reps reference = {x2:.1f} (p={chi2.sf(x2, 20):.2g}); simulated under 1/T rule + reference noise: median {np.median(sims):.1f}, 95th pct {np.percentile(sims, 95):.1f}")
open(os.path.join(ROOT, "results/llm/temperature_verify_diagnostic.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
