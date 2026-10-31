"""Diagnostic (plan §2.1, labelled): how much of the 1/T chi-square excess is explained by reference-logit repeat noise
(the temp_diag.py method, generalised). For each checked model, the reference logit per prompt is the mean of its two
repeats; chi2 is simulated under the 1/T rule plus that reference noise. Output: results/llm/confirm/temp_diag.json
"""
import hashlib, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import chi2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))


def refs_stage1(key):
    d = pd.read_csv(os.path.join(ROOT, f"results/llm/stage1/main/{key}.csv"), dtype={"counts": str})
    d = d[(d.arm == "neutral_own") & (d.perm == 0)]
    d["ident"] = [hashlib.sha256(f"{c}|{s}|{o}".encode()).hexdigest() for c, s, o in zip(d.counts, d.s0, d.order)]
    return d.groupby("ident").pA.agg(list).to_dict()


def refs_selected():
    d = pd.read_csv(os.path.join(ROOT, "results/llm/confirm/kavg_selected.csv"))
    P = pd.read_csv(os.path.join(ROOT, "results/llm/confirm/kavg_prompts_selected.csv"))
    d = d.merge(P[["prompt_id", "prompt"]], on="prompt_id"); d["ident"] = d.prompt.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    return d.groupby("ident").pA.agg(list).to_dict()


def diag(S, ref, ident_col="ident"):
    ids = S[ident_col].unique(); r0 = np.array([L(ref[i][0]) for i in ids]); r1 = np.array([L(ref[i][1]) for i in ids])
    s = float(np.std(r0 - r1) / np.sqrt(2)); lm = {i: (L(ref[i][0]) + L(ref[i][1])) / 2 for i in ids}; rng = np.random.default_rng(0); out = {"ref_repeat_sd": s}
    for T, g in S[S.answer.notna()].groupby("T"):
        a = g.groupby(ident_col).agg(n=("answer", "size"), nA=("answer", lambda x: (x == "A").sum())).reset_index()
        l = np.array([lm[i] for i in a[ident_col]]); p = 1 / (1 + np.exp(-l / T)); x2 = float((((a.nA - a.n * p) ** 2) / (a.n * p * (1 - p))).sum())
        sims = [float((((k - a.n * p) ** 2) / (a.n * p * (1 - p))).sum()) for k in
                (rng.binomial(a.n.values, 1 / (1 + np.exp(-(l + rng.normal(0, s / np.sqrt(2), len(a))) / T))) for _ in range(2000))]
        out[str(T)] = dict(chi2_vs_mean_ref=x2, p_nominal=float(chi2.sf(x2, len(a))), sim_median=float(np.median(sims)), sim_p95=float(np.percentile(sims, 95)),
                           within_noise=bool(x2 <= np.percentile(sims, 95)))
    return out


res = {}
for key in ("qwen-122b", "nemo-12b"):
    S = pd.read_csv(os.path.join(ROOT, f"results/llm/temperature_verify_{key}.csv")); res[key] = diag(S, refs_stage1(key))
S = pd.read_csv(os.path.join(ROOT, "results/llm/confirm/temp_selected_samples.csv")); res["gpt-4o-mini (selected)"] = diag(S, refs_selected())
json.dump(res, open(os.path.join(ROOT, "results/llm/confirm/temp_diag.json"), "w"), indent=1); print(json.dumps(res, indent=1))
