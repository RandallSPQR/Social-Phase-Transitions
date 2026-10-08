"""Temperature handling (PREREG Amendment 3b).

  python ../llm/temp_check.py semantics      # classify every logprob endpoint: pre- vs post-temperature logprobs
  python ../llm/temp_check.py verify         # GPT-4o-mini: sampled P at T=0.5 and T=1.5 vs analytic σ(logit₁/T)

semantics: 2 prompts at T = 0.5 and T = 1; 'pre' if the A–B logit is unchanged (|Δ| ≤ 0.2 nats), 'post' if it
scales by 1/T (within 20%), otherwise 'unclear'.
verify: 20 prompts from the GPT-4o-mini main battery (neutral_own, perm 0, rep 0) with P₁(A) in [0.1, 0.9],
taken in SHA-256 order of their identity (if fewer qualify, the closest to 0.5 fill up); 50 samples each at
T = 0.5 and T = 1.5. Pass if, at each T, the logistic-regression slope of samples on logit₁ has a 95% CI
containing 1/T, and the Pearson χ² over the 20 prompts (df = 20) has p > 0.01.
Outputs: results/llm/temperature_semantics.csv, results/llm/temperature_verify.csv/.json
"""
import csv, hashlib, json, math, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from scipy.stats import chi2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client, battery
from endpoints import E
from run_stage1 import parse_lp, parse_sample
from analysis import fit_ce, logit

OUT = os.path.join(client.ROOT, "results", "llm")


def semantics():
    rows = []
    texts = [battery.render("neutral", (1, 0, 1, 0), 1, 0, 0, 0)[0], battery.render("neutral", (0, 2, 1, 0), -1, 0, 1, 0)[0]]
    def one(args):
        key, i = args
        model, tag, mode, roff, _ = E[key]
        L = {}
        for T in (1.0, 0.5):
            r = client.chat(battery.body(model, tag, texts[i], "lp", roff, T), rep=800)["resp"]
            p = parse_lp(r["first"])["pA"]
            L[T] = None if p is None else float(np.clip(logit(np.array(p)), -15, 15))
        return key, i, L
    jobs = [(k, i) for k, v in E.items() if v[2] == "lp" for i in range(2)]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, jobs))
    by = {}
    for key, i, L in res:
        by.setdefault(key, []).append(L)
    for key, Ls in by.items():
        verdicts = []
        for L in Ls:
            a, b = L[1.0], L[0.5]
            if a is None or b is None: verdicts.append("unclear")
            elif abs(b - a) <= 0.2: verdicts.append("pre")
            elif abs(a) > 0.3 and abs(b / a - 2) <= 0.4: verdicts.append("post")
            else: verdicts.append("unclear")
        v = verdicts[0] if len(set(verdicts)) == 1 else "unclear"
        rows.append({"key": key, "model": E[key][0], "tag": E[key][1], "logit_T1": [L[1.0] for L in Ls],
                     "logit_T05": [L[0.5] for L in Ls], "logprobs": v})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "temperature_semantics.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def verify(key="gpt4o-mini", n_samples=50, temps=(0.5, 1.5)):
    d = pd.read_csv(os.path.join(OUT, "stage1", "main", f"{key}.csv"), dtype={"counts": str})
    d = d[(d.arm == "neutral_own") & (d.perm == 0) & (d.rep == 0)].copy()
    d["ident"] = [hashlib.sha256(f"{c}|{s}|{o}".encode()).hexdigest() for c, s, o in zip(d.counts, d.s0, d.order)]
    d = d.sort_values("ident")
    q = d[(d.pA >= 0.1) & (d.pA <= 0.9)].head(20)
    if len(q) < 20:
        rest = d.drop(q.index).assign(dist=lambda x: (x.pA - 0.5).abs()).sort_values("dist")
        q = pd.concat([q, rest.head(20 - len(q))])
    model, tag, _, roff, _ = E[key]
    jobs = []
    for _, r in q.iterrows():
        counts = tuple(int(c) for c in r["counts"].zfill(4))
        text, _ = battery.render("neutral", counts, int(r.s0), 0, int(r.order), 0)
        for T in temps:
            b = battery.body(model, tag, text, "sample", roff, T)
            jobs += [(r["ident"], float(r.pA), T, b, i) for i in range(n_samples)]
    def one(j):
        ident, p1, T, b, i = j
        ans = parse_sample(client.chat(b, rep=3000 + i)["resp"]["content"])
        return {"ident": ident, "p1": p1, "T": T, "answer": ans}
    with ThreadPoolExecutor(12) as ex:
        S = pd.DataFrame(list(ex.map(one, jobs)))
    sfx = "" if key == "gpt4o-mini" else f"_{key}"
    S.to_csv(os.path.join(OUT, f"temperature_verify{sfx}.csv"), index=False)
    res = {}
    for T, g in S.groupby("T"):
        g = g[g.answer.notna()]
        agg = g.groupby("ident").agg(p1=("p1", "first"), n=("answer", "size"), nA=("answer", lambda x: (x == "A").sum())).reset_index()
        l1 = logit(agg.p1.values); pred = 1 / (1 + np.exp(-l1 / T))
        x2 = float((((agg.nA - agg.n * pred) ** 2) / (agg.n * pred * (1 - pred))).sum())
        X = np.column_stack([l1]); y = agg.nA.values / agg.n.values; w = agg.n.values.astype(float)
        slope = float(fit_ce(X, y, w)[0])
        rng = np.random.default_rng(0); bs = []
        for _ in range(1000):   # bootstrap over prompts and samples
            idx = rng.integers(0, len(agg), len(agg)); nA = rng.binomial(agg.n.values[idx], y[idx])
            bs.append(fit_ce(X[idx], nA / agg.n.values[idx], w[idx])[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res[str(T)] = {"slope": slope, "slope_lo": float(lo), "slope_hi": float(hi), "expected": 1 / T,
                       "chi2": x2, "df": len(agg), "p_chi2": float(chi2.sf(x2, len(agg))), "n_invalid": int(S[(S["T"] == T) & S.answer.isna()].shape[0]),
                       "passes": bool(lo <= 1 / T <= hi and chi2.sf(x2, len(agg)) > 0.01)}
    json.dump(res, open(os.path.join(OUT, f"temperature_verify{sfx}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)); print("spend", client.spend())


if __name__ == "__main__":
    {"semantics": semantics, "verify": verify}[sys.argv[1]](*sys.argv[2:])
