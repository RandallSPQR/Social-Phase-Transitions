"""Three-kernel test, outcome-map branch "K_live fails -> investigate hidden state" (post-hoc, read-only, no API calls).

  python llm/kernel_hidden_state.py

By construction every live prompt is built from (c, s0) plus fresh per-call randomisation, so the live process can
leave "Markov in (c, s0)" only through the endpoint's reply. Checks:
 D1 identical-prompt repeats: logit-scale SD of the reply to the same full prompt text (logprob endpoints), vs the
    endpoint's Stage 1 repeat SD; share of bit-identical replies (provider-side caching).
 D2 time inhomogeneity: per (c, s0), P(+1) in warm-up sweeps 0-9 vs measured sweeps 10-19, and first vs second half
    of wall-clock time; weighted mean difference in P(stay) with a chain-cluster bootstrap CI.
 D3 history dependence: residual P(stay) - K_live(c, s0) split by whether the agent changed position at its previous
    update (Markov in (c, s0) implies no difference); chain-cluster bootstrap CI.
 D4 within-chain correlation: lag-1 correlation of consecutive residuals (any agent) within a chain.
Output: results/llm/kernel_test/hidden_state.json
"""
import json, math, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage3, stage3_audit as SA, kernel_test as K
from endpoints import E

lg = lambda p: np.clip(np.log(np.clip(p, 1e-12, 1) / np.clip(1 - p, 1e-12, 1)), -6, 6)


def boot_ci(d, stat, B=1000, seed=0):
    rng = np.random.default_rng(seed); ch = d.chain.unique(); g = {c: d[d.chain == c] for c in ch}
    v = [stat(pd.concat([g[c] for c in rng.choice(ch, len(ch))])) for _ in range(B)]
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def main():
    out = {}
    fits = {}
    for pid in K.POINTS:
        key, framing, rho, contacts = stage3.POINTS[pid]
        model, tag, mode, roff, _ = E[key]
        d = K.live_rows(pid)
        T, _ = K.live_kernel(d.groupby("idx").agg(S1=("p", "sum"), n=("p", "size")), K.N0_MAIN)
        d["k"] = T[d.idx.values]
        # P(stay) and its kernel value, in the own-position direction
        d["stay"] = np.where(d.s0 > 0, d.p, 1 - d.p); d["kstay"] = np.where(d.s0 > 0, d.k, 1 - d.k)
        d["res"] = d.stay - d.kstay
        d = d.sort_values(["chain", "sweep", "step"]).reset_index(drop=True)
        # previous update of the same agent in the same chain: did it change position?
        d["prev_changed"] = d.groupby(["chain", "agent"]).apply(lambda g: (g.new != g.s0).shift(1)).reset_index(level=[0, 1], drop=True)
        r = {}
        # D2 sweeps
        def period_diff(dd, col):
            a = dd.groupby(["idx", col]).stay.mean().unstack()
            n = dd.groupby("idx").size()
            a = a.dropna(); w = n.loc[a.index] / n.loc[a.index].sum()
            return float((w * (a[True] - a[False])).sum())
        d["late"] = d.sweep >= stage3.N_WARM
        r["D2_stay_late_minus_early_sweeps"] = period_diff(d, "late")
        r["D2_ci"] = boot_ci(d, lambda x: period_diff(x, "late"))
        # D3 history
        h = d.dropna(subset=["prev_changed"])
        f3 = lambda x: float(x[x.prev_changed == True].res.mean() - x[x.prev_changed == False].res.mean())
        r["D3_res_stay_after_change_minus_after_no_change"] = f3(h)
        r["D3_ci"] = boot_ci(h, f3)
        r["D3_n_after_change"] = int((h.prev_changed == True).sum())
        # D4 lag-1 residual correlation within chain
        f4 = lambda x: float(np.mean([np.corrcoef(g.res.values[:-1], g.res.values[1:])[0, 1] for _, g in x.groupby("chain") if g.res.std() > 0]))
        r["D4_lag1_residual_corr"] = f4(d)
        r["D4_ci"] = boot_ci(d, f4, B=300)
        # mean residual P(stay) in measured sweeps (what persistence sees)
        r["mean_res_stay_measured"] = float(d[d.late].res.mean())
        # D1 identical-prompt repeats + wall-clock drift (logprob endpoints)
        if mode == "lp":
            recs, _ = SA.load_cache(model)
            rows = []
            for x in d.itertuples(index=False):
                rec = recs[f"stage3:{pid}:{x.chain // 2}:{x.chain % 2}:{x.sweep}:{x.step}:{x.attempts}"]
                rows.append((rec["body"]["messages"][0]["content"], rec["t"], x.pA))
            P = pd.DataFrame(rows, columns=["prompt", "t", "pA"]); P["l"] = lg(P.pA.values)
            g = P.groupby("prompt").l
            rep = P[g.transform("size") >= 2]
            sd = np.sqrt(((rep.l - rep.groupby("prompt").l.transform("mean")) ** 2).sum() / (len(rep) - rep.prompt.nunique()))
            r["D1_prompts_repeated"] = int(rep.prompt.nunique()); r["D1_calls_in_repeats"] = int(len(rep))
            r["D1_within_prompt_sd_logit"] = float(sd)
            r["D1_share_bit_identical_to_first"] = float((rep.pA == rep.groupby("prompt").pA.transform("first")).mean())
            fit = json.load(open(os.path.join(stage3.ROOT, "results/llm/fits/surrogate", key + ".json")))
            r["stage1_repeat_sd"] = fit["arms"][{"neutral": "neutral_own", "political": "political", "workplace": "workplace"}[framing]].get("repeat_sd")
            # wall-clock drift on repeated prompts: late-half minus early-half mean logit within prompt
            tm = P.t.median(); rep = rep.assign(late=rep.t > tm)
            dd = rep.groupby(["prompt", "late"]).l.mean().unstack().dropna()
            r["D1_drift_logit_late_minus_early_walltime"] = float((dd[True] - dd[False]).mean())
            r["D1_drift_se"] = float((dd[True] - dd[False]).std(ddof=1) / math.sqrt(len(dd)))
            r["D1_drift_prompts"] = int(len(dd))
            r["run_wallclock_hours"] = float((P.t.max() - P.t.min()) / 3600)
        out[pid] = r
        print(pid, json.dumps(r), flush=True)
    json.dump(out, open(os.path.join(K.OUT, "hidden_state.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
