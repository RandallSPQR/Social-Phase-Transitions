"""Residual test 1: heterogeneity / mover-stayer (llm/RESIDUAL_TEST_PLAN.md; free, read-only, no API calls).

  python llm/residual_heterogeneity.py

a) What varies between agents in a Stage 3 request: every sent prompt is compared with the deterministic render of
   (framing, c, s0, letter mapping, option order, listed contact order); request bodies are compared field by field;
   the distinct prompt lines are listed.
b) Between-agent and between-graph-position variance of residual P(stay) vs K_live(c, s0), permutation null
   (labels shuffled within point, 1,000 permutations), Fisher combination over the 7 points.
c) Only if b) is significant (Fisher p < 0.05): heterogeneous-kernel simulation (not run otherwise).
Output: results/llm/residual/heterogeneity.json
"""
import collections, json, math, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery, stage3, stage3_audit as SA, kernel_test as K
from endpoints import E

OUT = os.path.join(K.ROOT, "results", "llm", "residual")


def tokens():
    res, lines = {}, collections.Counter()
    for pid in K.POINTS:
        key, framing, rho, contacts = stage3.POINTS[pid]
        model = E[key][0]; recs, _ = SA.load_cache(model)
        bodies = set(); n = mism = 0
        for g in range(stage3.N_GRAPHS):
            for r in (0, 1):
                d = pd.read_csv(os.path.join(stage3.OUT, "chains", pid, f"g{g}_r{r}.csv.gz"), dtype={"counts": str})
                for x in d.itertuples(index=False):
                    for att in range(1, x.attempts + 1):
                        rec = recs[f"stage3:{pid}:{g}:{r}:{x.sweep}:{x.step}:{att}"]
                        pr = rec["body"]["messages"][0]["content"]
                        P_ = SA.parse_prompt(pr, framing)
                        lP = P_["letter_of_P"] or (P_["own_letter"] if x.s0 > 0 else ("B" if P_["own_letter"] == "A" else "A"))
                        m = 0 if lP == "A" else 1
                        seq = [("a" if al else "r") + ("P" if le == lP else "Q") for al, le in P_["contacts"]]
                        counts = tuple(seq.count(t) for t in battery.TYPES) if contacts else tuple(int(c) for c in x.counts.zfill(4))
                        ref = stage3.render_live(framing, counts, int(x.s0), m, P_["order"], seq if contacts else
                                                 [t for t, k_ in zip(battery.TYPES, counts) for _ in range(k_)], contacts)
                        n += 1; mism += pr != ref
                        bodies.add(json.dumps({k: v for k, v in rec["body"].items() if k != "messages"}, sort_keys=True))
                        for l in pr.split("\n"):
                            lines[l] += 1
        res[pid] = dict(prompts=n, differ_from_deterministic_render=mism, distinct_request_bodies_excluding_message=len(bodies))
        print(pid, res[pid], flush=True)
    res["distinct_prompt_lines_all_points"] = len(lines)
    res["prompt_lines"] = sorted(lines)
    return res


def ss_between(lab, r):
    n = np.bincount(lab); s = np.bincount(lab, weights=r); m = r.mean()
    ok = n > 0
    return float((n[ok] * (s[ok] / n[ok] - m) ** 2).sum())


def variance_test(B=1000):
    res = {}
    for pid in K.POINTS:
        key, framing, rho, contacts = stage3.POINTS[pid]
        d = K.live_rows(pid)
        T, _ = K.live_kernel(d.groupby("idx").agg(S1=("p", "sum"), n=("p", "size")), K.N0_MAIN)
        k = T[d.idx.values]
        r = np.where(d.s0 > 0, d.p - k, k - d.p)                       # residual P(stay)
        units = {"agent": d.chain * 1000 + d.agent, "graph_position": (d.chain // 2) * 1000 + d.agent}
        rng = np.random.default_rng(stage3._seed("het-perm", pid)); out = {}
        for name, u in units.items():
            lab = pd.factorize(u)[0]
            obs = ss_between(lab, r)
            null = np.array([ss_between(rng.permutation(lab), r) for _ in range(B)])
            p = float(((null >= obs).sum() + 1) / (B + 1))
            tau2 = max(0.0, (obs - null.mean()) / len(r))      # method of moments: excess between-unit variance (prob. scale)
            out[name] = dict(units=int(lab.max() + 1), obs_ss=obs, null_mean=float(null.mean()), null_sd=float(null.std()),
                             z=float((obs - null.mean()) / null.std()), p=p, excess_ss_per_update=tau2)
        res[pid] = out
        print(pid, {k_: (round(v["z"], 2), round(v["p"], 3)) for k_, v in out.items()}, flush=True)
    fisher = {u: float(stats.combine_pvalues([res[p][u]["p"] for p in K.POINTS], method="fisher")[1]) for u in ("agent", "graph_position")}
    return res, fisher


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = tokens()
    var, fisher = variance_test()
    out = dict(a_tokens=tok, b_variance=var, b_fisher_p=fisher,
               c_triggered=bool(min(fisher.values()) < 0.05))
    json.dump(out, open(os.path.join(OUT, "heterogeneity.json"), "w"), indent=1)
    print("Fisher p:", fisher, "-> c) triggered:", out["c_triggered"])


if __name__ == "__main__":
    main()
