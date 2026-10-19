"""Residual test plan, amendment 1 §4: identical-request dependence (free, read-only, no API calls).

  python llm/residual_identical_requests.py

Within each chain, calls whose full prompt text repeats an earlier call by the same agent are paired with that agent's
most recent identical call. Observed letter agreement vs the independence expectation:
 (i)  P^2 + (1-P)^2, P = K_live(c, s0) in the prompt's letter space (owner's definition);
 (ii) P1 P2 + (1-P1)(1-P2) with prompt-level P: logprob endpoints the two calls' own returned P(A); P5 the share of A
      over all OTHER live calls with the same prompt text (pair left out). Pre-stated primary.
 (iii) CORRECTION (post-hoc, 2026-10-13): pairs exist only when the agent's prompt repeats, including its own
      position, which selects on the EARLIER choice (usually "stay"). Under independence the later call agrees with
      probability P_later(earlier letter), conditioning on the observed earlier choice; (i) and (ii) ignore this
      selection and understate the expected agreement. (iii) uses the later call's prompt-level P (as in (ii)).
Chain-cluster bootstrap 95% CI (1,000). Output: results/llm/residual/identical_requests.json
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage3, stage3_audit as SA, kernel_test as K
from endpoints import E

OUT = os.path.join(K.ROOT, "results", "llm", "residual")


def boot(df, f, B=1000, seed=0):
    rng = np.random.default_rng(seed); ch = df.chain.unique(); g = {c: df[df.chain == c] for c in ch}
    v = [f(pd.concat([g[c] for c in rng.choice(ch, len(ch))])) for _ in range(B)]
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def main():
    out = {}
    for pid in K.POINTS:
        key, framing, rho, contacts = stage3.POINTS[pid]
        model, tag, mode, roff, _ = E[key]
        recs, _ = SA.load_cache(model)
        d = K.live_rows(pid)
        T, _ = K.live_kernel(d.groupby("idx").agg(S1=("p", "sum"), n=("p", "size")), K.N0_MAIN)
        d["K1"] = T[d.idx.values]
        d["prompt"] = [recs[f"stage3:{pid}:{x.chain // 2}:{x.chain % 2}:{x.sweep}:{x.step}:{x.attempts}"]["body"]["messages"][0]["content"]
                       for x in d.itertuples(index=False)]
        d["KA"] = np.where(d.mapping == 0, d.K1, 1 - d.K1)              # K_live in letter space: P(letter A)
        d["A"] = (d.letter == "A").astype(int)
        d = d.sort_values(["chain", "sweep", "step"]).reset_index(drop=True)
        # prompt-level share of A (P5): totals per prompt text
        tot = d.groupby("prompt").A.agg(["sum", "size"])
        pairs = []
        for (ch, ag), g in d.groupby(["chain", "agent"]):
            last = {}
            for x in g.itertuples(index=False):
                if x.prompt in last:
                    y = last[x.prompt]
                    if mode == "lp":
                        p1, p2 = y.pA, x.pA
                    else:
                        s, n = tot.loc[x.prompt]
                        s, n = s - x.A - y.A, n - 2
                        p1 = p2 = (s / n) if n > 0 else np.nan
                    pairs.append(dict(chain=ch, agree=int(x.A == y.A), exp_i=x.KA ** 2 + (1 - x.KA) ** 2,
                                      exp_ii=p1 * p2 + (1 - p1) * (1 - p2), exp_iii=p2 if y.A == 1 else 1 - p2,
                                      earlier_stayed=int(y.new == y.s0), n_other=None if mode == "lp" else n))
                last[x.prompt] = x
        P = pd.DataFrame(pairs)
        r = dict(calls=int(len(d)), repeat_pairs=int(len(P)))
        if len(P):
            Q = P.dropna(subset=["exp_ii"])
            r.update(observed=float(P.agree.mean()), expected_i=float(P.exp_i.mean()),
                     excess_i=float(P.agree.mean() - P.exp_i.mean()),
                     ci_i=boot(P, lambda z: z.agree.mean() - z.exp_i.mean()),
                     pairs_ii=int(len(Q)), observed_ii=float(Q.agree.mean()), expected_ii=float(Q.exp_ii.mean()),
                     excess_ii=float(Q.agree.mean() - Q.exp_ii.mean()),
                     ci_ii=boot(Q, lambda z: z.agree.mean() - z.exp_ii.mean()))
            r["excess_found_prestated_ii"] = bool(r["ci_ii"][0] > 0)
            r.update(share_earlier_stayed=float(P.earlier_stayed.mean()), expected_iii=float(Q.exp_iii.mean()),
                     excess_iii=float(Q.agree.mean() - Q.exp_iii.mean()), ci_iii=boot(Q, lambda z: z.agree.mean() - z.exp_iii.mean()))
            r["excess_found_iii"] = bool(r["ci_iii"][0] > 0)
        out[pid] = r
        print(pid, json.dumps(r), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "identical_requests.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
