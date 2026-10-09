"""Parameter-recovery and null-calibration test for analysis.py on synthetic agents (no API calls).

Truth: logit P = k^-α (b_a Ma − b_r Mr) + γ s0 + h_C + h_L L + h_O O [+ λ Q_aa], plus N(0, σ) per prompt
(order/endpoint noise) and N(0, σ_rep) per replicate. Run: python ../llm/selftest.py
"""
import os, sys, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery, analysis
from scipy.special import expit

TRUE = dict(b_a=1.0, b_r=0.6, gamma=0.5, h_C=0.3, h_L=0.4, h_O=-0.2, alpha=0.0, lam=0.0)


def synth(arms, truth, sigma=0.3, sigma_rep=0.1, seed=1, filt=None, kmax=6):
    rng = np.random.default_rng(seed); rows = []
    for arm in arms:
        for perm_subset in (False, True):
            if perm_subset and arm != "neutral_own":
                continue
            for it in battery.items(arm, filt, extra_perms=perm_subset):
                c = it["counts"]
                if sum(c) > kmax:
                    continue
                Ma, Mr, k = c[0] - c[1], c[2] - c[3], sum(c)
                ka = c[0] + c[1]
                L = 1 if it["mapping"] == 0 else -1; O = 1 if it["mapping"] == it["order"] else -1
                neutral = it["framing"] == "neutral"
                eta = k ** (-truth["alpha"]) * (truth["b_a"] * Ma - truth["b_r"] * Mr) + truth["gamma"] * it["s0"] \
                    + truth["h_C"] + (0 if neutral else truth["h_L"] * L) + truth["h_O"] * O \
                    + truth["lam"] * (Ma ** 2 - ka) / 2 + rng.normal(0, sigma)
                for r in range(2):
                    pP = expit(eta + rng.normal(0, sigma_rep))
                    pA = pP if it["mapping"] == 0 else 1 - pP
                    row = {k2: v for k2, v in it.items() if k2 != "prompt"}
                    row.update(counts="".join(map(str, c)), key="synth", model="synth", tag="synth", rep=r,
                               pA=pA, leak=0.0, censored=0)
                    rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    # 1) recovery, full battery, no interactions
    d = synth(["neutral_own", "political"], TRUE); p = os.path.join(tmp, "s.csv"); d.to_csv(p, index=False)
    for r in analysis.analyse_file(p, B=200, nboot=199):
        c = r["coef"]
        print(r["arm"], {k: f"{v['est']:.3f} [{v['lo']:.3f},{v['hi']:.3f}]" for k, v in c.items()})
        print("   H3", {b: (round(t["p"], 3), round(t["cv_rel_improvement"], 4), t["passes_min_effect"]) for b, t in r["H3"].items()},
              "repeat_sd", round(r["repeat_sd"], 3), "perm_sd", r.get("perm_sd"))
    # 2) null calibration of the H3 wild-bootstrap F test: 40 synthetic datasets, pilot-sized
    filt = __import__("run_stage1").pilot_filter()
    ps = []
    for s in range(40):
        P, st = analysis.prompts(synth(["political"], TRUE, seed=100 + s, filt=filt))
        ps.append(analysis.score_test(P[P.perm == 0], "A1", 199, np.random.default_rng(s))["p"])
    ps = np.array(ps)
    print(f"H3 score-test null (pilot size, 40 sets): rejection rate at .05 = {(ps < .05).mean():.3f}, at .20 = {(ps < .2).mean():.2f}")
    # 3) power + α recovery: truth with interaction λ=0.3 and averaging α=1
    T2 = dict(TRUE, lam=0.3, alpha=1.0)
    d = synth(["neutral_own"], T2, seed=7); d.to_csv(p, index=False)
    r = analysis.analyse_file(p, B=200, nboot=199)[0]
    print("alpha est", r["coef"]["alpha"], "\n H3 pairwise", {k: v for k, v in r["H3"]["pairwise"].items() if k != "coef"},
          "\n Q_aa coef", r["H3"]["pairwise"]["coef"]["Q_aa"])
    # 4) alpha recovery without interactions (averaging agent)
    d = synth(["neutral_own"], dict(TRUE, alpha=1.0), seed=8); d.to_csv(p, index=False)
    r = analysis.analyse_file(p, B=200, nboot=199)[0]
    print("alpha=1, additive: est", {k: round(v, 3) for k, v in r["coef"]["alpha"].items()})
