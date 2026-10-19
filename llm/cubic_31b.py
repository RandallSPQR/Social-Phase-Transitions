"""Exploratory (owner request 2026-10-11): what is the cubic H3 effect in local Gemma-31B?

  python llm/cubic_31b.py [--key gemma-31b-local] [--B 1000]

Per arm (neutral_own, political: the cells where H3 cubic is supported; workplace as contrast), main battery (perm 0),
analysis.py's own preparation, design and soft-label fits:
1. Dominant cubic terms: the A2 fit (A0 + T_aaa, T_rrr, T_aar, T_arr) with cluster-bootstrap 95% CIs (B draws, cells
   resampled); and, per term, the grouped-CV cross-entropy gain of adding that term alone to A0.
2. Majority-rule response: the linear social terms (b_a M_a − b_r M_r) are replaced by sign(M_a), sign(M_r)
   (0 at a tie), fields unchanged: logit = c_a sgn(M_a) − c_r sgn(M_r) + Z θ. Compared by 10-fold grouped CV
   (folds by multiset, analysis.folds_by_multiset) against A0, the cubic A2, and A0 + majority.
Labelled exploratory; it cannot alter H1–H5. Output: results/llm/amendment5/cubic_<key>.json.
"""
import argparse, json, os, sys
import numpy as np
from scipy.special import expit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A

ROOT = A.os.path.dirname(A.os.path.dirname(A.os.path.abspath(A.__file__)))
CUBIC = ["T_aaa", "T_rrr", "T_aar", "T_arr"]


def designs(P):
    X0, n0 = A.design(P, "A0"); X2, n2 = A.design(P, "A2")
    sa, sr = np.sign(P["Ma"].values).astype(float), -np.sign(P["Mr"].values).astype(float)
    Z = X0[:, 2:]; nz = n0[2:]
    D = {"A0 (linear)": (X0, n0), "A2 (cubic)": (X2, n2),
         "majority rule": (np.column_stack([sa, sr, Z]), ["c_a", "c_r"] + nz),
         "A0 + majority": (np.column_stack([X0, sa, sr]), n0 + ["c_a", "c_r"])}
    for t in CUBIC:
        j = n2.index(t)
        D[f"A0 + {t}"] = (np.column_stack([X0, X2[:, j]]), n0 + [t])
    return D


def cv(P, X, K=10):
    y, w = P["y"].values, P["w"].values.astype(float)
    f = A.folds_by_multiset(P, K); pred = np.empty(len(y))
    for k in range(K):
        tr, te = f != k, f == k
        if te.sum() == 0:
            continue
        pred[te] = expit(X[te] @ A.fit_ce(X[tr], y[tr], w[tr]))
    return A.ce(y, pred, w)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--key", default="gemma-31b-local"); ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    d = A.load(os.path.join(ROOT, "results/llm/stage1/main", a.key + ".csv"))
    out = {"key": a.key, "B": a.B, "arms": {}}
    for arm in ("neutral_own", "political", "workplace"):
        P, _ = A.prompts(d[d["arm"] == arm]); P = P[P["perm"] == 0].copy()
        y, w = P["y"].values, P["w"].values.astype(float)
        D = designs(P)
        X2, n2 = D["A2 (cubic)"]; b2 = A.fit_ce(X2, y, w)
        rng = np.random.default_rng(7); mem = A.cluster_index(P); draws = []
        for _ in range(a.B):
            idx = np.concatenate([mem[i] for i in rng.integers(0, len(mem), len(mem))])
            draws.append(A.fit_ce(X2[idx], y[idx], w[idx], b2))
        draws = np.array(draws)
        terms = {t: dict(est=float(b2[n2.index(t)]), lo=float(np.percentile(draws[:, n2.index(t)], 2.5)),
                         hi=float(np.percentile(draws[:, n2.index(t)], 97.5))) for t in CUBIC}
        ce = {name: cv(P, X) for name, (X, _) in D.items()}
        base = ce["A0 (linear)"]
        maj_X, maj_n = D["majority rule"]; bm = A.fit_ce(maj_X, y, w)
        out["arms"][arm] = dict(
            n_prompts=int(len(P)), cubic_terms=terms,
            cv_ce=ce, cv_rel_gain_vs_A0={k: (base - v) / base for k, v in ce.items()},
            majority_rule_coef=dict(zip(maj_n, map(float, bm))))
        print(f"\n== {a.key} {arm} ({len(P)} prompts)")
        for t, v in terms.items():
            print(f"  {t}: {v['est']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]   CV gain alone {(base - ce['A0 + ' + t]) / base:+.4f}")
        for k in ("A0 (linear)", "A2 (cubic)", "majority rule", "A0 + majority"):
            print(f"  CV CE {k:15s} {ce[k]:.4f}  (rel. gain vs A0 {(base - ce[k]) / base:+.4f})")
    os.makedirs(os.path.join(ROOT, "results/llm/amendment5"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results/llm/amendment5", f"cubic_{a.key}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
