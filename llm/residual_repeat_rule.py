"""Residual plan amendment 1 §4, re-simulation step (pre-stated trigger: excess under (ii); run with both (ii) and the
selection-corrected (iii) excess). No API calls.

  python llm/residual_repeat_rule.py

Repeat rule, deliberately generous (an upper bound on the mechanism's effect): whenever an updating agent's
(c, s0) equals its (c, s0) at its previous update and that previous update was a stay, its P(stay) is raised by e
(the measured excess agreement at that point; capped at 1). The exact-prompt repeat is rarer than the configuration
repeat (the presentation must also match), so this overstates the effect. K_live (n0 = 20), live graphs and starts,
R = 400; Stouffer of live persistence and flip rate over P1-P4 + nc arms.
Output: results/llm/residual/repeat_rule.json
"""
import json, os, sys
import numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage3, kernel_test as K, kernel_estimator_check as KE

OUT = os.path.join(K.ROOT, "results", "llm", "residual")
ARMS = ["P1", "P2", "P3", "P4", "P2-nc", "P4-nc"]


@njit(cache=False)
def record_rep(indptr, nbr, tie, S0, T, e, nc, n_sweeps, seed, out_new, out_s0):
    np.random.seed(seed)
    N = indptr.shape[0] - 1
    s = S0.copy(); states = np.empty((n_sweeps + 1, N), np.int64); states[0] = s
    lastc = np.full(N, -1); laststay = np.zeros(N, np.int64)
    u = 0
    for t in range(n_sweeps):
        for _ in range(N):
            i = np.random.randint(N); old = s[i]; c = 0
            for q in range(indptr[i], indptr[i + 1]):
                if tie[q] > 0:
                    c += 125 if s[nbr[q]] > 0 else 25
                else:
                    c += 5 if s[nbr[q]] > 0 else 1
            c = c * 2 + (1 if old > 0 else 0)
            key = (old > 0) * 1 if nc else c
            p1 = T[c]
            pst = p1 if old > 0 else 1 - p1
            if lastc[i] == key and laststay[i] == 1:
                pst = min(1.0, pst + e)
            new = old if np.random.random() < pst else -old
            lastc[i] = key; laststay[i] = 1 if new == old else 0
            s[i] = new; out_new[u] = new; out_s0[u] = old; u += 1
        states[t + 1] = s
    return states


def main(R=400):
    ir = json.load(open(os.path.join(OUT, "identical_requests.json")))
    obs = pd.read_csv(os.path.join(K.ROOT, "results/llm/stage3/summary.csv")).set_index("point")
    out = {"per_point": {}, "stouffer": {}}
    zs = {}
    for pid in K.POINTS:
        key, framing, rho, contacts = stage3.POINTS[pid]
        d = K.live_rows(pid)
        T, _ = K.live_kernel(d.groupby("idx").agg(S1=("p", "sum"), n=("p", "size")), K.N0_MAIN)
        G = [stage3.graph(pid, g) for g in range(stage3.N_GRAPHS)]
        n = (stage3.N_WARM + stage3.N_MEAS) * stage3.N
        res = {}
        for name, e in (("ii_prestated_biased", max(0.0, ir[pid]["excess_ii"])), ("iii_corrected", max(0.0, ir[pid]["excess_iii"]))):
            rows = []
            for j in range(R):
                sts, upds = [], []
                for g in range(stage3.N_GRAPHS):
                    S0 = K.live_starts(pid, g)
                    for rr in (0, 1):
                        on, os_ = np.empty(n, np.int64), np.empty(n, np.int64)
                        st = record_rep(*G[g], S0[rr].copy(), T, e, not contacts, stage3.N_WARM + stage3.N_MEAS,
                                        int(stage3._seed("reprule", name, pid, j, g, rr) % 2**31), on, os_)
                        sts.append(st); upds.append(pd.DataFrame(dict(sweep=np.repeat(np.arange(stage3.N_WARM + stage3.N_MEAS), stage3.N), new=on, s0=os_)))
                rows.append(KE.measures_of(pid, sts, upds))
            H = pd.DataFrame(rows)
            res[name] = dict(e=e, **{m: dict(sim_mean=float(H[m].mean()), z=float((obs.loc[pid, m] - H[m].mean()) / H[m].std(ddof=1)))
                                      for m in ("persistence", "flip_rate", "abs_m")})
            if pid in ARMS:
                for m in ("persistence", "flip_rate"):
                    zs.setdefault(name, {}).setdefault(m, []).append(res[name][m]["z"])
        out["per_point"][pid] = res
        print(pid, {k: (round(v["e"], 4), round(v["persistence"]["z"], 2), round(v["flip_rate"]["z"], 2)) for k, v in res.items()}, flush=True)
    out["stouffer"] = {k: {m: float(np.sum(v) / np.sqrt(len(v))) for m, v in d_.items()} for k, d_ in zs.items()}
    out["homogeneous_reference"] = {"persistence": 2.72, "flip_rate": -2.77}
    json.dump(out, open(os.path.join(OUT, "repeat_rule.json"), "w"), indent=1)
    print(json.dumps(out["stouffer"], indent=1))


if __name__ == "__main__":
    main()
