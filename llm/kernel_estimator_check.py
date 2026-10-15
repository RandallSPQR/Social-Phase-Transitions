"""Three-kernel test, post-hoc check of the K_live ESTIMATOR (not in the plan; read-only, no API calls).

  python llm/kernel_estimator_check.py

1. Variants of K_live on the real data: fixed point estimate (no chain bootstrap) with n0 = 20, and the raw kernel
   (n0 = 0; prior only for configurations never seen live).
2. Known-answer test: take the fixed n0 = 20 K_live table as the TRUTH, generate 20 synthetic 'live' datasets with it
   (live graphs and starts, simulator update rule, recording every update), push each through the pre-registered
   K_live pipeline (chain bootstrap, n0 = 20, R = 200) and record the z of the synthetic observed measures. A correct
   estimator gives z ~ N(0, 1); a mean offset is estimator bias.
Output: results/llm/kernel_test/estimator_check.json
"""
import json, os, sys
import numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage3, surrogate as S, kernel_test as K
from endpoints import E


@njit(cache=False)
def record(indptr, nbr, tie, S0, T, n_sweeps, seed, out_idx, out_new, out_s0):
    """Same update rule as the kernel simulator; records (config index, new, s0) of every update of one replica."""
    np.random.seed(seed)
    N = indptr.shape[0] - 1
    s = S0.copy(); states = np.empty((n_sweeps + 1, N), np.int64); states[0] = s
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
            new = 1 if np.random.random() < T[c] else -1
            s[i] = new; out_idx[u] = c; out_new[u] = new; out_s0[u] = old; u += 1
        states[t + 1] = s
    return states


def measures_of(pid, chains_states, chains_upd):
    """Live-style measures from recorded replicas via the harness's graph_measures (same code as Stage 3)."""
    key, framing, rho, contacts = stage3.POINTS[pid]
    fl = pd.read_csv(os.path.join(K.ROOT, "results/llm/stage2/floor.csv"))
    floor = fl[(fl.graph == "rrg") & (fl.N == stage3.N)].groupby("rho").unsat_floor.mean().loc[rho]
    per = []
    for g in range(stage3.N_GRAPHS):
        st = np.stack([chains_states[g * 2], chains_states[g * 2 + 1]])
        upd = pd.concat([chains_upd[g * 2], chains_upd[g * 2 + 1]])
        m = stage3.graph_measures(*stage3.graph(pid, g), st, upd); m["excess_unsat"] = m["unsatisfied"] - floor
        per.append(m)
    return {m: float(np.mean([x[m] for x in per])) for m in K.MEAS}


def main(n_synth=20, R=200):
    out = {}
    obs = pd.read_csv(os.path.join(K.ROOT, "results/llm/stage3/summary.csv")).set_index("point")
    for pid in K.POINTS:
        key, framing, rho, contacts = stage3.POINTS[pid]
        lp = E[key][2] == "lp"
        d = K.live_rows(pid); st_all = d.groupby("idx").agg(S1=("p", "sum"), n=("p", "size"))
        T20, prior = K.live_kernel(st_all, 20)
        T0, _ = K.live_kernel(st_all, 0)
        res = {}
        for name, T in (("fixed_n0_20", T20), ("fixed_raw_n0_0", T0)):
            df = K.simulate(pid, lambda run, g, T=T: T, 400, "estcheck-" + name)
            res[name] = {m: float((obs.loc[pid, m] - df[m].mean()) / df[m].std(ddof=1)) for m in K.MEAS}
        # known-answer test of the full pre-registered pipeline
        G = [stage3.graph(pid, g) for g in range(stage3.N_GRAPHS)]
        zs = []
        for j in range(n_synth):
            sts, upds, rows = [], [], []
            for g in range(stage3.N_GRAPHS):
                S0 = K.live_starts(pid, g)
                for r in (0, 1):
                    n = (stage3.N_WARM + stage3.N_MEAS) * stage3.N
                    oi, on, os_ = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64)
                    states = record(*G[g], S0[r].copy(), T20, stage3.N_WARM + stage3.N_MEAS,
                                    int(stage3._seed("estcheck", pid, j, g, r) % 2**31), oi, on, os_)
                    sweep = np.repeat(np.arange(stage3.N_WARM + stage3.N_MEAS), stage3.N)
                    if not contacts:
                        oi = oi % 2
                    p = T20[oi] if lp else (on == 1).astype(float)
                    upd = pd.DataFrame(dict(sweep=sweep, new=on, s0=os_))
                    sts.append(states); upds.append(upd)
                    rows.append(pd.DataFrame(dict(chain=g * 2 + r, idx=oi, p=p)))
            synth_obs = measures_of(pid, sts, upds)
            cs = K.chain_stats(pd.concat(rows)); chains = sorted(cs)
            rng = np.random.default_rng(stage3._seed("estcheck-boot", pid, j))
            boots = [K.live_kernel(K.pooled(cs, list(rng.choice(chains, len(chains)))), 20)[0] for _ in range(R)]
            df = K.simulate(pid, lambda run, g: boots[run], R, f"estcheck-sim-{j}")
            zs.append({m: (synth_obs[m] - df[m].mean()) / df[m].std(ddof=1) for m in K.MEAS})
        Z = pd.DataFrame(zs)
        res["known_answer_z_mean"] = Z.mean().round(3).to_dict()
        res["known_answer_z_sd"] = Z.std(ddof=1).round(3).to_dict()
        res["known_answer_share_inside_90"] = ((Z.abs() <= 1.645).mean()).round(3).to_dict()
        out[pid] = res
        print(pid, json.dumps(res), flush=True)
    json.dump(out, open(os.path.join(K.OUT, "estimator_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
