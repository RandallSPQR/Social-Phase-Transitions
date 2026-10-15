"""Residual test 1c (triggered: Fisher p < 0.05 in 1b) plus robustness of 1b (labelled post-hoc).

  python llm/residual_heterogeneity_c.py

Robustness of b) (not pre-stated): the same statistic with (i) labels shuffled only within (point, c, s0) cells, so
agents that sit in high-variance configurations cannot create spurious between-agent variance, and (ii) the raw
kernel (n0 = 0), so smoothing bias in rare configurations cannot attach to nodes with rare tie patterns.
c) (pre-stated): tau = between-agent SD of a logit offset on P(stay), by method of moments from b)'s pre-stated
statistic; simulate K_live with per-agent offsets ~ N(0, tau^2), fixed per (replica, node), on the live graphs and
starts (R = 400); Stouffer z of live persistence and flip rate vs heterogeneous and homogeneous sims, P1-P4 + nc.
Output: results/llm/residual/heterogeneity_c.json
"""
import json, os, sys
import numpy as np, pandas as pd
from numba import njit
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage3, kernel_test as K, kernel_estimator_check as KE, residual_heterogeneity as RH

ARMS = ["P1", "P2", "P3", "P4", "P2-nc", "P4-nc"]


def resid(pid, n0):
    d = K.live_rows(pid)
    T, _ = K.live_kernel(d.groupby("idx").agg(S1=("p", "sum"), n=("p", "size")), n0)
    k = T[d.idx.values]
    stay = np.where(d.s0 > 0, d.p, 1 - d.p); kst = np.where(d.s0 > 0, k, 1 - k)
    return d, stay - kst, kst, T


def strat_perm(lab, strata, rng):
    lab = lab.copy()
    for s in np.unique(strata):
        ii = np.flatnonzero(strata == s); lab[ii] = lab[rng.permutation(ii)]
    return lab


@njit(cache=False)
def record_het(indptr, nbr, tie, S0, T, delta, n_sweeps, seed, out_idx, out_new, out_s0):
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
            p1 = T[c]
            pst = p1 if old > 0 else 1 - p1
            pst = min(max(pst, 1e-9), 1 - 1e-9)
            ls = np.log(pst / (1 - pst)) + delta[i]
            pst = 1 / (1 + np.exp(-ls))
            new = old if np.random.random() < pst else -old
            s[i] = new; out_idx[u] = c; out_new[u] = new; out_s0[u] = old; u += 1
        states[t + 1] = s
    return states


def main(R=400, B=1000):
    het = json.load(open(os.path.join(RH.OUT, "heterogeneity.json")))
    out = {"robustness_b": {}, "tau": {}, "c": {}}
    obs = pd.read_csv(os.path.join(K.ROOT, "results/llm/stage3/summary.csv")).set_index("point")
    for pid in K.POINTS:
        rb = {}
        for n0 in (K.N0_MAIN, 0):
            d, r, kst, T = resid(pid, n0)
            strata = d.idx.values
            rng = np.random.default_rng(stage3._seed("het-strat", pid, n0))
            for name, u in (("agent", d.chain * 1000 + d.agent), ("graph_position", (d.chain // 2) * 1000 + d.agent)):
                lab = pd.factorize(u)[0]; o = RH.ss_between(lab, r)
                null = np.array([RH.ss_between(strat_perm(lab, strata, rng), r) for _ in range(B // 4)])
                rb[f"{name}_n0{n0}_stratified"] = dict(z=float((o - null.mean()) / null.std()), p=float(((null >= o).sum() + 1) / (len(null) + 1)))
        out["robustness_b"][pid] = rb
        print(pid, {k: (round(v["z"], 2), round(v["p"], 3)) for k, v in rb.items()}, flush=True)
        # tau (pre-stated: from b's statistic, agent unit, n0 = 20)
        d, r, kst, T = resid(pid, K.N0_MAIN)
        b = het["b_variance"][pid]["agent"]
        lab = pd.factorize(d.chain * 1000 + d.agent)[0]; n_a = np.bincount(lab); N = len(r)
        tau_p2 = max(0.0, (b["obs_ss"] - b["null_mean"]) / (N - (n_a ** 2).sum() / N))
        slope = float(np.mean(kst * (1 - kst)))                 # d P(stay) / d logit, averaged over updates
        out["tau"][pid] = dict(tau_prob=float(np.sqrt(tau_p2)), tau_logit=float(np.sqrt(tau_p2) / max(slope, 1e-6)), mean_dP_dlogit=slope)
    # c) heterogeneous simulations
    zs = {"het": {"persistence": [], "flip_rate": []}, "hom": {"persistence": [], "flip_rate": []}}
    for pid in K.POINTS:
        key, framing, rho, contacts = stage3.POINTS[pid]
        d, r, kst, T = resid(pid, K.N0_MAIN)
        tau = out["tau"][pid]["tau_logit"]
        G = [stage3.graph(pid, g) for g in range(stage3.N_GRAPHS)]
        n = (stage3.N_WARM + stage3.N_MEAS) * stage3.N
        rows = []
        for j in range(R):
            sts, upds = [], []
            rng = np.random.default_rng(stage3._seed("het-sim", pid, j))
            for g in range(stage3.N_GRAPHS):
                S0 = K.live_starts(pid, g)
                for rr in (0, 1):
                    delta = rng.normal(0, tau, stage3.N)
                    oi, on, os_ = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64)
                    st = record_het(*G[g], S0[rr].copy(), T, delta, stage3.N_WARM + stage3.N_MEAS,
                                    int(stage3._seed("het-sim-dyn", pid, j, g, rr) % 2**31), oi, on, os_)
                    sts.append(st); upds.append(pd.DataFrame(dict(sweep=np.repeat(np.arange(stage3.N_WARM + stage3.N_MEAS), stage3.N), new=on, s0=os_)))
            rows.append(KE.measures_of(pid, sts, upds))
        H = pd.DataFrame(rows); H.to_csv(os.path.join(RH.OUT, f"sims_{pid}_K_live_het.csv"), index=False)
        hom = pd.read_csv(os.path.join(K.OUT, f"sims_{pid}_K_live_n020.csv"))
        res = {}
        for m in K.MEAS:
            res[m] = dict(observed=float(obs.loc[pid, m]), het_mean=float(H[m].mean()), het_lo=float(np.percentile(H[m], 5)), het_hi=float(np.percentile(H[m], 95)),
                          z_het=float((obs.loc[pid, m] - H[m].mean()) / H[m].std(ddof=1)), hom_mean=float(hom[m].mean()),
                          z_hom=float((obs.loc[pid, m] - hom[m].mean()) / hom[m].std(ddof=1)))
        out["c"][pid] = res
        if pid in ARMS:
            for m in ("persistence", "flip_rate"):
                zs["het"][m].append(res[m]["z_het"]); zs["hom"][m].append(res[m]["z_hom"])
        print(pid, "tau_logit", round(tau, 3), {m: (round(res[m]["z_hom"], 2), round(res[m]["z_het"], 2)) for m in ("persistence", "flip_rate", "abs_m")}, flush=True)
    st = {k: {m: float(np.sum(v) / np.sqrt(len(v))) for m, v in d_.items()} for k, d_ in zs.items()}
    out["stouffer_P1_P4_nc"] = st
    out["heterogeneity_explains"] = bool(st["het"]["persistence"] < 1.645 and st["hom"]["persistence"] >= 1.645)
    json.dump(out, open(os.path.join(RH.OUT, "heterogeneity_c.json"), "w"), indent=1)
    print(json.dumps({"stouffer": st, "heterogeneity_explains": out["heterogeneity_explains"]}, indent=1))


if __name__ == "__main__":
    main()
