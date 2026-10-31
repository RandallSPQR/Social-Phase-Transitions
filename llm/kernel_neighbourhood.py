"""Three-kernel test, K_live-failure investigation continued (owner OK 2026-10-12; post-hoc, read-only, no API calls).

  python llm/kernel_neighbourhood.py

Residual of each live update: r = p(+1 | call) − K_live(c, s0) in position space (logprob endpoints: the call's own
probability; P5: the sampled outcome). Markov in (c, s0) means r is unrelated to anything else.
 N1 wider neighbourhood: regress r (signed toward the agent's own position) on the 2-hop state (mean position of
    neighbours-of-neighbours, excluding i and its contacts, signed the same way), the number of contacts that changed
    since the agent's previous update, and the number of sweeps since its previous update. Chain-cluster bootstrap CIs.
 N2 concurrency: all calls of an endpoint across all chains, binned by wall-clock time (5 s). Intra-bin correlation of
    logit residuals (one-way ANOVA F, permutation p over 1,000 shuffles), and lag-1 correlation of bin means.
 N3 where the persistence comes from: in the measured sweeps, the mean K_live flip probability of the configurations
    the live trajectory visited vs the same quantity along 200 K_live-simulated trajectories (live graphs and starts);
    and the live realised flips vs the sum of the live calls' own flip probabilities (self-consistency).
Output: results/llm/kernel_test/neighbourhood.json
"""
import json, math, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage3, stage3_audit as SA, kernel_test as K, kernel_estimator_check as KE
from endpoints import E

lg = lambda p: np.clip(np.log(np.clip(p, 1e-6, 1) / np.clip(1 - p, 1e-6, 1)), -6, 6)


def boot(d, stat, B=500, seed=1):
    rng = np.random.default_rng(seed); ch = d.chain.unique(); g = {c: d[d.chain == c] for c in ch}
    v = [stat(pd.concat([g[c] for c in rng.choice(ch, len(ch))])) for _ in range(B)]
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X]); return np.linalg.lstsq(X, y, rcond=None)[0][1:]


def features(pid, d):
    """Replay each chain; per update: 2-hop state signed toward own position, contacts changed since the agent's
    previous update, sweeps since previous update."""
    out = []
    for (g, r), dd in d.groupby([d.chain // 2, d.chain % 2]):
        ip, nb, tie = stage3.graph(pid, g)
        N = stage3.N
        nbrs = [set(nb[ip[i]:ip[i + 1]]) for i in range(N)]
        two = [sorted(set().union(*[nbrs[j] for j in nbrs[i]]) - nbrs[i] - {i}) for i in range(N)]
        s = np.load(os.path.join(stage3.OUT, "chains", pid, f"g{g}_r{r}_states.npy"))[0].astype(np.int64).copy()
        last_change = np.full(N, -1); last_upd = np.full(N, -1)
        dd = dd.sort_values(["sweep", "step"])
        h2, nch, gap = [], [], []
        for u, row in enumerate(dd.itertuples(index=False)):
            i = row.agent
            h2.append(row.s0 * np.mean(s[two[i]]) if two[i] else 0.0)
            nch.append(int(sum(last_change[j] > last_upd[i] for j in nbrs[i])) if last_upd[i] >= 0 else -1)
            gap.append((u - last_upd[i]) / N if last_upd[i] >= 0 else np.nan)
            if row.new != s[i]:
                last_change[i] = u
            s[i] = row.new; last_upd[i] = u
        dd = dd.assign(h2=h2, nch=nch, gap=gap); out.append(dd)
    return pd.concat(out)


def main(R=200):
    res = {}
    by_model = {}
    for pid in K.POINTS:
        key, framing, rho, contacts = stage3.POINTS[pid]
        model, tag, mode, roff, _ = E[key]
        d = K.live_rows(pid)
        T, _ = K.live_kernel(d.groupby("idx").agg(S1=("p", "sum"), n=("p", "size")), K.N0_MAIN)
        d["k"] = T[d.idx.values]
        d["r_own"] = np.where(d.s0 > 0, d.p - d.k, d.k - d.p)          # residual toward staying
        r = {}
        # N1
        f = features(pid, d)
        h = f[f.nch >= 0].copy()
        cols = ["h2", "nch", "gap"] if contacts else ["h2", "gap"]
        fit = lambda x: dict(zip(cols, ols(x[cols].values.astype(float), x.r_own.values)))
        est = fit(h)
        cis = {c: boot(h, lambda x, c=c: fit(x)[c]) for c in cols}
        r["N1"] = {c: dict(slope=float(est[c]), ci=cis[c], sd_feature=float(h[c].std())) for c in cols}
        # N3 visited-configuration stability (measured sweeps)
        meas = d[d.sweep >= stage3.N_WARM]
        kflip = np.where(meas.s0 > 0, 1 - meas.k, meas.k)
        callflip = np.where(meas.s0 > 0, 1 - meas.p, meas.p)
        r["N3_live_mean_kernel_flip_prob_visited"] = float(kflip.mean())
        r["N3_live_realised_flip_rate"] = float((meas.new != meas.s0).mean())
        r["N3_live_mean_call_flip_prob"] = float(callflip.mean())
        G = [stage3.graph(pid, g) for g in range(stage3.N_GRAPHS)]
        sims = []
        n = (stage3.N_WARM + stage3.N_MEAS) * stage3.N
        for j in range(R):
            v = []
            for g in range(stage3.N_GRAPHS):
                S0 = K.live_starts(pid, g)
                for rr in (0, 1):
                    oi, on, os_ = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64)
                    KE.record(*G[g], S0[rr].copy(), T, stage3.N_WARM + stage3.N_MEAS,
                              int(stage3._seed("nbhd", pid, j, g, rr) % 2**31), oi, on, os_)
                    m = slice(stage3.N_WARM * stage3.N, n)
                    kk = T[oi[m]]; v.append(np.where(os_[m] > 0, 1 - kk, kk).mean())
            sims.append(np.mean(v))
        sims = np.array(sims)
        r["N3_sim_mean_kernel_flip_prob_visited"] = float(sims.mean())
        r["N3_z_live_vs_sim"] = float((r["N3_live_mean_kernel_flip_prob_visited"] - sims.mean()) / sims.std(ddof=1))
        # collect for N2
        if mode == "lp":
            recs, _ = SA.load_cache(model)
            t = [recs[f"stage3:{pid}:{x.chain // 2}:{x.chain % 2}:{x.sweep}:{x.step}:{x.attempts}"]["t"] for x in d.itertuples(index=False)]
            by_model.setdefault(model, []).append(pd.DataFrame(dict(t=t, lr=lg(d.p.values) - lg(d.k.values), point=pid)))
        res[pid] = r
        print(pid, json.dumps(r), flush=True)
    # N2 concurrency, per endpoint (all its points' calls together)
    n2 = {}
    rng = np.random.default_rng(7)
    for model, parts in by_model.items():
        P = pd.concat(parts); P["bin"] = ((P.t - P.t.min()) // 5).astype(int)
        def F(x, b):
            g = pd.Series(x).groupby(b); n_ = g.size(); m_ = g.mean(); gm = x.mean()
            ssb = (n_ * (m_ - gm) ** 2).sum(); ssw = ((x - g.transform("mean").values) ** 2).sum()
            k_, N_ = len(n_), len(x)
            return (ssb / (k_ - 1)) / (ssw / (N_ - k_))
        x, b = P.lr.values, P["bin"].values
        f0 = F(x, b); perm = [F(rng.permutation(x), b) for _ in range(1000)]
        bm = P.groupby("bin").lr.mean()
        n2[model] = dict(points=sorted(P.point.unique().tolist()), calls=int(len(P)), bins=int(P["bin"].nunique()),
                         F=float(f0), perm_p=float((np.array(perm) >= f0).mean()),
                         icc_approx=float((f0 - 1) / (f0 - 1 + len(P) / P["bin"].nunique())),
                         lag1_corr_bin_means=float(np.corrcoef(bm.values[:-1], bm.values[1:])[0, 1]))
        print(model, n2[model], flush=True)
    json.dump(dict(points=res, N2_concurrency=n2), open(os.path.join(K.OUT, "neighbourhood.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
