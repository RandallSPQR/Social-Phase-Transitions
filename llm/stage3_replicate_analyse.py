"""Pre-registered analysis of the option B replicate (llm/RESIDUAL_TEST_PLAN.md + amendment 1). Written and committed
before the run finished; tested only on the mock run's output.

  python llm/stage3_replicate_analyse.py [--mock]

1. Measures per point (stage3.graph_measures on the replicate chains, mean of 5 graphs; excess unsatisfied as Stage 3).
2. Primary: Stouffer Z over P1-P4, P2-nc, P4-nc of z_i = (observed persistence - reference mean) / reference SD,
   reference = the committed predictions (K_avg; nc arms: the k = 0 K_avg battery). "Excess repeats" iff Z >= 1.645.
3. Per point: observed mean of 5 inside the committed 90% interval, every measure; flip rate reported with persistence.
4. Repeat-or-move: P1 |m| below its interval? P3 persistence above its interval?
5. Sentinels: drift trace per endpoint (per round, mean over sentinels of the deviation from each sentinel's run mean,
   logit scale toward position P; P5: share toward P); ratio of round variance to measurement noise (logprob: Stage 1
   repeat SD^2; P5: binomial variance of 5 samples); lag-1 autocorrelation of the round series. Correlation test: per
   replicate chain, mean drift during the chain's time window vs the chain's persistence and signed m (toward P);
   Spearman per endpoint, Fisher-z combined over endpoints, two-sided p < 0.05.
Output: results/llm/stage3_rep1[_mock]/analysis.json (+ drift_trace.csv, chain_windows.csv)
"""
import argparse, json, math, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client, stage3, kernel_test as K, stage3_replicate as SR
from endpoints import E

ARMS = ["P1", "P2", "P3", "P4", "P2-nc", "P4-nc"]
lg = lambda p: np.clip(np.log(np.clip(p, 1e-6, 1) / np.clip(1 - p, 1e-6, 1)), -6, 6)


def chain_files(out, pid, g, r):
    d = os.path.join(out, "chains", pid)
    return pd.read_csv(os.path.join(d, f"g{g}_r{r}.csv.gz")), np.load(os.path.join(d, f"g{g}_r{r}_states.npy")).astype(np.int64)


def main(mock):
    cfg = SR.Cfg(mock); out = cfg.out
    pred = json.load(open(os.path.join(K.ROOT, "results/llm/replicate/predictions.json")))["points"]
    ref = {p: pred[p] for p in K.POINTS if "-nc" not in p}
    ref.update({p: pred[p + " (K_avg k=0, primary)"] for p in ("P2-nc", "P4-nc")})
    fl = pd.read_csv(os.path.join(K.ROOT, "results/llm/stage2/floor.csv"))
    floor = fl[(fl.graph == "rrg") & (fl.N == stage3.N)].groupby("rho").unsat_floor.mean()
    res = {"per_point": {}}; chain_rows = []
    for pid in K.POINTS:
        key, framing, rho, contacts = stage3.POINTS[pid]
        per = []
        for g in range(stage3.N_GRAPHS):
            ups, sts = [], []
            for r in (0, 1):
                df, st = chain_files(out, pid, g, r); ups.append(df); sts.append(st)
                ref_s = st[stage3.N_WARM]
                pers = float(np.mean([(st[t] * ref_s).mean() for t in range(stage3.N_WARM + 1, stage3.N_WARM + stage3.N_MEAS + 1)]))
                msig = float(np.mean([st[t].mean() for t in range(stage3.N_WARM + 1, stage3.N_WARM + stage3.N_MEAS + 1)]))
                chain_rows.append(dict(point=pid, endpoint=key, graph=g, replica=r, persistence=pers, m=msig))
            m = stage3.graph_measures(*stage3.graph(pid, g), np.stack(sts), pd.concat(ups))
            m["excess_unsat"] = m["unsatisfied"] - floor.loc[rho]; per.append(m)
        obs = {m: float(np.mean([x[m] for x in per])) for m in K.MEAS}
        rp = ref[pid]
        res["per_point"][pid] = {m: dict(observed=obs[m], ref_mean=rp[m]["mean"], ref_lo=rp[m]["lo"], ref_hi=rp[m]["hi"],
                                         inside=bool(rp[m]["lo"] <= obs[m] <= rp[m]["hi"]), z=(obs[m] - rp[m]["mean"]) / rp[m]["sd"])
                                 for m in K.MEAS}
    zs = [res["per_point"][p]["persistence"]["z"] for p in ARMS]
    Z = float(np.sum(zs) / np.sqrt(len(zs)))
    res["primary"] = dict(arms=ARMS, z=dict(zip(ARMS, zs)), stouffer_Z=Z, p_one_sided=float(stats.norm.sf(Z)), excess_repeats=bool(Z >= 1.645))
    res["flip_rate_stouffer"] = float(np.sum([res["per_point"][p]["flip_rate"]["z"] for p in ARMS]) / np.sqrt(len(ARMS)))
    pp = res["per_point"]
    res["repeat_or_move"] = dict(P1_abs_m_below=bool(pp["P1"]["abs_m"]["observed"] < pp["P1"]["abs_m"]["ref_lo"]),
                                 P1_abs_m_z=pp["P1"]["abs_m"]["z"],
                                 P3_persistence_above=bool(pp["P3"]["persistence"]["observed"] > pp["P3"]["persistence"]["ref_hi"]),
                                 P3_persistence_z=pp["P3"]["persistence"]["z"])
    # ---- sentinels
    sp = json.load(open(os.path.join(out, "sentinel_prompts.json")))
    drift_rows, sent = [], {}
    for name, pts in SR.STREAMS:
        pid = pts[0]; key = stage3.POINTS[pid][0]; mode = E[key][2]
        f = os.path.join(out, f"sentinels_{key}.csv")
        if not os.path.exists(f):
            continue
        S = pd.read_csv(f).dropna(subset=["pA"])
        S["mapping"] = S.sentinel.map(lambda j: sp[pid][j]["mapping"])
        if mode == "lp":
            S["y"] = np.where(S.mapping == 0, lg(S.pA.values), -lg(S.pA.values))          # logit toward P
            fits = json.load(open(os.path.join(K.ROOT, "results/llm/fits/surrogate", key + ".json")))
            noise = float(fits["arms"][{"neutral": "neutral_own", "political": "political", "workplace": "workplace"}[stage3.POINTS[pid][1]]]["repeat_sd"]) ** 2
        else:
            S["y"] = np.where(S.mapping == 0, S.pA, 1 - S.pA); noise = None
        g = S.groupby(["sentinel", "round"]).agg(y=("y", "mean"), t=("t", "mean"), n=("y", "size")).reset_index()
        g["dev"] = g.y - g.groupby("sentinel").y.transform("mean")
        tr = g.groupby("round").agg(drift=("dev", "mean"), t=("t", "mean")).reset_index(); tr["endpoint"] = key
        drift_rows.append(tr)
        if mode != "lp":
            pbar = g.groupby("sentinel").y.transform("mean"); noise = float((pbar * (1 - pbar) / g.n).mean())
        rv = g.groupby("sentinel").dev.var(ddof=1)
        sent[key] = dict(rounds=int(tr["round"].nunique()), var_round_over_noise=float(rv.mean() / noise) if noise else None,
                         lag1_autocorr=float(tr.drift.autocorr(1)) if len(tr) > 2 else None)
    D = pd.concat(drift_rows) if drift_rows else pd.DataFrame()
    D.to_csv(os.path.join(out, "drift_trace.csv"), index=False)
    # chain time windows from the cache (first/last call of each chain) and mean drift inside
    C = pd.DataFrame(chain_rows)
    if len(D):
        win = {}
        for name, pts in (SR.STREAMS if not mock else []):
            model = E[stage3.POINTS[pts[0]][0]][0]
            for l in open(client._path(model)):
                if '"rep":"stage3-rep1:' in l:
                    r = json.loads(l); _, p, g_, r_ = r["rep"].split(":")[:4]
                    k_ = (p, int(g_), int(r_)); a, b = win.get(k_, (r["t"], r["t"])); win[k_] = (min(a, r["t"]), max(b, r["t"]))
        if mock:      # mock run has no cache: split each stream's sentinel time span evenly over its chains, in run order
            for name, pts in SR.STREAMS:
                key = stage3.POINTS[pts[0]][0]; tt = D[D.endpoint == key].t
                ks = [(p, g, r) for p in pts for g in range(stage3.N_GRAPHS) for r in (0, 1)]
                edges = np.linspace(tt.min(), tt.max(), len(ks) + 1)
                win.update({k_: (edges[i], edges[i + 1]) for i, k_ in enumerate(ks)})
        C["t0"] = [win.get((x.point, x.graph, x.replica), (np.nan, np.nan))[0] for x in C.itertuples()]
        C["t1"] = [win.get((x.point, x.graph, x.replica), (np.nan, np.nan))[1] for x in C.itertuples()]
        C["drift"] = [D[(D.endpoint == x.endpoint) & (D.t >= x.t0) & (D.t <= x.t1)].drift.mean() for x in C.itertuples()]
        corr = {}
        for key, g in C.dropna(subset=["drift"]).groupby("endpoint"):
            if len(g) >= 4:
                corr[key] = {m: dict(rho=float(stats.spearmanr(g.drift, g[m])[0]), n=int(len(g))) for m in ("persistence", "m")}
        comb = {}
        for m in ("persistence", "m"):
            zz = [np.arctanh(np.clip(v[m]["rho"], -0.999, 0.999)) * math.sqrt(max(v[m]["n"] - 3, 1)) for v in corr.values()]
            Zc = float(np.sum(zz) / np.sqrt(len(zz))) if zz else float("nan")
            comb[m] = dict(Z=Zc, p_two_sided=float(2 * stats.norm.sf(abs(Zc))) if zz else None)
        res["sentinel_correlation"] = dict(per_endpoint=corr, combined=comb)
    C.to_csv(os.path.join(out, "chain_windows.csv"), index=False)
    res["sentinels"] = sent
    json.dump(res, open(os.path.join(out, "analysis.json"), "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("primary", "repeat_or_move", "sentinels")}, indent=1, default=float))
    print(pd.DataFrame({p: {m: f"{v['observed']:.3f} [{v['ref_lo']:.3f}, {v['ref_hi']:.3f}]{'' if v['inside'] else ' OUT'}" for m, v in d.items()}
                        for p, d in res["per_point"].items()}).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mock", action="store_true")
    main(ap.parse_args().mock)
