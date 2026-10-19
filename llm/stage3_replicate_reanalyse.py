"""Replicate reanalysis on the graphs actually used (DEVIATION, 2026-10-13; see EXECUTION_LOG entry 51).

  python llm/stage3_replicate_reanalyse.py

The pods built their graphs with networkx 3.3 (preinstalled in the image; the unpinned `pip install networkx` kept
it), the session container with networkx 3.7. Same generator and seeds, different random-regular graphs and tie signs.
69/70 replicate chains ran on the networkx-3.3 graphs; P3 g0 r0 (finished in the session container) on the original
3.7 graphs (`results/llm/stage3_rep1/graphs/chain_graph.json`, verified against every logged configuration).
The pre-registered analysis (llm/stage3_replicate_analyse.py) measured and predicted on the 3.7 graphs, so its
graph-dependent parts are invalid for these chains. This script repeats it mechanically on the actual graphs:
 - observed measures per replica on that replica's own graph (|q| from the two replicas' states);
 - reference = the SAME committed kernel tables (K_avg for P1-P5 from k_avg_calls.csv; the k = 0 K_avg battery for the
   nc arms from k_avg_nc_calls.csv), the SAME replicate starts, simulated with the same simulator on the actual graphs
   (R = 400; a replica whose graph differs from its partner's is simulated on its own graph);
 - the SAME pre-registered rules: Stouffer Z of persistence over P1-P4 + P2-nc + P4-nc, excess repeats iff Z >= 1.645;
   per-point 90% intervals; repeat-or-move for P1 |m| and P3 persistence.
Sentinel results and per-step kernel comparisons do not depend on the graph and stand as computed.
Output: results/llm/stage3_rep1/reanalysis.json
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage3, surrogate as S, kernel_test as K, k_avg as KA
from replicate_predictions import starts as rep_starts
from endpoints import E

OUT = os.path.join(K.ROOT, "results/llm/stage3_rep1")
GDIR = os.path.join(OUT, "graphs")
ARMS = ["P1", "P2", "P3", "P4", "P2-nc", "P4-nc"]
R = 400


def graph_of(pid, g, r, cg):
    z = np.load(os.path.join(GDIR, f"{pid.split('-')[0]}_g{g}_{cg[f'{pid}:{g}:{r}']}.npz"))
    return z["ip"], z["nb"], z["tie"]


def replica_measures(ip, nb, tie, st, upd):
    """One replica: the time-averaged measures of stage3.graph_measures / surrogate.run for a single replica."""
    ref = st[stage3.N_WARM]; recv = np.repeat(np.arange(stage3.N), np.diff(ip)); acc = []
    for t in range(stage3.N_WARM + 1, stage3.N_WARM + stage3.N_MEAS + 1):
        s = st[t]; agree = s[recv] == s[nb]
        un = (((tie > 0) & ~agree) | ((tie < 0) & agree)).sum() / max(ip[-1], 1)
        acc.append((abs(s.mean()), s.mean(), (s * ref).mean(), un))
    a = np.mean(acc, axis=0); meas = upd[upd.sweep >= stage3.N_WARM]
    return dict(abs_m=a[0], m=a[1], persistence=a[2], unsatisfied=a[3], flip_rate=float((meas.new != meas.s0).mean()))


def qbar(st0, st1):
    return float(np.mean([abs((st0[t] * st1[t]).mean()) for t in range(stage3.N_WARM + 1, stage3.N_WARM + stage3.N_MEAS + 1)]))


def tables(pid, calls, nccalls):
    key = stage3.POINTS[pid][0]
    if pid.endswith("-nc"):
        rho = stage3.POINTS[pid][2]
        g = nccalls[(nccalls.point == pid) & (nccalls.leak <= stage3.LEAK_MAX)].groupby(["s0", "mapping", "order"]).pA.mean().reset_index()
        g["P"] = np.where(g.mapping == 0, g.pA, 1 - g.pA)
        T = np.full(K.NT, np.nan)
        for s0 in (1, -1):
            T[K.idx((0, 0, 0, 0), s0)] = g[g.s0 == s0].P.mean()
        T = K.expand_nc(T, rho)
        return lambda run: T
    if E[key][2] == "sample":                      # P5: Beta draw per prompt per run, as the committed predictions
        rng = np.random.default_rng(stage3._seed("rep1-kavg-beta", pid)); tabs = [KA.kavg_table(pid, calls, rng) for _ in range(R)]
        return lambda run: tabs[run]
    T0 = KA.kavg_table(pid, calls)
    return lambda run: T0


def main():
    cg = json.load(open(os.path.join(GDIR, "chain_graph.json")))
    calls = pd.read_csv(os.path.join(K.OUT, "k_avg_calls.csv"), dtype={"counts": str, "seq": str}); calls["counts"] = calls.counts.str.zfill(4)
    nccalls = pd.read_csv(os.path.join(K.ROOT, "results/llm/replicate/k_avg_nc_calls.csv"))
    fl = pd.read_csv(os.path.join(K.ROOT, "results/llm/stage2/floor.csv"))
    floor = fl[(fl.graph == "rrg") & (fl.N == stage3.N)].groupby("rho").unsat_floor.mean()
    pre = json.load(open(os.path.join(K.ROOT, "results/llm/replicate/predictions.json")))["points"]
    prim = {p["id"].split(" ")[0]: p["primary"] for p in json.load(open(os.path.join(K.ROOT, "results/llm/stage2/stage3_predictions.json")))}
    prim.update({"P2-nc": ["m", "persistence"], "P4-nc": ["m", "persistence"]})
    sim = K.sim()
    res = {"per_point": {}}
    for pid in K.POINTS:
        rho = stage3.POINTS[pid][2]; fl_ = floor.loc[rho]; T = tables(pid, calls, nccalls); S0 = rep_starts(pid)
        # observed
        per = []
        for g in range(stage3.N_GRAPHS):
            rm, sts = [], []
            for r in (0, 1):
                df = pd.read_csv(os.path.join(OUT, "chains", pid, f"g{g}_r{r}.csv.gz"))
                st = np.load(os.path.join(OUT, "chains", pid, f"g{g}_r{r}_states.npy")).astype(np.int64)
                rm.append(replica_measures(*graph_of(pid, g, r, cg), st, df)); sts.append(st)
            m = {k: float(np.mean([x[k] for x in rm])) for k in rm[0]}; m["abs_q"] = qbar(*sts); m["excess_unsat"] = m["unsatisfied"] - fl_
            per.append(m)
        obs = {k: float(np.mean([x[k] for x in per])) for k in K.MEAS}
        # reference simulations on the actual graphs
        rows = []
        for run in range(R):
            Tr = T(run); gm = []
            for g in range(stage3.N_GRAPHS):
                same = cg[f"{pid}:{g}:0"] == cg[f"{pid}:{g}:1"]
                if same:
                    ip, nb, tie = graph_of(pid, g, 0, cg)
                    o = dict(zip(S.MEASURES, sim(ip, nb, tie, S0[g], Tr, stage3.N_WARM, stage3.N_MEAS,
                                                 int(stage3._seed("rep1-reanalysis", pid, run, g) % 2**31))))
                else:                              # replicas on different graphs: each on its own graph
                    parts = []
                    for r in (0, 1):
                        ip, nb, tie = graph_of(pid, g, r, cg)
                        parts.append(dict(zip(S.MEASURES, sim(ip, nb, tie, np.stack([S0[g][r], S0[g][r]]), Tr, stage3.N_WARM,
                                                              stage3.N_MEAS, int(stage3._seed("rep1-reanalysis", pid, run, g, r) % 2**31)))))
                    o = {k: float(np.mean([p_[k] for p_ in parts])) for k in parts[0]}
                o["excess_unsat"] = o["unsatisfied"] - fl_; gm.append(o)
            rows.append({k: float(np.mean([x[k] for x in gm])) for k in K.MEAS})
        D = pd.DataFrame(rows)
        res["per_point"][pid] = {k: dict(observed=obs[k], ref_mean=float(D[k].mean()), ref_sd=float(D[k].std(ddof=1)),
                                         ref_lo=float(np.percentile(D[k], 5)), ref_hi=float(np.percentile(D[k], 95)),
                                         inside=bool(np.percentile(D[k], 5) <= obs[k] <= np.percentile(D[k], 95)),
                                         z=float((obs[k] - D[k].mean()) / D[k].std(ddof=1)),
                                         prereg_ref_mean_on_original_graphs=pre[pid if not pid.endswith("-nc") else pid + " (K_avg k=0, primary)"][k]["mean"])
                                 for k in K.MEAS}
        pp = res["per_point"][pid]
        print(pid, {k: f"{v['observed']:.3f} [{v['ref_lo']:.3f},{v['ref_hi']:.3f}] z{v['z']:+.2f}" for k, v in pp.items()}, flush=True)
    zs = [res["per_point"][p]["persistence"]["z"] for p in ARMS]; Z = float(np.sum(zs) / np.sqrt(len(zs)))
    res["primary"] = dict(arms=ARMS, z=dict(zip(ARMS, zs)), stouffer_Z=Z, p_one_sided=float(stats.norm.sf(Z)), excess_repeats=bool(Z >= 1.645))
    res["flip_rate_stouffer"] = float(np.sum([res["per_point"][p]["flip_rate"]["z"] for p in ARMS]) / np.sqrt(len(ARMS)))
    pp = res["per_point"]
    res["repeat_or_move"] = dict(P1_abs_m_below=bool(pp["P1"]["abs_m"]["observed"] < pp["P1"]["abs_m"]["ref_lo"]), P1_abs_m_z=pp["P1"]["abs_m"]["z"],
                                 P3_persistence_above=bool(pp["P3"]["persistence"]["observed"] > pp["P3"]["persistence"]["ref_hi"]),
                                 P3_persistence_z=pp["P3"]["persistence"]["z"])
    res["per_point_primary_pass"] = {p: all(pp[p][m]["inside"] for m in prim[p]) for p in K.POINTS}
    json.dump(res, open(os.path.join(OUT, "reanalysis.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("primary", "flip_rate_stouffer", "repeat_or_move", "per_point_primary_pass")}, indent=1))


if __name__ == "__main__":
    main()
