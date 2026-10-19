"""Combined effect size of the persistence excess, original Stage 3 run + option B replicate (llm/REPLAY_AUDIT_PLAN.md).

  python llm/persistence_effect_combined.py

Per arm and run: effect = observed persistence (mean of 5) - reference mean, SE = SD of the reference's mean-of-5
distribution; reference = K_avg simulated on that run's actual graphs and starts (original P1-P4: kernel-test K_avg sims;
original nc arms: the k = 0 K_avg table simulated now on the original graphs and starts, R = 400; replicate:
reanalysis.json). Fixed-effect inverse-variance pooling (raw and standardised), per run and over both runs; Cochran's Q
between the two runs. Arms P1-P4, P2-nc, P4-nc; P5 separately. Output: results/llm/stage3_rep1/persistence_effect_combined.json
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage3, kernel_test as K

ROOT = K.ROOT
ARMS = ["P1", "P2", "P3", "P4", "P2-nc", "P4-nc"]


def nc_table(pid, nccalls):
    rho = stage3.POINTS[pid][2]
    g = nccalls[(nccalls.point == pid) & (nccalls.leak <= stage3.LEAK_MAX)].groupby(["s0", "mapping", "order"]).pA.mean().reset_index()
    g["P"] = np.where(g.mapping == 0, g.pA, 1 - g.pA); T = np.full(K.NT, np.nan)
    for s0 in (1, -1):
        T[K.idx((0, 0, 0, 0), s0)] = g[g.s0 == s0].P.mean()
    return K.expand_nc(T, rho)


def pool(d, se):
    w = 1 / np.square(se); est = float(np.sum(w * d) / np.sum(w)); s = float(1 / np.sqrt(np.sum(w)))
    return dict(estimate=est, se=s, ci95=[est - 1.96 * s, est + 1.96 * s], z=est / s, p_two_sided=float(2 * stats.norm.sf(abs(est / s))))


def main():
    obs0 = pd.read_csv(os.path.join(ROOT, "results/llm/stage3/summary.csv")).set_index("point")
    rea = json.load(open(os.path.join(ROOT, "results/llm/stage3_rep1/reanalysis.json")))["per_point"]
    nccalls = pd.read_csv(os.path.join(ROOT, "results/llm/replicate/k_avg_nc_calls.csv"))
    rows = []
    for pid in ARMS + ["P5"]:
        if pid.endswith("-nc"):
            T = nc_table(pid, nccalls)
            sims = K.simulate(pid, lambda run, g: T, 400, "combined-orig-nc")    # original graphs (networkx 3.7) and starts
        else:
            sims = pd.read_csv(os.path.join(K.OUT, f"sims_{pid}_K_avg.csv"))
        rows.append(dict(run="original", arm=pid, observed=float(obs0.loc[pid, "persistence"]), ref_mean=float(sims.persistence.mean()),
                         se=float(sims.persistence.std(ddof=1))))
        r = rea[pid]["persistence"]
        rows.append(dict(run="replicate", arm=pid, observed=r["observed"], ref_mean=r["ref_mean"], se=r["ref_sd"]))
    D = pd.DataFrame(rows); D["d"] = D.observed - D.ref_mean; D["z"] = D.d / D.se
    out = {"per_arm": D.round(5).to_dict(orient="records")}
    M = D[D.arm.isin(ARMS)]
    for run in ("original", "replicate"):
        x = M[M.run == run]
        out[run] = dict(raw=pool(x.d.values, x.se.values),
                        standardised=dict(mean_z=float(x.z.mean()), se=1 / np.sqrt(len(x)), ci95=[float(x.z.mean() - 1.96 / np.sqrt(len(x))), float(x.z.mean() + 1.96 / np.sqrt(len(x)))],
                                          stouffer=float(x.z.sum() / np.sqrt(len(x)))))
    out["pooled"] = dict(raw=pool(M.d.values, M.se.values),
                         standardised=dict(mean_z=float(M.z.mean()), se=1 / np.sqrt(len(M)), ci95=[float(M.z.mean() - 1.96 / np.sqrt(len(M))), float(M.z.mean() + 1.96 / np.sqrt(len(M)))],
                                           stouffer=float(M.z.sum() / np.sqrt(len(M)))))
    e = np.array([out[r]["raw"]["estimate"] for r in ("original", "replicate")]); s = np.array([out[r]["raw"]["se"] for r in ("original", "replicate")])
    Q = float(np.sum((e - out["pooled"]["raw"]["estimate"]) ** 2 / s ** 2))
    out["heterogeneity_between_runs"] = dict(Q=Q, df=1, p=float(stats.chi2.sf(Q, 1)))
    json.dump(out, open(os.path.join(ROOT, "results/llm/stage3_rep1/persistence_effect_combined.json"), "w"), indent=1)
    print(D.round(4).to_string(index=False))
    print(json.dumps({k: out[k] for k in ("original", "replicate", "pooled", "heterogeneity_between_runs")}, indent=1))


if __name__ == "__main__":
    main()
