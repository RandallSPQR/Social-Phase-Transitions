"""Stage 2 summaries.

  python llm/stage2_report.py

Inputs: results/llm/stage2/{stage3,fss,floor,jitter}.csv. Outputs: results/llm/stage2/summary_stage3.csv,
summary_fss.csv, summary_jitter.csv.

Per condition (population × framing × mode × graph × ρ × ε):
  - run-level 90% predictive interval (5th–95th percentile over runs; each run = one parameter draw, one graph,
    one dynamics seed) and the 90% interval of the MEAN OF 5 RUNS (Stage 3 design: 5 graph seeds), by resampling;
  - excess unsatisfied = unsatisfied − frustration floor (same graph kind, ρ, N);
  - regime label (thresholds fixed here, before looking at results, and applied mechanically):
        consensus         |m| ≥ 0.5
        frozen-stubborn   |m| < 0.5, persistence ≥ 0.8, flip rate ≤ 0.02, excess unsatisfied ≥ 0.10
        frozen-frustrated |m| < 0.5, persistence ≥ 0.8, flip rate ≤ 0.02, excess unsatisfied < 0.10
        camps (glassy)    |m| < 0.5, 0.4 ≤ persistence < 0.8, |q| ≥ 0.3
        churn/disorder    |m| < 0.5, persistence < 0.4
        mixed             otherwise
FSS (long runs): slope x of log⟨m²⟩ on log N over N = 100, 400, 1600 with a bootstrap CI over runs;
x ≈ −1 means m² ∝ 1/N (no order: finite-size fluctuations only), x ≈ 0 means order survives as N grows.
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "llm", "stage2")
COND = ["pop", "framing", "mode", "graph", "rho", "eps"]
M = ["abs_m", "m", "abs_q", "persistence", "flip_rate", "unsatisfied", "excess_unsat"]


def regime(r):
    if r.abs_m >= 0.5: return "consensus"
    if r.persistence >= 0.8 and r.flip_rate <= 0.02:
        return "frozen-stubborn" if r.excess_unsat >= 0.10 else "frozen-frustrated"
    if 0.4 <= r.persistence < 0.8 and r.abs_q >= 0.3: return "camps"
    if r.persistence < 0.4: return "churn"
    return "mixed"


def add_floor(df):
    fl = pd.read_csv(os.path.join(D, "floor.csv")).groupby(["graph", "rho", "N"]).unsat_floor.mean().rename("floor").reset_index()
    df = df.merge(fl, on=["graph", "rho", "N"], how="left")
    df["excess_unsat"] = df["unsatisfied"] - df["floor"]
    return df


def summarise(df, n_mean=5, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for key, g in df.groupby(COND + (["N"] if "N" in df and df.N.nunique() > 1 else []), dropna=False):
        row = dict(zip(COND + (["N"] if len(key) > len(COND) else []), key)); row["runs"] = len(g)
        for m in M:
            x = g[m].values
            row[m] = float(np.median(x)); row[m + "_lo"], row[m + "_hi"] = np.percentile(x, [5, 95])
            means = x[rng.integers(0, len(x), (B, n_mean))].mean(1)
            row[m + "_mean5_lo"], row[m + "_mean5_hi"] = np.percentile(means, [5, 95])
        row["regime"] = regime(pd.Series({m: row[m] for m in M}))
        out.append(row)
    return pd.DataFrame(out)


def fss(df, B=1000, seed=1):
    rng = np.random.default_rng(seed); out = []
    for key, g in df.groupby(COND):
        Ns = sorted(g.N.unique())
        if len(Ns) < 3: continue
        def slope(gg):
            y = np.log([max(gg[gg.N == n].m2.mean(), 1e-12) for n in Ns]); return np.polyfit(np.log(Ns), y, 1)[0]
        bs = []
        groups = {n: g[g.N == n] for n in Ns}
        for _ in range(B):
            gg = pd.concat([groups[n].sample(len(groups[n]), replace=True, random_state=int(rng.integers(2**31))) for n in Ns])
            bs.append(slope(gg))
        row = dict(zip(COND, key)); row["x"] = slope(g); row["x_lo"], row["x_hi"] = np.percentile(bs, [2.5, 97.5])
        for n in Ns:
            row[f"m2_N{n}"] = g[g.N == n].m2.mean(); row[f"abs_m_N{n}"] = g[g.N == n].abs_m.mean()
            row[f"pers_N{n}"] = g[g.N == n].persistence.mean()
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    for name in ("stage3", "jitter"):
        p = os.path.join(D, f"{name}.csv")
        if os.path.exists(p):
            df = add_floor(pd.read_csv(p))
            if name == "jitter":
                df["mode"] = np.where(df.explicit_jitter, "explicit", "marginal")
            summarise(df).to_csv(os.path.join(D, f"summary_{name}.csv"), index=False); print("wrote", name)
    p = os.path.join(D, "fss.csv")
    if os.path.exists(p):
        df = add_floor(pd.read_csv(p))
        fss(df).to_csv(os.path.join(D, "summary_fss.csv"), index=False)
        summarise(df).to_csv(os.path.join(D, "summary_fss_bysize.csv"), index=False); print("wrote fss")
