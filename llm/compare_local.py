"""Amendment 4, confirmatory item 4: local bf16 Gemma vs provider endpoints (PREREG Amendment 4).

  python llm/compare_local.py [--B 1000] [--out results/llm/amendment4] [--pairs ...] [--workers 4]

Pairs: local 31B vs Io Net 31B (gemma-31b); local 26B vs Parasail 26B (gemma-26b); local 26B vs CoreWeave 26B
(gemma-26b-cw). Arms: neutral_own, neutral_noown, political, workplace.
- Data preparation, design, fits and derived coefficients are analysis.py's own (prompts, design, fit_ce, fit_A3,
  derived); main battery only (perm 0), as in the confirmatory fits.
- Paired bootstrap: both endpoints are restricted to the cells they share after their own leak exclusions; each
  draw resamples those shared cells with replacement and refits BOTH endpoints on the same cells (B draws).
  Difference = provider − local. A coefficient "agrees" if the 95% percentile CI of the difference contains 0.
- Attenuation: provider estimates regressed on local estimates across all (arm, coefficient) entries except β and α,
  through the origin, weights 1/(se²_provider + se²_local) (se = bootstrap SD), slope CI from the same paired draws.
  Prediction κ = (1 + πσ²/8)^(-1/2), σ = the provider's pooled repeat SD from its Stage 1 fit.
- Verdict per pair: "jitter averages out" if ≥ 90% of coefficients agree AND the slope CI contains 1; else
  "jitter biases estimates". Confound note: quantisation, chat-template handling and serving stack also differ.
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = [("gemma-31b-local", "gemma-31b", "Io Net"), ("gemma-26b-local", "gemma-26b", "Parasail"),
         ("gemma-26b-local", "gemma-26b-cw", "CoreWeave")]
ARMS = ["neutral_own", "neutral_noown", "political", "workplace"]
EXCLUDE_FROM_SLOPE = ("beta", "alpha")


def prep(path, arm):
    d = A.load(path)
    P, st = A.prompts(d[d["arm"] == arm])
    return P[P["perm"] == 0].copy(), st


def fit_all(M):
    X, names = A.design(M, "A0")
    b = A.fit_ce(X, M["y"].values, M["w"].values.astype(float))
    est = A.derived(dict(zip(names, b)))
    est["alpha"] = float(A.fit_A3(M)[0])
    return {k: float(v) for k, v in est.items() if k not in ("w_r_over_w_a", "b_r_minus_b_a")}


def compare_arm(job):
    loc_path, prov_path, arm, B, seed = job
    L, _ = prep(loc_path, arm); Pv, _ = prep(prov_path, arm)
    shared = np.array(sorted(set(L["cell"]) & set(Pv["cell"])))
    L, Pv = L[L["cell"].isin(shared)].reset_index(drop=True), Pv[Pv["cell"].isin(shared)].reset_index(drop=True)
    memL = {c: np.flatnonzero(L["cell"].values == c) for c in shared}
    memP = {c: np.flatnonzero(Pv["cell"].values == c) for c in shared}
    estL, estP = fit_all(L), fit_all(Pv)
    names = list(estL)
    rng = np.random.default_rng(seed)
    dL, dP = {n: [] for n in names}, {n: [] for n in names}
    for _ in range(B):
        pick = shared[rng.integers(0, len(shared), len(shared))]           # SAME cells for both endpoints
        iL = np.concatenate([memL[c] for c in pick]); iP = np.concatenate([memP[c] for c in pick])
        fl, fp = fit_all(L.iloc[iL]), fit_all(Pv.iloc[iP])
        for n in names:
            dL[n].append(fl[n]); dP[n].append(fp[n])
    rows = []
    for n in names:
        a, b = np.array(dL[n]), np.array(dP[n]); diff = b - a
        rows.append(dict(arm=arm, coef=n, local=estL[n], provider=estP[n], diff=estP[n] - estL[n],
                         diff_lo=float(np.percentile(diff, 2.5)), diff_hi=float(np.percentile(diff, 97.5)),
                         se_local=float(a.std(ddof=1)), se_provider=float(b.std(ddof=1)),
                         agrees=bool(np.percentile(diff, 2.5) <= 0 <= np.percentile(diff, 97.5)),
                         n_shared_cells=len(shared), n_local=len(L), n_provider=len(Pv)))
    return rows, {n: (np.array(dL[n]), np.array(dP[n])) for n in names}


def slope(x, y, w):
    return float((w * x * y).sum() / (w * x * x).sum())


def kappa(sigma):
    return float((1 + np.pi * sigma ** 2 / 8) ** -0.5)


def provider_sigma(fit_json):
    """Pooled repeat SD (summarize.pooled_sd) from the provider's Stage 1 fit."""
    arms = json.load(open(fit_json))["arms"]
    n = np.array([a["prompts"] for a in arms]); sd = np.array([a["repeat_sd"] for a in arms])
    return float(np.sqrt((n * sd ** 2).sum() / n.sum()))


def _cached(job, cache):
    """compare_arm with a per-(pair, arm) cache: results and bootstrap draws are saved the moment they exist, so a
    restart resumes instead of recomputing (CLAUDE.md: record output as it is produced)."""
    loc_path, prov_path, arm, B, seed = job
    if cache:
        f = os.path.join(cache, f"{os.path.basename(loc_path)[:-4]}__{os.path.basename(prov_path)[:-4]}__{arm}__B{B}.npz")
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            return list(z["rows"]), {n: (z["L_" + n], z["P_" + n]) for n in z["names"]}
    rows, draws = compare_arm(job)
    if cache:
        os.makedirs(cache, exist_ok=True)
        np.savez(f + ".tmp.npz", rows=np.array(rows, dtype=object), names=np.array(list(draws)),
                 **{"L_" + n: v[0] for n, v in draws.items()}, **{"P_" + n: v[1] for n, v in draws.items()})
        os.replace(f + ".tmp.npz", f)
    return rows, draws


def compare_pair(loc_key, prov_key, label, B, stage1_dir, fits_dir, workers, seed=11, cache=None):
    jobs = [(os.path.join(stage1_dir, loc_key + ".csv"), os.path.join(stage1_dir, prov_key + ".csv"), arm, B, seed + i)
            for i, arm in enumerate(ARMS)]
    from functools import partial
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(partial(_cached, cache=cache), jobs))
    rows = [r for rr, _ in res for r in rr]
    T = pd.DataFrame(rows); T.insert(0, "pair", f"{prov_key} ({label}) vs {loc_key}")
    keep = ~T["coef"].isin(EXCLUDE_FROM_SLOPE)
    S = T[keep].reset_index(drop=True)
    w = 1.0 / (S["se_provider"] ** 2 + S["se_local"] ** 2)
    b_hat = slope(S["local"].values, S["provider"].values, w.values)
    draws = []
    for d in range(B):
        x = np.array([res[ARMS.index(r.arm)][1][r.coef][0][d] for r in S.itertuples()])
        y = np.array([res[ARMS.index(r.arm)][1][r.coef][1][d] for r in S.itertuples()])
        draws.append(slope(x, y, w.values))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    sig = provider_sigma(os.path.join(fits_dir, prov_key + ".json")) if os.path.exists(os.path.join(fits_dir, prov_key + ".json")) else None
    k = kappa(sig) if sig is not None else None
    frac = float(T["agrees"].mean())
    verdict = "jitter averages out" if (frac >= 0.9 and lo <= 1 <= hi) else "jitter biases estimates"
    summ = dict(pair=T["pair"].iloc[0], n_coefficients=int(len(T)), frac_agree=frac,
                slope=b_hat, slope_lo=float(lo), slope_hi=float(hi), slope_ci_contains_1=bool(lo <= 1 <= hi),
                provider_sigma=sig, kappa=k, slope_ci_contains_kappa=(bool(lo <= k <= hi) if k else None),
                n_slope_entries=int(len(S)), verdict=verdict)
    return T, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage1", default=os.path.join(ROOT, "results/llm/stage1/main"))
    ap.add_argument("--fits", default=os.path.join(ROOT, "results/llm/fits/main"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/llm/amendment4"))
    ap.add_argument("--pairs", nargs="*", help="override as loc:prov:label ...")
    a = ap.parse_args()
    pairs = [tuple(p.split(":")) for p in a.pairs] if a.pairs else PAIRS
    os.makedirs(a.out, exist_ok=True)
    tabs, summs = [], []
    for loc, prov, label in pairs:
        T, s = compare_pair(loc, prov, label, a.B, a.stage1, a.fits, a.workers, cache=os.path.join(a.out, "cache"))
        tabs.append(T); summs.append(s)
        T.to_csv(os.path.join(a.out, f"local_vs_provider_{prov}.csv"), index=False)             # per pair, at once
        json.dump(s, open(os.path.join(a.out, f"local_vs_provider_{prov}.json"), "w"), indent=1)
        print(json.dumps(s, indent=1), flush=True)
    pd.concat(tabs).to_csv(os.path.join(a.out, "local_vs_provider_coefficients.csv"), index=False)
    json.dump(summs, open(os.path.join(a.out, "local_vs_provider_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
