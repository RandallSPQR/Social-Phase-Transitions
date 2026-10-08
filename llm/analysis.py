"""Stage 1 analysis (PREREG §4–6).

  python ../llm/analysis.py results/llm/stage1/pilot/llama-8b.csv  [--B 1000] [--out results/llm/fits/pilot]

Per arm (neutral_own, neutral_noown, political, workplace): exclusions, repeat noise, A0/A1/A2/A3 fits by
soft-label cross-entropy, cluster bootstrap over cells, H3 cluster score tests with wild score bootstrap (primary, Amendment 1)
and logit-scale wild-bootstrap F tests (secondary),
grouped-CV held-out cross-entropy, minimum-effect criteria. Confirmatory fits use the main battery (perm 0);
the order subset (perm 1, 2) is used only for permutation noise and order effects.
"""
import argparse, json, math, os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

CLIP = 6.0           # logit clip for logit-scale operations (PREREG §4, §6)
LEAK_MAX = 0.05
NOISE_GATE = 0.30


# ---------------------------------------------------------------- data
def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def load(path):
    d = pd.read_csv(path, dtype={"counts": str, "seq": str})
    d["counts"] = d["counts"].str.zfill(4)
    return d


def prompts(d):
    """Collapse replicate calls to one row per prompt; apply leak exclusion. Returns (prompt table, stats)."""
    key = ["arm", "framing", "counts", "s0", "mapping", "order", "perm", "seq"]
    d = d.copy()
    sampled = "answer" in d.columns and d["answer"].notna().any() or ("answer" in d.columns and d["pA"].isna().all())
    st = {"calls": int(len(d)), "calls_missing_lp": int(d["pA"].isna().sum()) if not sampled else 0}
    if sampled:
        d["valid"] = d["answer"].notna()
        g = d.groupby(key, dropna=False)
        P = g.agg(n=("valid", "size"), nv=("valid", "sum"), nA=("pA", "sum")).reset_index()
        P["leak"] = 1 - P["nv"] / P["n"]
        P["excluded"] = P["leak"] > LEAK_MAX
        P["pA"] = P["nA"] / P["nv"].clip(lower=1)
        P["w"] = P["nv"]
        st["repeat_sd"] = None
    else:
        d["ylog"] = np.clip(logit(d["pA"].astype(float)), -CLIP, CLIP)
        g = d.groupby(key, dropna=False)
        P = g.agg(pA=("pA", "mean"), leak_max=("leak", "max"), nrep=("pA", "count"),
                  yl_sd=("ylog", lambda x: abs(x.iloc[0] - x.iloc[1]) / math.sqrt(2) if len(x) >= 2 else np.nan),
                  censored=("censored", "max")).reset_index()
        P["excluded"] = (P["leak_max"] > LEAK_MAX) | P["pA"].isna()
        P["w"] = 1.0
        st["repeat_sd"] = float(np.sqrt(np.nanmean(P.loc[~P["excluded"], "yl_sd"] ** 2)))
        st["censored_prompts"] = int(P["censored"].fillna(0).sum())
    st["prompts"] = int(len(P)); st["excluded_prompts"] = int(P["excluded"].sum())
    st["excluded_frac"] = float(P["excluded"].mean())
    P = P[~P["excluded"]].copy()
    # outcome: P(choose content P)
    P["y"] = np.where(P["mapping"] == 0, P["pA"], 1 - P["pA"]).astype(float)
    c = np.array([[int(ch) for ch in s] for s in P["counts"]])
    P["naP"], P["naQ"], P["nrP"], P["nrQ"] = c.T
    P["ka"] = P["naP"] + P["naQ"]; P["kr"] = P["nrP"] + P["nrQ"]; P["k"] = P["ka"] + P["kr"]
    P["Ma"] = P["naP"] - P["naQ"]; P["Mr"] = P["nrP"] - P["nrQ"]
    P["L"] = np.where(P["mapping"] == 0, 1, -1)
    P["O"] = np.where(P["mapping"] == P["order"], 1, -1)          # P's letter listed first
    P["cell"] = P["counts"] + "|" + P["s0"].astype(str)
    # order effects (exploratory): signed pull of first- and last-listed contact toward P
    def pull(tok):
        s = 1 if tok[1] == "P" else -1
        return s if tok[0] == "a" else -s
    P["F1"] = [pull(s[:2]) for s in P["seq"]]
    P["Flast"] = [pull(s[-2:]) for s in P["seq"]]
    return P, st


# ---------------------------------------------------------------- designs
def e2(M, k):
    return (M ** 2 - k) / 2


def e3(M, k):
    return (M ** 3 - 3 * k * M + 2 * M) / 6


def design(P, model="A0"):
    neutral = (P["framing"].iloc[0] == "neutral")
    own = (P["s0"] != 0).any()
    cols = {"b_a": P["Ma"], "b_r": -P["Mr"]}
    if own:
        cols["gamma"] = P["s0"]
    cols["h" if neutral else "h_C"] = np.ones(len(P))
    if not neutral:
        cols["h_L"] = P["L"]
    cols["h_O"] = P["O"]
    if model in ("A1",):
        cols.update(Q_aa=e2(P["Ma"], P["ka"]), Q_rr=e2(P["Mr"], P["kr"]), Q_ar=P["Ma"] * P["Mr"])
        if own:
            cols.update(s0Ma=P["s0"] * P["Ma"], s0Mr=P["s0"] * P["Mr"])
    if model in ("A2",):
        cols.update(T_aaa=e3(P["Ma"], P["ka"]), T_rrr=e3(P["Mr"], P["kr"]),
                    T_aar=e2(P["Ma"], P["ka"]) * P["Mr"], T_arr=P["Ma"] * e2(P["Mr"], P["kr"]))
    if model == "Aord":
        cols.update(first=P["F1"], last=P["Flast"])
    X = np.column_stack([np.asarray(v, float) for v in cols.values()])
    return X, list(cols)


BLOCKS = {"A1": "pairwise", "A2": "cubic"}


# ---------------------------------------------------------------- fitting
def fit_ce(X, y, w, x0=None):
    """Soft-label (fractional) logistic regression by cross-entropy; w = weights (n for binomial)."""
    def f(b):
        eta = X @ b
        ll = w * (y * eta - np.logaddexp(0, eta))
        g = X.T @ (w * (expit(eta) - y))
        return -ll.sum(), g
    r = minimize(f, np.zeros(X.shape[1]) if x0 is None else x0, jac=True, method="L-BFGS-B",
                 options={"maxiter": 2000, "gtol": 1e-9})
    return r.x


def fit_A3(P, x0=None):
    """logit y = k^-α (b_a Ma − b_r Mr) + Z θ. Returns (α, coefficient dict)."""
    X0, names = design(P, "A0")
    Z = X0[:, 2:]; Ma, Mr = P["Ma"].values.astype(float), P["Mr"].values.astype(float)
    lk = np.log(P["k"].values.astype(float)); y = P["y"].values; w = P["w"].values.astype(float)
    def f(t):
        a, ba, br, th = t[0], t[1], t[2], t[3:]
        s = np.exp(-a * lk); core = ba * Ma - br * Mr
        eta = s * core + Z @ th
        r = w * (expit(eta) - y)
        ll = (w * (y * eta - np.logaddexp(0, eta))).sum()
        g = np.concatenate([[(r * (-lk * s * core)).sum(), (r * s * Ma).sum(), (r * (-s * Mr)).sum()], Z.T @ r])
        return -ll, g
    t0 = np.zeros(3 + Z.shape[1]) if x0 is None else x0
    r = minimize(f, t0, jac=True, method="L-BFGS-B", bounds=[(-3, 4)] + [(None, None)] * (len(t0) - 1),
                 options={"maxiter": 3000, "gtol": 1e-9})
    return r.x


def derived(c):
    out = dict(c)
    if "b_a" in c:
        out["beta"] = (c["b_a"] + c["b_r"]) / 2
        out["w_r_over_w_a"] = c["b_r"] / c["b_a"] if c["b_a"] != 0 else np.nan
        out["b_r_minus_b_a"] = c["b_r"] - c["b_a"]
    return out


def ce(y, p, w):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())


# ---------------------------------------------------------------- inference
def cluster_index(P):
    cells = P["cell"].values
    u, inv = np.unique(cells, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(u))]
    return members


def bootstrap(P, B, rng):
    """Cluster bootstrap over cells for A0 coefficients (+derived) and A3 α. Returns dict name -> array(B)."""
    members = cluster_index(P)
    X, names = design(P, "A0")
    y, w = P["y"].values, P["w"].values.astype(float)
    b0 = fit_ce(X, y, w); a30 = fit_A3(P)
    out = {n: [] for n in names}; out.update(beta=[], w_r_over_w_a=[], b_r_minus_b_a=[], alpha=[])
    for _ in range(B):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        b = fit_ce(X[idx], y[idx], w[idx], b0)
        dv = derived(dict(zip(names, b)))
        for n in out:
            if n in dv:
                out[n].append(dv[n])
        out["alpha"].append(fit_A3(P.iloc[idx], a30)[0])
    return {k: np.array(v) for k, v in out.items()}


def boot_summary(est, draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    B = len(draws)
    p = max(2 * min((draws <= 0).mean(), (draws >= 0).mean()), 1 / B)
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "p": float(min(p, 1.0)), "se": float(draws.std(ddof=1))}


def ols_rss(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return float(r @ r), b


def wild_F(P, model, nboot, rng):
    """LR (F) test of an interaction block on the logit scale, null imposed, wild cluster bootstrap (Rademacher by cell)."""
    X0, _ = design(P, "A0"); X1, n1 = design(P, model)
    y = np.clip(logit(P["y"].values), -CLIP, CLIP)
    n, p0, p1 = len(y), X0.shape[1], X1.shape[1]
    q = p1 - p0
    rss0, b0 = ols_rss(X0, y); rss1, _ = ols_rss(X1, y)
    F = ((rss0 - rss1) / q) / (rss1 / (n - p1))
    fit0, e0 = X0 @ b0, y - X0 @ b0
    _, inv = np.unique(P["cell"].values, return_inverse=True)
    G = inv.max() + 1
    # projection helpers for speed
    H0 = X0 @ np.linalg.pinv(X0); H1 = X1 @ np.linalg.pinv(X1)
    Fs = np.empty(nboot)
    for i in range(nboot):
        v = rng.choice([-1.0, 1.0], G)[inv]
        ys = fit0 + e0 * v
        r0 = ys - H0 @ ys; r1 = ys - H1 @ ys
        a, b = r0 @ r0, r1 @ r1
        Fs[i] = ((a - b) / q) / (b / (n - p1))
    p = (1 + (Fs >= F).sum()) / (1 + nboot)
    return {"F": float(F), "q": q, "n": n, "p": float(p), "resid_sd_A0": float(math.sqrt(rss0 / (n - p0))),
            "resid_sd_A1": float(math.sqrt(rss1 / (n - p1)))}


def score_test(P, model, nboot, rng):
    """Cluster-robust score test of an interaction block on the cross-entropy (quasi-binomial) fit, with
    wild score bootstrap (Kline & Santos 2012; Rademacher weights by cell). Asymptotically equivalent to the
    LR test; needs no logit clipping, so saturated responses do not masquerade as curvature (Amendment 1)."""
    X0, _ = design(P, "A0"); X1, _ = design(P, model)
    y, w = P["y"].values, P["w"].values.astype(float)
    p0 = expit(X0 @ fit_ce(X0, y, w))
    XI = X1[:, X0.shape[1]:]
    Wt = w * p0 * (1 - p0)
    # efficient score: residualise the block regressors on the nuisance ones, information-weighted
    A = np.linalg.solve(X0.T @ (Wt[:, None] * X0), X0.T @ (Wt[:, None] * XI))
    Xt = XI - X0 @ A
    u = (w * (y - p0))[:, None] * Xt
    _, inv = np.unique(P["cell"].values, return_inverse=True)
    G = inv.max() + 1
    sg = np.zeros((G, Xt.shape[1])); np.add.at(sg, inv, u)
    V = sg.T @ sg
    Vi = np.linalg.pinv(V)
    S = sg.sum(0); T = float(S @ Vi @ S)
    v = rng.choice([-1.0, 1.0], (nboot, G))
    Sb = v @ sg
    Tb = np.einsum("bi,ij,bj->b", Sb, Vi, Sb)
    from scipy.stats import chi2
    return {"T": T, "q": XI.shape[1], "G": int(G), "p": float((1 + (Tb >= T).sum()) / (1 + nboot)),
            "p_chi2": float(chi2.sf(T, XI.shape[1]))}


def binom_LR(P, model):
    from scipy.stats import chi2
    X0, _ = design(P, "A0"); X1, _ = design(P, model)
    y, w = P["y"].values, P["w"].values.astype(float)
    b0 = fit_ce(X0, y, w); b1 = fit_ce(X1, y, w)
    def ll(X, b):
        eta = X @ b
        return float((w * (y * eta - np.logaddexp(0, eta))).sum())
    LR = 2 * (ll(X1, b1) - ll(X0, b0)); q = X1.shape[1] - X0.shape[1]
    return {"LR": LR, "q": q, "p": float(chi2.sf(LR, q))}


def folds_by_multiset(P, K=10, seed=12345):
    ms = np.array(sorted(P["counts"].unique()))
    rng = np.random.default_rng(seed); rng.shuffle(ms)
    fold = {m: i % K for i, m in enumerate(ms)}
    return P["counts"].map(fold).values


def cv_ce(P, model, K=10):
    X, _ = design(P, model); y, w = P["y"].values, P["w"].values.astype(float)
    f = folds_by_multiset(P, K); pred = np.empty(len(y))
    for k in range(K):
        tr, te = f != k, f == k
        if te.sum() == 0: continue
        b = fit_ce(X[tr], y[tr], w[tr]); pred[te] = expit(X[te] @ b)
    return ce(y, pred, w)


def min_effect(P, model):
    X0, _ = design(P, "A0"); X1, _ = design(P, model)
    y, w = P["y"].values, P["w"].values.astype(float)
    p0 = expit(X0 @ fit_ce(X0, y, w)); p1 = expit(X1 @ fit_ce(X1, y, w))
    dP = pd.Series(np.abs(p1 - p0)).groupby(P["cell"].values).mean()
    ce0, ce1 = cv_ce(P, "A0"), cv_ce(P, model)
    rel = (ce0 - ce1) / ce0
    frac = float((dP >= 0.05).mean())
    return {"cv_ce_A0": ce0, "cv_ce": ce1, "cv_rel_improvement": float(rel), "frac_cells_dP_ge_0.05": frac,
            "max_cell_dP": float(dP.max()), "passes_min_effect": bool(rel >= 0.02 and frac >= 0.10)}


# ---------------------------------------------------------------- driver
def analyse_arm(P, st, B=1000, nboot=999, seed=0):
    rng = np.random.default_rng(seed)
    sampled = P["w"].max() > 1
    main = P[P["perm"] == 0].copy()
    res = {"arm": main["arm"].iloc[0], "framing": main["framing"].iloc[0], "n_prompts": int(len(main)),
           "n_cells": int(main["cell"].nunique()), **st}
    X, names = design(main, "A0"); y, w = main["y"].values, main["w"].values.astype(float)
    b = fit_ce(X, y, w); est = derived(dict(zip(names, b)))
    a3 = fit_A3(main); est["alpha"] = a3[0]
    res["A0_in_sample_ce"] = ce(y, expit(X @ b), w)
    # entropy floor of the soft labels: the best any model could do in-sample
    res["label_entropy"] = ce(y, y, w)
    draws = bootstrap(main, B, rng)
    res["coef"] = {k: boot_summary(est[k], draws[k]) for k in draws}
    res["A3"] = {"alpha": float(a3[0]), "b_a_k1": float(a3[1]), "b_r_k1": float(a3[2])}
    res["H3"] = {}
    for m in ("A1", "A2"):
        t = score_test(main, m, nboot, rng)                       # primary (Amendment 1)
        t["secondary"] = binom_LR(main, m) if sampled else wild_F(main, m, nboot, rng)
        _, n1 = design(main, m); b1 = fit_ce(*design(main, m)[:1], y, w)
        t["coef"] = dict(zip(n1, map(float, b1)))
        t.update(min_effect(main, m))
        res["H3"][BLOCKS[m]] = t
    # order effects (exploratory), on all perms
    Xo, no = design(P, "Aord"); bo = fit_ce(Xo, P["y"].values, P["w"].values.astype(float))
    res["order_effects"] = {k: float(v) for k, v in zip(no, bo) if k in ("first", "last")}
    # permutation noise from the order subset (logit SD across perms of the same cell+variant)
    if (P["perm"] > 0).any() and not sampled:
        g = P.assign(yl=np.clip(logit(P["y"]), -CLIP, CLIP)).groupby(["counts", "s0", "mapping", "order"])["yl"]
        sd = g.std(ddof=1).dropna()
        res["perm_sd"] = float(np.sqrt((sd ** 2).mean())) if len(sd) else None
    res["endpoint_passes_noise_gate"] = (st.get("repeat_sd") is None) or (st["repeat_sd"] <= NOISE_GATE)
    return res


def analyse_file(path, B=1000, nboot=999, out_dir=None):
    d = load(path)
    results = []
    for arm, da in d.groupby("arm"):
        P, st = prompts(da)
        if len(P) < 10:
            continue
        results.append(analyse_arm(P, st, B, nboot))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(out_dir, name + ".json"), "w") as f:
            json.dump({"source": path, "endpoint": d[["key", "model", "tag"]].iloc[0].to_dict(), "arms": results},
                      f, indent=1, default=float)
    return results


def holm(pvals):
    """Holm step-down adjusted p-values for a list (None kept as None)."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    order = sorted(idx, key=lambda i: pvals[i]); m = len(order); adj = [None] * len(pvals); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * pvals[i])); adj[i] = run
    return adj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--B", type=int, default=1000); ap.add_argument("--nboot", type=int, default=999)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    for r in analyse_file(a.csv, a.B, a.nboot, a.out):
        c = r["coef"]
        print(f"\n== {r['arm']}  prompts={r['n_prompts']} cells={r['n_cells']} excluded={r['excluded_prompts']} "
              f"repeat_sd={r.get('repeat_sd')} perm_sd={r.get('perm_sd')}")
        for k, v in c.items():
            print(f"  {k:15s} {v['est']:8.3f}  [{v['lo']:7.3f}, {v['hi']:7.3f}]  p={v['p']:.4f}")
        for blk, t in r["H3"].items():
            print(f"  H3 {blk:9s} T={t['T']:.2f} (q={t['q']}) p={t['p']:.4f} [secondary p={t['secondary']['p']:.4f}]  cvCE {t['cv_ce_A0']:.4f}->{t['cv_ce']:.4f} "
                  f"rel={t['cv_rel_improvement']:+.3f} fracdP={t['frac_cells_dP_ge_0.05']:.2f} pass={t['passes_min_effect']}")
        print(f"  order effects {r['order_effects']}  A0 CE={r['A0_in_sample_ce']:.4f} floor={r['label_entropy']:.4f}")
