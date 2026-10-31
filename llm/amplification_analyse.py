"""Amplification curve: analysis (plan: llm/AMPLIFICATION_PLAN.md §4-7). Written before stage B results exist.

  python llm/amplification_analyse.py

Reads results/llm/amplification/{locate.json, grid.json, perturb/<proto>/*.csv.gz}. Writes amplification.csv (every
point, N, T, shape, size, measure, protocol: Δ, ε, A with 95% bootstrap CI, A in SD units), growth.csv (G(T)),
verdict.json, placement.csv and fig1-fig4 (PNG).
"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import amplification_curve as A

OUT = A.OUT
B = 1000
XS = ["abs_m", "persistence"]
SYN = ["stay", "majority", "noise"]
REAL = ["fixed", "a3"]


def shape_of(s):
    n, v = s.split(":")
    return ("noise" if n.startswith("noise") else n), float(v)


def unit_stats(df, sd):
    """One (proto, point, N, T) unit -> rows of A with bootstrap draws (paired by run)."""
    base = df[df.setting == "base"].set_index("run").sort_index()
    R = len(base); idx = np.random.default_rng(sd).integers(R, size=(B, R))
    rows, draws = [], {}
    groups = {}
    for s, g in df[df.setting != "base"].groupby("setting"):
        groups.setdefault(shape_of(s), []).append((s, g.set_index("run").sort_index()))
    for (shape, size), members in groups.items():
        for X in XS:
            a_pt, a_bt, d_pt, eps_l = [], [], [], []
            for s, g in members:
                d = (g[X] - base[X]).values; eps = float(g.eps.iloc[0]); eps_l.append(eps)
                d_pt.append(d.mean()); a_pt.append(abs(d.mean()) / eps if eps > 0 else np.nan)
                a_bt.append(np.abs(d[idx].mean(1)) / eps if eps > 0 else np.full(B, np.nan))
            a, ab = float(np.mean(a_pt)), np.mean(a_bt, axis=0)
            sdb = float(base[X].std(ddof=1))
            rows.append(dict(shape=shape, size=size, measure=X, delta=float(np.mean(d_pt)) if shape != "noise" else float(np.mean(np.abs(d_pt))),
                             eps=float(np.mean(eps_l)), A=a, A_lo=float(np.nanpercentile(ab, 2.5)), A_hi=float(np.nanpercentile(ab, 97.5)),
                             A_sd_units=a / sdb if sdb > 0 else np.nan, base_mean=float(base[X].mean()), base_sd=sdb, R=R))
            draws[(shape, size, X)] = ab
    return rows, draws


def load():
    rows, draws = [], {}
    for f in sorted(glob.glob(os.path.join(OUT, "perturb", "*", "*.csv.gz"))):
        df = pd.read_csv(f)
        proto, pid, N, T = df.proto.iloc[0], df.point.iloc[0], int(df.N.iloc[0]), float(df["T"].iloc[0])
        r, d = unit_stats(df, A.seed("amp-boot", proto, pid, N, T))
        meta = dict(proto=proto, point=pid, N=N, T=T, filled_share=float(df.filled_share.iloc[0]))
        rows += [dict(meta, **x) for x in r]
        for k, v in d.items():
            draws[(proto, pid, N, T) + k] = v
    return pd.DataFrame(rows), draws


def ci(x):
    x = np.asarray(x)[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [np.nan, np.nan]


def growth(D, draws):
    rows = []
    for (proto, pid, T, shape, size, X), g in D.groupby(["proto", "point", "T", "shape", "size", "measure"]):
        a = g.set_index("N").A
        if 100 in a.index and 1600 in a.index:
            bt = draws[(proto, pid, 1600, T, shape, size, X)] / draws[(proto, pid, 100, T, shape, size, X)]
            lo, hi = ci(bt)
            rows.append(dict(proto=proto, point=pid, T=T, shape=shape, size=size, measure=X, G=a[1600] / a[100], G_lo=lo, G_hi=hi))
    return pd.DataFrame(rows)


def verdict(D, Gd, draws, Tc):
    grid = sorted(D["T"].unique())
    Ts = min(grid, key=lambda t: abs(np.log(t / Tc)))
    out = dict(Tc_P1=Tc, T_star=Ts, grid=grid)
    for X in XS:
        h = {}
        for shape in SYN:
            key = lambda pid, N: draws[("eq", pid, N, Ts, shape, 0.1, X)]
            gP = key("P1", 1600) / key("P1", 100)
            G = float(Gd.query("proto=='eq' and point=='P1' and T==@Ts and shape==@shape and size==0.1 and measure==@X").G.iloc[0])
            r = dict(G=G, G_ci=ci(gP), H1=bool(ci(gP)[0] > 1))
            for c in ("P2", "P4"):
                gc = key(c, 1600) / key(c, 100); rat = gP / gc
                Gc = float(Gd.query("proto=='eq' and point==@c and T==@Ts and shape==@shape and size==0.1 and measure==@X").G.iloc[0])
                r[f"G_{c}"] = Gc; r[f"G_{c}_ci"] = ci(gc); r[f"ratio_vs_{c}"] = G / Gc; r[f"ratio_vs_{c}_ci"] = ci(rat)
            r["H2"] = bool(r["H1"] and all(r[f"ratio_vs_{c}_ci"][0] > 1 for c in ("P2", "P4")))
            # expectation (i): peak of A(T, 1600) within ±1 grid step of T*
            a16 = D.query("proto=='eq' and point=='P1' and N==1600 and shape==@shape and size==0.1 and measure==@X").set_index("T").A.sort_index()
            r["peak_T"] = float(a16.idxmax()); r["peak_within_1_step"] = bool(abs(grid.index(a16.idxmax()) - grid.index(Ts)) <= 1)
            h[shape] = r
        n1 = [s for s in SYN if h[s]["H1"]]
        sup_H1 = len(n1) >= 2; sup_H2 = sup_H1 and all(h[s]["H2"] for s in n1)
        out[X] = dict(per_shape=h, H1=sup_H1, H2=sup_H2, verdict="supported" if (sup_H1 and sup_H2) else
                      ("refuted: N-growth near T_c absent" if not sup_H1 else "refuted: N-growth matches the controls"))
    # expectations (ii), (iii), (v)
    g = Gd[(Gd.proto == "eq") & (Gd.measure == "abs_m") & (Gd["size"] == 0.1) & Gd["shape"].isin(SYN)]
    far = g[(g.point == "P1") & ((g["T"] <= Tc / 1.6) | (g["T"] >= 1.6 * Tc))]
    out["exp_ii_far_P1_G_below_2"] = dict(holds=bool((far.G < 2).all()), max_G=float(far.G.max()) if len(far) else None, n=int(len(far)))
    ctrl = g[g.point.isin(["P2", "P4"])]
    out["exp_iii_controls_G_below_2"] = dict(holds=bool((ctrl.G < 2).all()), max_G=float(ctrl.G.max()),
                                             where=ctrl.loc[ctrl.G.idxmax(), ["point", "T", "shape"]].to_dict())
    rl = Gd[(Gd.proto == "eq") & (Gd.measure == "abs_m") & (Gd["size"] == 1.0) & Gd["shape"].isin(REAL) & (Gd["T"] == Ts)]
    out["exp_v_real_shapes_at_Tstar"] = rl[["point", "shape", "G", "G_lo", "G_hi"]].to_dict(orient="records")
    lin = D[(D.proto == "eq") & (D.measure == "abs_m") & D["shape"].isin(SYN) & D["size"].isin([0.03, 0.1])]
    lin = lin[(lin["T"] <= Tc / 1.6) | (lin["T"] >= 1.6 * Tc)].pivot_table(index=["point", "N", "T", "shape"], columns="size", values="A")
    out["exp_vi_linear_response_median_ratio_A0.1_over_A0.03"] = float((lin[0.1] / lin[0.03]).median())
    return out


# ---------------------------------------------------------------- figures (Tufte preset)
def style():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["ET Book", "Palatino", "Palatino Linotype", "Georgia", "DejaVu Serif"],
        "font.size": 11, "figure.facecolor": "#fffff8", "axes.facecolor": "#fffff8", "savefig.facecolor": "#fffff8",
        "figure.dpi": 150, "axes.edgecolor": "#cccccc", "axes.linewidth": 0.5, "axes.labelcolor": "#666666",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "xtick.color": "#999999",
        "ytick.color": "#999999", "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.major.size": 3, "ytick.major.size": 3, "lines.linewidth": 1.4,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.2})
    return plt


NCOL = {100: "#bbbbbb", 400: "#888888", 1600: "#333333"}
ACC = "#e41a1c"
LABEL = {"P1": "P1  Qwen-122B (test)", "P2": "P2  Qwen-9B (control)", "P4": "P4  Nemo-12B (control)"}
SHL = {"stay": "stay-bias", "majority": "majority-shift", "noise": "random noise", "fixed": "K_fixed − K_avg", "a3": "A3 − K_avg"}


def head(fig, title, sub, note, y=(1.07, 1.02)):
    fig.text(0.01, y[0], title, fontsize=16, color="#111", ha="left", va="bottom")
    fig.text(0.01, y[1], sub, fontsize=10.5, color="#666", ha="left", va="bottom")
    fig.text(0.01, -0.03, note, fontsize=8, color="#999", ha="left", va="top", family="sans-serif")


def tticks(ax):
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5])); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))


def marks(ax, Tc, ylo):
    ax.axvline(Tc, color=ACC, lw=0.6, alpha=0.5); ax.axvline(1.0, color="#999", lw=0.6, ls=":"); tticks(ax)


def fig1(loc):
    plt = style(); T = loc["fine"]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, p in zip(axs, A.PTS):
        ax.set_ylim(-0.05, 0.7)
        for N in A.NS:
            u = loc[p]["binder"][str(N)]
            ax.plot(T, u, color=NCOL[N], marker="o", ms=2.5)
            ax.annotate(f"N={N}", (T[-1], u[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=NCOL[N], va="center")
        ax.set_xscale("log"); tticks(ax); ax.set_title(LABEL[p], loc="left", fontsize=10, color="#333")
        if loc[p]["Tc"]:
            ax.axvline(loc[p]["Tc"], color=ACC, lw=0.6); ax.annotate(f"T_c ≈ {loc[p]['Tc']:.2f}", (loc[p]["Tc"], 0.62), xytext=(4, 0), textcoords="offset points", fontsize=8, color=ACC, style="italic")
        ax.axvline(1.0, color="#999", lw=0.6, ls=":"); ax.set_xlabel("temperature T (logits ÷ T)")
    axs[0].set_ylabel("Binder cumulant U")
    head(fig, "Binder cumulants of the three kernels across temperature",
         "Crossing of the N = 400 and 1,600 curves locates T_c (order → disorder edge); dotted line = Stage 3 temperature (T = 1)",
         f"K_avg tables with per-prompt logits divided by T; random-regular d = 4; random start, 1,000 warm-up + 1,000 measured sweeps; R = 100 runs per point (plan amendment 1). Source: results/llm/amplification/locate2.json")
    fig.savefig(os.path.join(OUT, "fig1_binder.png")); plt.close(fig)


def fig2(D, Tc, X="abs_m"):
    plt = style(); shapes = [("stay", 0.1), ("majority", 0.1), ("noise", 0.1), ("fixed", 1.0), ("a3", 1.0)]
    fig, axs = plt.subplots(len(shapes), 3, figsize=(10, 9.5), sharex=True, sharey=True)
    for i, (sh, sz) in enumerate(shapes):
        for j, p in enumerate(A.PTS):
            ax = axs[i, j]
            for N in A.NS:
                g = D.query("proto=='eq' and point==@p and N==@N and shape==@sh and size==@sz and measure==@X").sort_values("T")
                if not len(g):
                    continue
                c = (ACC if N == 1600 else NCOL[N]) if p == "P1" else NCOL[N]
                ax.plot(g["T"], g.A, color=c); ax.fill_between(g["T"], g.A_lo, g.A_hi, color=c, alpha=0.12, lw=0)
                if i == 0 and j == 0:
                    k = g.A.values.argmax()
                    ax.annotate(f"N = {N:,}", (g["T"].iloc[k], g.A.iloc[k]), xytext=(6, {100: -22, 400: -10, 1600: 4}[N]), textcoords="offset points",
                                fontsize=7.5, color=c, va="center")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 5); marks(ax, Tc, None)
            if i == 0:
                ax.set_title(LABEL[p], loc="left", fontsize=10, color="#333")
            if j == 0:
                ax.set_ylabel(f"{SHL[sh]}\n" + (f"δ = {sz} nats" if sh in SYN else "λ = 1"), fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel("temperature T")
    head(fig, f"Amplification of per-step error into {'|m|' if X == 'abs_m' else 'persistence'}, by temperature and size",
         "A = |Δ collective| / per-step error (nats); red vertical = P1's T_c, dotted = Stage 3 (T = 1); bands = 95% bootstrap CI",
         "Equilibrium protocol (500 + 500 sweeps, plan amendment 1), R = 400 paired runs per setting (common random numbers), 5 graphs per N. Source: results/llm/amplification/amplification.csv", y=(0.955, 0.935))
    fig.subplots_adjust(top=0.91)
    fig.savefig(os.path.join(OUT, f"fig2_amplification_{X}.png")); plt.close(fig)


def fig3(Gd, Tc, X="abs_m"):
    plt = style(); shapes = [("stay", 0.1), ("majority", 0.1), ("noise", 0.1), ("fixed", 1.0), ("a3", 1.0)]
    fig, axs = plt.subplots(1, len(shapes), figsize=(12, 3.4), sharey=True)
    for ax, (sh, sz) in zip(axs, shapes):
        for p in A.PTS:
            g = Gd.query("proto=='eq' and point==@p and shape==@sh and size==@sz and measure==@X").sort_values("T")
            c = ACC if p == "P1" else ("#666" if p == "P2" else "#aaa")
            ax.plot(g["T"], g.G, color=c); ax.fill_between(g["T"], g.G_lo, g.G_hi, color=c, alpha=0.12, lw=0)
            ax.annotate(p, (g["T"].iloc[-1], g.G.iloc[-1]), xytext=(3, 0), textcoords="offset points", fontsize=8, color=c, va="center")
        ax.axhline(1, color="#ccc", lw=0.6); ax.axhline(2, color="#ccc", lw=0.6, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.03, 100); marks(ax, Tc, None)
        ax.set_title(SHL[sh], loc="left", fontsize=10, color="#333"); ax.set_xlabel("T")
    axs[0].set_ylabel("G = A(N=1,600) / A(N=100)")
    head(fig, "Growth of amplification with system size, by temperature",
         "G > 1: amplification grows with N. Solid grey = 1, dashed = 2 (the 'bounded' expectation); red vertical = P1's T_c; y clipped to 0.03–100",
         "|m|, equilibrium protocol; δ = 0.1 nats (synthetic) or λ = 1 (real shapes); bands = 95% bootstrap CI. Source: results/llm/amplification/growth.csv")
    fig.savefig(os.path.join(OUT, f"fig3_growth_{X}.png")); plt.close(fig)


def fig4(D, X="abs_m"):
    plt = style()
    fig, axs = plt.subplots(2, 3, figsize=(10, 6.4), sharex=True, sharey=True)
    for i, proto in enumerate(["s3", "eq"]):
        for j, p in enumerate(A.PTS):
            ax = axs[i, j]
            for N, ls in ((100, "-"), (1600, "--")):
                for sh in SYN:
                    g = D.query("proto==@proto and point==@p and N==@N and T==1.0 and shape==@sh and measure==@X").sort_values("size")
                    ax.plot(g.eps, g.delta.abs(), color="#999", lw=0.9, ls=ls, alpha=0.8 if N == 100 else 0.5)
                for sh, mk in (("fixed", "o"), ("a3", "s")):
                    g = D.query("proto==@proto and point==@p and N==@N and T==1.0 and shape==@sh and size==1.0 and measure==@X")
                    if len(g):
                        ax.plot(g.eps, g.delta.abs(), mk, color=ACC if N == 100 else "#333", ms=5, mfc="none" if N == 1600 else ACC)
                        if N == 100:
                            ax.annotate(SHL[sh], (g.eps.iloc[0], abs(g.delta.iloc[0])), xytext=(5, -2), textcoords="offset points", fontsize=7, color="#333")
            ax.set_xscale("log"); ax.set_yscale("log")
            if i == 0:
                ax.set_title(LABEL[p], loc="left", fontsize=10, color="#333")
            if j == 0:
                ax.set_ylabel(("Stage 3 protocol (10 + 10)" if proto == "s3" else "equilibrium (500 + 500)") + f"\n|Δ {'|m|' if X == 'abs_m' else 'persistence'}|", fontsize=9)
            if i == 1:
                ax.set_xlabel("per-step error ε (nats)")
    head(fig, "Where the real evaluation errors fall at the Stage 3 temperature (T = 1)",
         "Grey lines: synthetic shapes (stay, majority, noise) at N = 100 (solid) and 1,600 (dashed); filled = real errors at N = 100, open = N = 1,600",
         "Real error = K_fixed (fixed-order battery) or A3 (trait summary) in place of K_avg, λ = 1. R = 400 paired runs. Source: results/llm/amplification/placement.csv")
    fig.savefig(os.path.join(OUT, f"fig4_placement_{X}.png")); plt.close(fig)


def main():
    loc = json.load(open(os.path.join(OUT, "locate2.json"))); Tc = loc["P1"]["Tc"]   # amendment 1: stage A2
    D, draws = load(); D.to_csv(os.path.join(OUT, "amplification.csv"), index=False)
    Gd = growth(D, draws); Gd.to_csv(os.path.join(OUT, "growth.csv"), index=False)
    v = verdict(D, Gd, draws, Tc); v["Tc_P1_ci"] = loc["P1"]["Tc_ci"]; v["T1_over_Tc"] = 1.0 / Tc
    v["equilibration"] = {k: loc[k] for k in loc if k.startswith("hotcold_gap") or k == "stageB"}
    v["frozen_low_T"] = {p: loc[p]["frozen_low_T"] for p in A.PTS}
    pl = D[(D["T"] == 1.0) & D["shape"].isin(REAL) & (D["size"] == 1.0)][["proto", "point", "N", "shape", "measure", "eps", "delta", "A", "A_lo", "A_hi", "A_sd_units", "filled_share"]]
    pl.to_csv(os.path.join(OUT, "placement.csv"), index=False)
    json.dump(v, open(os.path.join(OUT, "verdict.json"), "w"), indent=1, default=float)
    fig1(loc); fig2(D, Tc, "abs_m"); fig2(D, Tc, "persistence"); fig3(Gd, Tc, "abs_m"); fig3(Gd, Tc, "persistence"); fig4(D, "abs_m"); fig4(D, "persistence")
    print(json.dumps({k: v[k] for k in ("Tc_P1", "Tc_P1_ci", "T_star", "T1_over_Tc")}, default=float))
    for X in XS:
        print(X, v[X]["verdict"], {s: {k: v[X]["per_shape"][s][k] for k in ("G", "G_ci", "H1", "H2", "ratio_vs_P2", "ratio_vs_P4", "peak_T")} for s in SYN})
    print(pl.to_string())


if __name__ == "__main__":
    main()
