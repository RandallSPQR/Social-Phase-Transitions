"""Stage 2 figures (Tufte style, matching sim/plots.py).

  python llm/plot_stage2.py

figures/llm_stage2_predictions.png  – the Stage 3 candidate points: fitted surrogate vs fields-only vs Ising reading
figures/llm_stage2_freezing.png     – persistence vs excess unsatisfied ties: stubbornness- vs frustration-freezing
figures/llm_stage2_fss.png          – finite-size scaling of <m²> (if summary_fss.csv exists)
"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_stage1 import C, note

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "llm", "stage2"); FIG = os.path.join(ROOT, "figures")
NAME = {"llama-70b": "Llama-3.3 70B", "qwen-9b": "Qwen3.5 9B", "qwen-122b": "Qwen3.5 122B-A10B", "nemo-12b": "Mistral Nemo 12B",
        "gemma-26b": "Gemma-4 26B-A4B †", "gemma-31b": "Gemma-4 31B †", "mistral-large": "Mistral Large 4 †",
        "gpt6-luna": "GPT-6-luna (s)", "haiku": "Claude Haiku 5.5 (s)", "gemini-lite": "Gemini 3.1 Flash-Lite (s)"}
POINTS = [("P1", "qwen-122b", "neutral", 0.25, "abs_m", "|m|"), ("P2", "qwen-9b", "political", 0.5, "m", "m (toward P)"),
          ("P3", "llama-70b", "political", 0.25, "abs_m", "|m|"), ("P4", "nemo-12b", "workplace", 0.25, "m", "m (toward P)")]
MODES = [("fitted", "fitted surrogate", C["hl"], 0.0), ("fields_only", "fields + inertia only", "#999999", -0.22),
         ("ising", "naive Ising reading", "#333333", 0.22)]


def fig_predictions():
    s = pd.read_csv(os.path.join(D, "summary_stage3.csv"))
    fig, axes = plt.subplots(len(POINTS), 3, figsize=(14, 2.0 * len(POINTS) + 1.6), sharex="col")
    fig.subplots_adjust(hspace=0.55, wspace=0.12, left=0.16)
    cols = [("order", "order parameter (mean of 5 graphs)", (-0.1, 1.0)), ("persistence", "persistence", (0, 1.0)),
            ("excess_unsat", "excess unsatisfied ties", (0, 0.55))]
    for i, (pid, pop, fr, rho, om, olab) in enumerate(POINTS):
        q = s[(s["pop"] == pop) & (s.framing == fr) & (s.rho == rho) & (s.graph == "rrg") & (s.eps == 0)].set_index("mode")
        for j, (cname, ctitle, lim) in enumerate(cols):
            ax = axes[i, j]; meas = om if cname == "order" else cname
            for mode, lab, col, off in MODES:
                if mode not in q.index: continue
                r = q.loc[mode]
                ax.plot([r[meas + "_mean5_lo"], r[meas + "_mean5_hi"]], [off, off], color=col, lw=2.2, solid_capstyle="butt")
                ax.plot(r[meas], off, "o", ms=4.5, color=col)
                if j == 0:
                    ax.text(-0.02, 0.5 + off / 0.9, lab, color=col, fontsize=9, va="center", ha="right", transform=ax.transAxes)
            ax.set_xlim(*lim); ax.set_ylim(-0.45, 0.45); ax.set_yticks([])
            ax.spines["left"].set_visible(False); ax.spines["bottom"].set_bounds(max(lim[0], 0), lim[1])
            if j == 0:
                ax.set_title(f"{pid}  {NAME[pop]}, {fr}, ρ = {rho}   ({olab})", loc="left", fontsize=11, color=C["text"])
            if i == len(POINTS) - 1:
                ax.set_xlabel(ctitle, fontsize=10)
    fig.text(0.02, 1.0, "Stage 2 predictions for four Stage 3 points: each fitted surrogate separates from both controls on at least one measure",
             fontsize=15, color=C["text"])
    fig.text(0.02, 0.968, "N = 100 agents, random-regular graph (4 contacts each), 10 + 10 sweeps from random starts, two replicas; "
             "dot: median run; bar: 90% interval of the mean of 5 graphs", fontsize=10.5, color=C["text2"])
    note(fig, "Source: Stage 2 surrogate simulations (llm/stage2.py), 40 runs per condition; each run draws one parameter vector from the endpoint's "
              "cluster-bootstrap fit, one graph and one dynamics seed.\nFitted surrogate: A3 logit with k^-α scaling (+ first-contact voter step where it won held-out), "
              "inertia γ, content field h_C, letter/order fields randomised per call. Fields-only: same γ and fields, no coupling. "
              "Ising reading: Stage 1 additive β only.", y=-0.01)
    fig.savefig(os.path.join(FIG, "llm_stage2_predictions.png")); plt.close(fig)


def fig_freezing():
    s = pd.read_csv(os.path.join(D, "summary_stage3.csv"))
    f = s[(s["mode"] == "fitted") & (s.graph == "rrg") & (s.eps == 0) & (~s["pop"].str.contains(r"\+"))]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    cols = {"neutral": "#111111", "political": C["hl"], "workplace": "#4e79a7"}
    for fr, g in f.groupby("framing"):
        ax.scatter(g.excess_unsat, g.persistence, s=18 + 60 * g.abs_m, color=cols[fr], alpha=0.75, lw=0)
    ax.axhline(0.8, color=C["axis"], lw=0.6, ls="--"); ax.axvline(0.10, color=C["axis"], lw=0.6, ls="--")
    ax.text(0.53, 0.985, "frozen by stubbornness\n(persistent, many unsatisfied ties)", fontsize=9, color=C["text3"], ha="right", va="top", style="italic")
    ax.text(0.005, 0.985, "frozen near a\nlow-frustration state", fontsize=9, color=C["text3"], va="top", style="italic")
    ax.text(0.53, 0.02, "churn", fontsize=9, color=C["text3"], ha="right", style="italic")
    for fr, (x, y) in {"neutral": (0.43, 0.30), "political": (0.36, 0.72), "workplace": (0.02, 0.62)}.items():
        ax.text(x, y, fr, color=cols[fr], fontsize=11)
    ax.set_xlim(-0.02, 0.55); ax.set_ylim(0, 1.0); ax.spines["bottom"].set_bounds(0, 0.55); ax.spines["left"].set_bounds(0, 1)
    ax.set_xlabel("excess unsatisfied ties (above the frustration floor of the same graph and ρ)"); ax.set_ylabel("persistence")
    fig.text(0.06, 1.0, "Political framing freezes agents in place; the frozen states are not low-frustration states",
             fontsize=14.5, color=C["text"])
    fig.text(0.06, 0.965, "Each dot: one endpoint × ρ (0, 0.25, 0.5), fitted surrogate, N = 100, random-regular d = 4; dot size ∝ |m|",
             fontsize=10.5, color=C["text2"])
    note(fig, "Source: Stage 2 surrogate simulations, median of 40 runs per condition. Frustration floor: β = 6 quench on the same graph family "
              "(0 at ρ = 0; ≈ 0.15 at ρ ≥ 0.25). † gate-failed endpoints included with jitter-marginal fits.", y=-0.01)
    fig.savefig(os.path.join(FIG, "llm_stage2_freezing.png")); plt.close(fig)


def fig_fss():
    p = os.path.join(D, "summary_fss.csv")
    if not os.path.exists(p): return
    s = pd.read_csv(p); s = s[(s.graph == "rrg") & (s.eps == 0)]
    pops = [k for k in NAME if k in set(s["pop"])]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.38)
    Ns = np.array([100, 400, 1600])
    for (i, fr), (j, rho) in [((i, fr), (j, rho)) for i, fr in enumerate(("neutral", "political")) for j, rho in enumerate((0.0, 0.25, 0.5))]:
        ax = axes[i, j]
        ax.plot(Ns, 1 / Ns * 1.6, color=C["axis"], lw=0.8, ls=":")   # m² ∝ 1/N reference (no order)
        labels = []
        for pop in pops:
            r = s[(s["pop"] == pop) & (s.framing == fr) & (s.rho == rho) & (s["mode"] == "fitted")]
            if r.empty: continue
            r = r.iloc[0]; y = [r[f"m2_N{n}"] for n in Ns]
            ordered = r.x_lo > -0.5
            ax.plot(Ns, y, marker="o", ms=3, lw=1.3 if ordered else 0.8, color=C["hl"] if ordered else "#999999", alpha=1 if ordered else 0.6)
            if ordered:
                labels.append([np.log10(max(y[-1], 1e-4)), NAME[pop]])
        labels.sort(key=lambda t: -t[0]); last = None          # push labels apart (≥ 0.13 decades)
        for t in labels:
            if last is not None and last - t[0] < 0.13: t[0] = last - 0.13
            last = t[0]; ax.text(Ns[-1] * 1.1, 10 ** t[0], t[1], fontsize=7.5, color=C["hl"], va="center")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"{fr}, ρ = {rho}", loc="left", fontsize=11, color=C["text"])
        if i == 1: ax.set_xlabel("N (agents)")
        if j == 0: ax.set_ylabel("⟨m²⟩")
        if i == 0 and j == 0: ax.text(110, 1.6 / 110 * 0.55, "⟨m²⟩ ∝ 1/N (no order)", fontsize=8.5, color=C["text3"], rotation=-22)
    fig.text(0.02, 1.0, "Finite-size scaling: spontaneous order persists with N for some neutral-framing surrogates; political alignment is field-driven",
             fontsize=14, color=C["text"])
    fig.text(0.02, 0.965, "Red: slope of log⟨m²⟩ on log N has a 95% CI above −0.5 (decays slower than 1/N); grey: consistent with ⟨m²⟩ ∝ 1/N (no order). "
             "Random-regular d = 4, 200 + 200 sweeps.", fontsize=10, color=C["text2"])
    note(fig, "Source: Stage 2 surrogate simulations, 16 runs per (endpoint, framing, ρ, N); fitted surrogates only. Neutral framing has no net external field "
              "(letter field randomised per call), so persistent ⟨m²⟩ there is spontaneous;\npolitical framing has a content field h_C, so persistent ⟨m²⟩ there "
              "includes field-driven alignment. Fixed run length (400 sweeps): slow coarsening can look like order. (s) = sampled spot check; † = gate-failed endpoint.", y=-0.01)
    fig.savefig(os.path.join(FIG, "llm_stage2_fss.png")); plt.close(fig)


if __name__ == "__main__":
    fig_predictions(); fig_freezing(); fig_fss(); print("ok")
