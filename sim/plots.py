import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import summarize, binder, scaled
from vbsim import analytic_TF, analytic_TSG

C = {"bg": "#fffff8", "text": "#111111", "text2": "#666666", "text3": "#999999",
     "axis": "#cccccc", "series": "#666666", "hl": "#e41a1c"}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["ET Book", "Palatino", "Palatino Linotype", "Georgia", "DejaVu Serif"],
    "font.size": 12, "figure.facecolor": C["bg"], "axes.facecolor": C["bg"],
    "savefig.facecolor": C["bg"], "figure.dpi": 150,
    "axes.edgecolor": C["axis"], "axes.linewidth": 0.6, "axes.labelcolor": C["text2"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "xtick.color": C["text3"], "ytick.color": C["text3"], "xtick.labelsize": 10,
    "ytick.labelsize": 10, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3, "lines.linewidth": 1.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
})
OUT = _os.path.join(ROOT, "figures")
import os; os.makedirs(OUT, exist_ok=True)


def title(fig, t, s, y=0.92):
    fig.text(0.06, y + 0.06, t, fontsize=17, color=C["text"], ha="left")
    fig.text(0.06, y + 0.015, s, fontsize=12, color=C["text2"], ha="left")


def note(fig, t, y=-0.02):
    fig.text(0.06, y, t, fontsize=8.5, color="#999999", ha="left", va="top", family="sans-serif")


def best(sm):
    """Crossing estimate from the largest size pair, scaled-chi estimator (primary)."""
    N1, N2, b, s = sm["res"][-1]
    return s[0], s[1], s[2], b, (N1, N2)


# ---------- Figure 1: phase diagram (rho, T) at c = 4 ----------
def fig_phase():
    c = 4
    fig, ax = plt.subplots(figsize=(9, 6))
    rho_F = np.linspace(0, 0.25, 200)
    TF = np.array([analytic_TF(c, r) for r in rho_F])
    TSG = analytic_TSG(c)
    ax.plot(rho_F, TF, color=C["series"], lw=1.6)
    ax.plot([0.25, 0.5], [TSG, TSG], color=C["series"], lw=1.6)
    ax.plot([0.25], [TSG], "o", ms=4, color=C["series"])
    # sim points
    pts = []
    for tag, rho in [("f_r00", 0.0), ("f_r10", 0.1), ("f_r20", 0.2), ("sg_c4", 0.5)]:
        sm = summarize(tag, "sg" if tag.startswith("sg") else "f")
        if sm is None or not sm["res"]:
            continue
        e, lo, hi, b, pair = best(sm)
        pts.append((rho, e, lo, hi, sm["pred"], pair))
        ax.errorbar([rho], [e], yerr=[[e - lo], [hi - e]], fmt="o", ms=6, color=C["hl"],
                    ecolor=C["hl"], elinewidth=1.2, capsize=0, zorder=5)
    # region labels
    ax.text(0.28, 3.55, "Disorder", fontsize=15, color=C["text"])
    ax.text(0.28, 3.33, "paramagnet: states keep fluctuating", fontsize=10.5,
            style="italic", color=C["text2"])
    ax.text(0.015, 1.25, "Consensus", fontsize=15, color=C["text"])
    ax.text(0.015, 1.05, "ferromagnet: one shared state", fontsize=10.5, style="italic", color=C["text2"])
    ax.text(0.335, 1.25, "Frozen camps", fontsize=15, color=C["text"])
    ax.text(0.335, 1.05, "spin glass: many incompatible\nlocally stable states", fontsize=10.5,
            style="italic", color=C["text2"], va="top")
    ax.annotate("c(1−2ρ) tanh(1/T) = 1", xy=(0.12, analytic_TF(c, 0.12)), xytext=(0.135, 3.55),
                fontsize=10.5, color=C["text2"], style="italic",
                arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.6))
    ax.text(0.5, TSG + 0.08, "c tanh²(1/T) = 1", fontsize=10.5, color=C["text2"], style="italic", ha="right")
    ax.annotate("multicritical point", xy=(0.25, TSG), xytext=(0.205, 1.6), fontsize=10,
                color=C["text3"], style="italic",
                arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.6))
    ax.text(0.255, 0.62, "consensus / glass boundary: not computed here", fontsize=9,
            color=C["text3"], style="italic")
    ax.set_xlim(-0.01, 0.51); ax.set_ylim(0.5, 4.3)
    ax.set_xlabel("share of negative (antagonistic) ties, ρ")
    ax.set_ylabel("noise, T")
    ax.spines["bottom"].set_bounds(0, 0.5); ax.spines["left"].set_bounds(0.5, 4.3)
    title(fig, "The simulation lands on the predicted phase boundaries",
          "Gray: cavity-method predictions, mean degree c = 4. Red: simulated transition, 95% bootstrap CI.")
    lines = "; ".join(f"ρ={p[0]:.1f}: sim {p[1]:.2f} vs theory {p[4]:.2f}" for p in pts)
    note(fig, "Viana–Bray ±J model on Erdős–Rényi graphs, mean degree 4. Heat-bath dynamics with parallel tempering, "
         "two replicas.\nTransition = crossing of N²ᐟ³⟨q²⟩ (glass) or N¹ᐟ²⟨m²⟩ (consensus)\n"
         f"for the two largest sizes; CI from 1,000 bootstrap resamples over graphs.\n{lines}.")
    fig.savefig(f"{OUT}/fig1_phase_diagram_c4.png")
    plt.close(fig)
    return pts


# ---------- Figure 2: glass line vs connectivity at rho = 0.5 ----------
def fig_connectivity():
    fig, ax = plt.subplots(figsize=(9, 6))
    cs = np.concatenate([1 + np.logspace(-12, -1, 200), np.linspace(1.11, 7, 300)])
    ax.plot(cs, [analytic_TSG(c) for c in cs], color=C["series"], lw=1.6)
    pts = []
    for tag, c in [("sg_c3", 3), ("sg_c4", 4), ("sg_c6", 6)]:
        sm = summarize(tag, "sg")
        if sm is None or not sm["res"]:
            continue
        e, lo, hi, b, pair = best(sm)
        pts.append((c, e, lo, hi, sm["pred"]))
        ax.errorbar([c], [e], yerr=[[e - lo], [hi - e]], fmt="o", ms=6, color=C["hl"],
                    ecolor=C["hl"], elinewidth=1.2, capsize=0, zorder=5)
    ax.plot([1], [0], "o", ms=6, mfc=C["bg"], mec=C["text"], zorder=6)
    ax.annotate("percolation threshold, c = 1:\nwith zero noise, the glass appears\nexactly when a giant component does",
                xy=(1, 0), xytext=(1.5, 0.35), fontsize=10.5, style="italic", color=C["text2"],
                arrowprops=dict(arrowstyle="-", color=C["axis"], lw=0.6))
    ax.text(5.2, 1.05, "Frozen camps", fontsize=14, color=C["text"], ha="center")
    ax.text(2.6, 2.5, "Disorder", fontsize=14, color=C["text"], ha="center")
    ax.text(6.95, analytic_TSG(6.95) + 0.1, "c tanh²(1/T) = 1", fontsize=10.5, style="italic",
            color=C["text2"], ha="right")
    ax.set_xlim(0.8, 7.1); ax.set_ylim(-0.08, 3.0)
    ax.spines["bottom"].set_bounds(1, 7); ax.spines["left"].set_bounds(0, 3)
    ax.set_xlabel("connectivity: mean ties per agent, c")
    ax.set_ylabel("noise, T")
    title(fig, "The glass line runs down to the percolation threshold",
          "Half the ties antagonistic (ρ = 0.5). Gray: cavity prediction. Red: simulated transition, 95% bootstrap CI.")
    lines = "; ".join(f"c={p[0]}: sim {p[1]:.2f} vs theory {p[4]:.2f}" for p in pts)
    note(fig, "Effective bond probability tanh²(1/T) → 1 as T → 0, so the line ends at the percolation threshold c = 1. "
         "Same model and estimator as Figure 1.\n" + lines + ".")
    fig.savefig(f"{OUT}/fig2_glass_vs_connectivity.png")
    plt.close(fig)
    return pts


# ---------- Figure 3: finite-size crossings (diagnostic) ----------
def fig_crossings():
    sm = summarize("sg_c4", "sg")
    Ns = sorted(sm["runs"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    fig.subplots_adjust(wspace=0.25)
    shades = np.linspace(0.75, 0.1, len(Ns))
    for j, (ax, kind) in enumerate(zip(axes, ["scaled", "binder"])):
        for N, sh in zip(Ns, shades):
            r = sm["runs"][N]
            T = r["Ts"]
            y = scaled(r["m"], 0, N, 2 / 3) if kind == "scaled" else binder(r["m"], 0, 1)
            col = str(sh)
            ax.plot(T, y, color=col, lw=1.4)
            ax.text(T[0] - 0.03, y[0], f"N={N:,}", fontsize=9.5, color=col, va="center", ha="right")
        ax.axvline(sm["pred"], color=C["hl"], lw=0.8, ls=(0, (3, 3)))
        ax.text(sm["pred"] + 0.03, 8 if kind == "scaled" else 0.62,
                f"theory {sm['pred']:.2f}", color=C["hl"], fontsize=10, style="italic", va="top")
        ax.set_xlabel("noise, T")
        if kind == "scaled": ax.set_yscale("log")
        ax.set_title(r"$N^{2/3}\langle q^2\rangle$, scaled glass susceptibility (log scale)" if kind == "scaled" else "Binder cumulant of overlap",
                     fontsize=12, color=C["text"], loc="left")
        ax.set_xlim(1.0, 2.65)
        ax.spines["bottom"].set_bounds(T[0], T[-1])
    fig.text(0.06, 1.08, "Curves for different system sizes cross at the predicted transition", fontsize=17, color=C["text"])
    fig.text(0.06, 1.02, "c = 4, ρ = 0.5. Crossing of size curves locates the transition in the infinite-size limit.",
             fontsize=12, color=C["text2"])
    note(fig, "Each curve: disorder average over 160–400 graphs, measured in the second half of 2^12–2^13 sweeps. "
         "Binder cumulant has larger finite-size drift in diluted mean-field glasses; scaled susceptibility is the primary estimator.")
    fig.savefig(f"{OUT}/fig3_crossings_c4.png")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which: print(fig_phase())
    if "2" in which: print(fig_connectivity())
    if "3" in which: fig_crossings()
