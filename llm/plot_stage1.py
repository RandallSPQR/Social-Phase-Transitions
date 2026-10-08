"""Stage 1 figures, Tufte style matching sim/plots.py (off-white, serif, direct labels, CIs and n shown).

  python ../llm/plot_stage1.py results/llm/stage1/pilot/llama-8b.csv results/llm/fits/pilot/llama-8b.json figures/llm_pilot_llama8b.png --pilot
"""
import argparse, json, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import expit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis

ROOT = analysis.__file__.rsplit("/llm/", 1)[0]
# Tufte rc + colour dict, copied verbatim from sim/plots.py so figures match fig1-fig4
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

ARM_LABEL = {"neutral_own": "neutral", "neutral_noown": "neutral, own position hidden",
             "political": "political", "workplace": "workplace"}
ARM_ORDER = ["neutral_own", "neutral_noown", "political", "workplace"]
PARAMS = [("b_a", "ally coupling  b_ally"), ("b_r", "rival coupling  b_rival"), ("gamma", "inertia  γ (own position)"),
          ("h", "letter-A field  h (neutral)"), ("h_C", "content field  h_C"), ("h_L", "letter-A field  h_L"),
          ("h_O", "listed-first field  h_O")]


def note(fig, t, y=-0.02):
    fig.text(0.06, y, t, fontsize=8.5, color="#999999", ha="left", va="top", family="sans-serif")


def figure(csv, fits_json, out, pilot=False, title=None):
    F = json.load(open(fits_json)); ep = F["endpoint"]
    arms = {a["arm"]: a for a in F["arms"]}
    d = analysis.load(csv)
    fig = plt.figure(figsize=(13, 9.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.55, wspace=0.18)
    xm = 1.0
    for arm in ARM_ORDER:
        if arm in arms:
            P, _ = analysis.prompts(d[d["arm"] == arm]); P = P[P["perm"] == 0]
            X, names = analysis.design(P, "A0")
            xm = max(xm, np.abs(X @ np.array([arms[arm]["coef"][n]["est"] for n in names])).max())
    xm = float(np.ceil(xm))
    # --- top row: observed P(choose P) against the additive model's linear predictor, one panel per arm
    for i, arm in enumerate(ARM_ORDER):
        ax = fig.add_subplot(gs[0, i])
        if arm not in arms:
            ax.axis("off"); continue
        P, st = analysis.prompts(d[d["arm"] == arm]); P = P[P["perm"] == 0]
        X, names = analysis.design(P, "A0")
        b = np.array([arms[arm]["coef"][n]["est"] for n in names])
        eta = X @ b
        k = P["k"].values
        ax.scatter(eta, P["y"], s=7, c=np.where(k >= 5, "#333333", "#aaaaaa"), lw=0, alpha=0.8)
        g = np.linspace(-xm, xm, 200); ax.plot(g, expit(g), color=C["hl"], lw=1.2)
        ax.set_xlim(-xm * 1.06, xm * 1.06); ax.set_ylim(-0.03, 1.03)
        ax.spines["bottom"].set_bounds(-xm, xm); ax.spines["left"].set_bounds(0, 1)
        ax.set_title(ARM_LABEL[arm], loc="left", fontsize=12, color=C["text"])
        r = arms[arm]
        ax.text(-xm, 0.97, f"n = {r['n_prompts']} prompts\n{r['n_cells']} cells", fontsize=8.5, color=C["text3"], va="top")
        if i == 0:
            ax.set_ylabel("P(choose P)")
            ax.text(0.25 * xm, 0.18, "additive logit\n(model A0)", color=C["hl"], fontsize=9.5)
            ax.text(-xm, 0.66, "dark: k = 5–6\nlight: k = 1–4", color=C["text3"], fontsize=8.5, style="italic")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("fitted field η", fontsize=10)
    # --- bottom: coefficients with 95% bootstrap CIs, arms offset vertically, labelled directly
    ax = fig.add_subplot(gs[1, :3])
    shades = {"neutral_own": "#111111", "neutral_noown": "#999999", "political": C["hl"], "workplace": "#4e79a7"}
    off = {"neutral_own": -0.27, "neutral_noown": -0.09, "political": 0.09, "workplace": 0.27}
    rows = [p for p in PARAMS if any(p[0] in arms[a]["coef"] for a in arms)]
    for j, (pn, plabel) in enumerate(rows):
        for arm in ARM_ORDER:
            if arm in arms and pn in arms[arm]["coef"]:
                c = arms[arm]["coef"][pn]; y = j + off[arm]
                ax.plot([c["lo"], c["hi"]], [y, y], color=shades[arm], lw=1.4)
                ax.plot(c["est"], y, "o", ms=3.8, color=shades[arm])
    ax.axvline(0, color=C["axis"], lw=0.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows], fontsize=10.5, color=C["text2"])
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.invert_yaxis(); ax.set_xlabel("coefficient (logit units per neighbour; fields in logit units)")
    lo = min(arms[a]["coef"][p]["lo"] for a in arms for p, _ in rows if p in arms[a]["coef"])
    hi = max(arms[a]["coef"][p]["hi"] for a in arms for p, _ in rows if p in arms[a]["coef"])
    ax.spines["bottom"].set_bounds(np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2)
    # key in the emptiest region (rows 2-3, right): order of the four marks within every row
    xk = 0.55 * np.ceil(hi * 2) / 2 + 0.45 * np.floor(lo * 2) / 2 + 0.25 * (hi - lo)
    ax.text(xk, 1.75, "within each row, top to bottom:", fontsize=9, color=C["text3"], style="italic")
    for i, arm in enumerate(a for a in ARM_ORDER if a in arms):
        ax.text(xk, 2.1 + 0.33 * i, ARM_LABEL[arm], color=shades[arm], fontsize=9.5, va="center")
    ax.set_title("Additive response function, per framing (dots: estimate; bars: 95% CI)", loc="left", fontsize=12, color=C["text"])
    # --- bottom right: derived quantities table, as text
    ax2 = fig.add_subplot(gs[1, 3]); ax2.axis("off")
    lines = []
    for arm in ARM_ORDER:
        if arm not in arms: continue
        r = arms[arm]; c = r["coef"]
        def f(n): return f"{c[n]['est']:.2f} [{c[n]['lo']:.2f}, {c[n]['hi']:.2f}]"
        h3 = r["H3"]
        lines += [(ARM_LABEL[arm], C["text"], 10.5),
                  (f"β {f('beta')}", C["text2"], 9.5), (f"b_rival − b_ally {f('b_r_minus_b_a')}", C["text2"], 9.5),
                  (f"α (k-scaling) {f('alpha')}", C["text2"], 9.5),
                  (f"pairwise p={h3['pairwise']['p']:.3f}, ΔCE {100*h3['pairwise']['cv_rel_improvement']:+.1f}%", C["text2"], 9.5),
                  (f"cubic p={h3['cubic']['p']:.3f}, ΔCE {100*h3['cubic']['cv_rel_improvement']:+.1f}%", C["text2"], 9.5),
                  ("", C["text2"], 5)]
    y = 1.02
    for t, col, fs in lines:
        ax2.text(0, y, t, color=col, fontsize=fs, transform=ax2.transAxes, va="top"); y -= 0.047 if fs > 6 else 0.02
    model = f"{ep['model']} @ {ep['tag']}"
    head = title or f"{model.split('/')[1]}: additive response function"
    fig.text(0.06, 0.985, head, fontsize=16, color=C["text"])
    fig.text(0.06, 0.955, "Top: each dot is one prompt; red curve is the fitted additive logit. Bottom: fitted couplings and fields with 95% cluster-bootstrap CIs.",
             fontsize=11, color=C["text2"])
    rs = [arms[a].get("repeat_sd") for a in arms]
    note(fig, f"Source: OpenRouter, {model}, first-token logprobs at T = 1, renormalised over {{A, B}}; two identical calls per prompt (repeat SD ≤ {max(x for x in rs if x is not None):.2f} nats). "
         f"Neighbour order randomised per prompt.\nCells = neighbour multiset × own position; CIs from 1,000 bootstrap resamples of cells. "
         f"{'Pilot: 25 of 209 multisets, instruction v2; exploratory, not confirmatory (PREREG Amendment 2).' if pilot else ''}", y=0.035)
    fig.savefig(out); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("fits"); ap.add_argument("out")
    ap.add_argument("--pilot", action="store_true"); ap.add_argument("--title", default=None); a = ap.parse_args()
    figure(a.csv, a.fits, a.out, a.pilot, a.title)
