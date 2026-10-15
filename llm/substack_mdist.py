"""Substack figure: distribution of the magnetisation m over the measured window, pooled over 50 seed pairs (run A).
No API calls. Output: figures/substack/m_distribution.png (+ m_distribution.json).

  python llm/substack_mdist.py

FIXED IN ADVANCE (committed before rendering):
- The same setting as llm/substack_visuals.py: the same network (N = 400, d = 4, rho = 0.25, seed `substack-graph`),
  the same kernels and temperatures, the same seed pairs j = 0..49 (start `substack-start|j`, dynamics
  `substack-dyn|j`), 1,000 sweeps.
- "Run A" = the model's table without error (for Qwen and Gemini these are exactly the twin runs' run A).
- Measured window: sweeps 501-1,000 (the second half; >= the confirmed equilibration length for Qwen, Gemini and
  GPT-4o-mini; equal to Gemma's run length of 1,000 only for the whole run, so Gemma's window is stated as such).
- Quantity: signed m = mean position (+1 = A, -1 = B) after every sweep in the window, pooled over 50 runs
  (25,000 values per panel); histogram with 40 bins on [-1, 1], density-normalised.
- Gemma: hot start (random 50/50, pair j) and cold start (all A, dynamics seed of pair j) overlaid.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import substack_visuals as V

WIN = (501, 1000)
BINS = np.linspace(-1, 1, 41)


def main():
    G = V.graph(); ip, nb, tie = G; res = {}
    for m in ("P1", "gemini", "selected", "gemma"):
        for T in V.temps(m):
            L = V.logit_table(m, T)
            for start in (("hot", "cold") if m == "gemma" else ("hot",)):
                ms = []
                for j in range(V.NPAIRS):
                    s0 = V.hot_start(j) if start == "hot" else np.ones(V.N, np.int64)
                    st = V.run_record(ip, nb, tie, L, s0, V.seed("substack-dyn", j), V.SWEEPS)
                    ms.append(st[WIN[0]:WIN[1] + 1].mean(1))
                ms = np.concatenate(ms); h, _ = np.histogram(ms, BINS, density=True)
                res[f"{m}|{T:.4f}|{start}"] = dict(model=m, T=T, start=start, hist=h.tolist(), mean_abs_m=float(np.abs(ms).mean()),
                                                    share_abs_m_gt_0_5=float((np.abs(ms) > 0.5).mean()), n=int(len(ms)))
    os.makedirs(V.OUT, exist_ok=True)
    json.dump(dict(window=WIN, bins=BINS.tolist(), panels=res), open(os.path.join(V.OUT, "m_distribution.json"), "w"), indent=1)
    plt = V.style()
    fig, axs = plt.subplots(4, 3, figsize=(10, 9.5), sharex=True)
    ctr = 0.5 * (BINS[1:] + BINS[:-1])
    for i, m in enumerate(("P1", "gemini", "selected", "gemma")):
        for k, T in enumerate(V.temps(m)):
            ax = axs[i, k]
            for start, col, ls in (("hot", "#333333", "-"), ("cold", "#999999", "--")):
                key = f"{m}|{T:.4f}|{start}"
                if key not in res:
                    continue
                h = np.array(res[key]["hist"]); ax.step(ctr, h, where="mid", color=col, lw=1.3, ls=ls)
                ax.fill_between(ctr, h, step="mid", color=col, alpha=0.08, lw=0)
                if m == "gemma" and k == 0:
                    ax.annotate(f"{start} start", (ctr[np.argmax(h)], h.max()), xytext=(8, -2 if start == "hot" else -14),
                                textcoords="offset points", fontsize=8, color=col)
            ax.axvspan(-1, 0, color=V.CB, alpha=0.06, lw=0); ax.axvspan(0, 1, color=V.CA, alpha=0.06, lw=0)
            ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlim(-1, 1)
            lab = ({0.8: "0.8·T_c", 1.0: "T_c", 1.25: "1.25·T_c"}[round(T / V.tc(m), 2)] + f" (T = {T:.2f})") if m in ("P1", "gemini") else f"T = {T:.0f}"
            ax.set_title(lab, loc="left", fontsize=8.5, color="#333")
            if k == 0:
                ax.set_ylabel(V.NAME[m], fontsize=9, color="#333")
            if i == 3:
                ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_xticklabels(["all B", "", "even split", "", "all A"])
    axs[3, 1].set_xlabel("m = share holding A minus share holding B")
    V.head(fig, "How lopsided does each population get?",
           "Distribution of m over the second half of each run (sweeps 501–1,000), pooled over 50 runs; two peaks = a population that picks a side",
           y=(0.95, 0.928))
    fig.subplots_adjust(top=0.9, hspace=0.45)
    V.foot(fig, V.CAPTION + " Qwen and Gemini at 0.8, 1 and 1.25 × their tipping point T_c; GPT-4o-mini and Gemma at T = 1, 2, 3 "
           "(GPT-4o-mini: no transition; Gemma: unresolved, slow relaxation near an ordering boundary). Gemma: random start (solid) and all-A start (dashed). Same runs as run A of the twin figure.", y=0.045)
    fig.savefig(os.path.join(V.OUT, "m_distribution.png"), dpi=150); plt.close(fig)
    for key, r in res.items():
        print(key, f"mean|m| {r['mean_abs_m']:.3f}", f"share|m|>0.5 {r['share_abs_m_gt_0_5']:.2f}")


if __name__ == "__main__":
    main()
