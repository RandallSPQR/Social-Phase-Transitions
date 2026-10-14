"""Stop point 2 figure: Binder cumulants per model (coarse sweep, hot starts; fine band where a T_c was located).
Output: results/llm/confirm/fig_binder_step3.png"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import amplification_analyse as AA
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.join(ROOT, "results/llm/confirm/sim")
M = [("P1", "P1  Qwen-122B (fresh)"), ("gemini", "Gemini 3.1 Flash-Lite"), ("selected", "GPT-4o-mini (selected)"), ("gemma", "Gemma-4 31B (local)"), ("luna", "GPT-6-luna (exploratory)")]
plt = AA.style()
fig, axs = plt.subplots(1, len(M), figsize=(14, 3.4), sharey=True)
for ax, (m, lab) in zip(axs, M):
    f = os.path.join(S, f"locate_{m}.json")
    if not os.path.exists(f):
        ax.set_axis_off(); continue
    r = json.load(open(f))
    for N in (100, 400, 1600):
        ax.plot(r["coarse"], r["binder_coarse"][str(N)], color=AA.NCOL[N], lw=1.0, alpha=0.6)
        if r.get("fine") and r.get("binder_fine"):
            ax.plot(r["fine"], r["binder_fine"][str(N)], color=AA.NCOL[N], lw=1.6)
        ax.annotate(f"N={N:,}", (r["coarse"][-1], r["binder_coarse"][str(N)][-1]), xytext=(3, {100: 8, 400: 0, 1600: -8}[N]), textcoords="offset points", fontsize=7, color=AA.NCOL[N], va="center")
    for T in r.get("frozen", []):
        ax.axvspan(T / 1.08, T * 1.08, color="#dddddd", alpha=0.35, lw=0)
    if r.get("Tc"):
        ax.axvline(r["Tc"], color=AA.ACC, lw=0.7); ax.annotate(f"T_c = {r['Tc']:.2f}", (r["Tc"], 0.6), xytext=(4, 0), textcoords="offset points", fontsize=8, color=AA.ACC, style="italic")
    else:
        ax.annotate("no transition\nin swept range", (0.25, 0.58), fontsize=8, color="#666", style="italic")
    ax.axvline(1.0, color="#999", lw=0.6, ls=":"); ax.set_xscale("log"); AA.tticks(ax); ax.set_ylim(-0.3, 0.7)
    ax.set_title(lab, loc="left", fontsize=9.5, color="#333"); ax.set_xlabel("T (logits ÷ T)")
axs[0].set_ylabel("Binder cumulant U")
AA.head(fig, "Where each model's kernel orders, by decision-noise temperature",
        "Crossing of the N = 400 and 1,600 curves locates T_c; grey bands = frozen (hot and cold starts disagree); dotted = T = 1",
        "Fresh random-regular d = 4 graphs (ρ = 0.25; per model), hot starts, 1,000 + 1,000 sweeps, R = 100; thin = coarse grid, thick = fine band. Source: results/llm/confirm/sim/locate_*.json")
fig.savefig(os.path.join(ROOT, "results/llm/confirm/fig_binder_step3.png")); print("saved")
