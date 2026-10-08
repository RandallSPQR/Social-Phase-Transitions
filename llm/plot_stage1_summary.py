"""Stage 1 cross-endpoint figures (Tufte style, matching sim/plots.py).

  python ../llm/plot_stage1_summary.py

fig: figures/llm_stage1_comprehension.png  – comprehension accuracy per endpoint and question type
fig: figures/llm_stage1_invariants.png     – temperature-invariant response quantities per endpoint × framing:
     valence asymmetry (b_rival − b_ally)/β, inertia γ/β, degree scaling α, letter/content fields h/β
fig: figures/llm_stage1_H4.png             – β at T = 1, small vs large per family (H4), comprehension-annotated
All intervals: 95% percentile CIs from 1,000 cluster-bootstrap draws (cells), ratios computed draw-by-draw.
"""
import glob, json, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_stage1 import C, note
from endpoints import E, H4_PAIRS
from summarize import WORKHORSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
ORDER = ["llama-8b", "llama-70b", "llama-70b-pa", "qwen-9b", "qwen-122b", "gemma-26b", "gemma-26b-cw", "gemma-31b",
         "nemo-12b", "mistral-small", "mistral-large", "gpt4o-mini", "gpt4o-mini_sampled", "haiku", "gpt6-luna", "gemini-lite"]
NAME = {"llama-8b": "Llama-3.1 8B", "llama-70b": "Llama-3.3 70B", "llama-70b-pa": "Llama-3.3 70B (Parasail fp8)",
        "qwen-9b": "Qwen3.5 9B", "qwen-122b": "Qwen3.5 122B-A10B", "gemma-26b": "Gemma-4 26B-A4B",
        "gemma-31b": "Gemma-4 31B", "nemo-12b": "Mistral Nemo 12B", "mistral-small": "Mistral Small 24B",
        "mistral-large": "Mistral Large 4", "gpt4o-mini": "GPT-4o-mini (bridge)",
        "gemma-26b-cw": "Gemma-4 26B-A4B (CoreWeave)", "gemma-31b-nv": "Gemma-4 31B (Novita)",
        "haiku": "Claude Haiku 5.5 (sampled)", "gpt6-luna": "GPT-6-luna (sampled)", "gemini-lite": "Gemini 3.1 Flash-Lite (sampled)", "gpt4o-mini_sampled": "GPT-4o-mini (bridge, sampled)"}
FRAME = {"neutral_own": ("neutral", "#111111", -0.22), "political": ("political", C["hl"], 0.0),
         "workplace": ("workplace", "#4e79a7", 0.22)}


def load():
    F = {}
    for p in glob.glob(os.path.join(ROOT, "results", "llm", "fits", "main", "*.json")):
        j = json.load(open(p)); F[os.path.basename(p)[:-5]] = {a["arm"]: a for a in j["arms"]}   # key = file stem
    comp = pd.read_csv(os.path.join(ROOT, "results", "llm", "comprehension_summary.csv")).set_index("key")
    return F, comp


def ci(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return np.percentile(x, [2.5, 50, 97.5]) if len(x) else (np.nan,) * 3


def fig_comprehension(comp):
    keys = [k for k in ORDER if k in comp.index]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(keys) + 1.6))
    marks = {"own": ("o", "own position"), "ally_majority": ("s", "ally majority"), "rival_majority": ("D", "rival majority")}
    for i, k in enumerate(keys):
        r = comp.loc[k]; col = C["text"] if r["passes"] else C["hl"]
        for q, (m, lab) in marks.items():
            ax.plot(r[f"acc_{q}"], i, m, ms=5, color=col, mfc="none" if q != "own" else col, mew=1.1)
        why = "" if r["passes"] else ("  — fails: answers majority questions only after reasoning (leak > 5%)"
                                         if r.get("leak_gt5", 0) > 0.2 * r["n"] else "  — fails: does not parse task")
        ax.text(-0.01, i, NAME[k] + why, ha="right", va="center", fontsize=10, color=col)
    ax.axvline(0.9, color=C["axis"], lw=0.8, ls="--"); ax.text(0.9, -0.9, "pass: ≥ 90% on every question type", fontsize=9, color=C["text3"], ha="center")
    for j, (q, (m, lab)) in enumerate(marks.items()):   # symbol key in the empty upper-middle region
        ax.plot(0.30 + 0.17 * j, -0.9, m, ms=5, color=C["text2"], mfc="none" if q != "own" else C["text2"], mew=1.1)
        ax.text(0.315 + 0.17 * j, -0.9, lab, fontsize=9.5, color=C["text2"], va="center")
    ax.set_xlim(0, 1.01); ax.set_ylim(len(keys) - 0.5, -1.3); ax.set_yticks([])
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_bounds(0, 1.0)
    ax.set_xlabel("share of prompts answered correctly (renormalised P > 0.5, leak ≤ 5%)")
    n_fail = int((~comp.loc[keys, "passes"].astype(bool)).sum())
    fig.text(0.02, 1.0, f"Most models read the scenario correctly; {n_fail} of {len(keys)} endpoints fail the 90% bar", fontsize=15, color=C["text"])
    fig.text(0.02, 0.965, "Own position (836 prompts), majority of allies and of rivals (640 each), k = 1–6, both letter orders", fontsize=11, color=C["text2"])
    note(fig, "Source: OpenRouter, pinned endpoints (llm/endpoints.py), first-token logprobs at T = 1; one call per prompt. Neutral framing. "
              "PREREG Amendment 3a.", y=-0.01)
    fig.savefig(os.path.join(FIG, "llm_stage1_comprehension.png")); plt.close(fig)


def quantities(arm):
    d = arm["draws"]; b, ba, br = np.array(d["beta"]), np.array(d["b_a"]), np.array(d["b_r"])
    out = {"valence": br / ba, "alpha": np.array(d["alpha"])}
    if "gamma" in d:
        out["inertia"] = np.array(d["gamma"]) / b
    c = arm["coef"]
    # point estimates (draw-free) for fields, CI via draw-wise beta with the field's own bootstrap spread
    return out


def fig_invariants(F, comp):
    from summarize import pooled_sd, NOISE_GATE
    keys = [k for k in ORDER if k in F]
    noisy = {k for k in keys if F[k].get("neutral_own", {}).get("repeat_sd") is not None and pooled_sd(F[k]) > NOISE_GATE}
    panels = [("valence", "rival ÷ ally weight  b_rival / b_ally\n(1 symmetric; < 0 rivals attract)", (-1.5, 2.5)),
              ("inertia", "inertia in neighbour-equivalents\nγ / β", (-10, 30)), ("alpha", "degree scaling  α\n(0 = sum, 1 = average)", (-1, 3))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.5 * len(keys) + 2.2), sharey=True)
    fig.subplots_adjust(wspace=0.22)
    for ax, (q, label, lim) in zip(axes, panels):
        for i, k in enumerate(keys):
            gated = (k in comp.index and not comp.loc[k, "passes"]) or k in noisy
            for arm, (fl, col, off) in FRAME.items():
                if arm not in F[k]: continue
                Q = quantities(F[k][arm])
                if q not in Q: continue
                lo, med, hi = ci(Q[q]); al = 0.35 if gated else 1
                ax.plot([max(lo, lim[0]), min(hi, lim[1])], [i + off] * 2, color=col, lw=1.2, alpha=al)
                if lim[0] <= med <= lim[1]:
                    ax.plot(med, i + off, "o", ms=3.5, color=col, alpha=al)
                else:   # off-scale: edge marker with the value
                    xe = lim[1] if med > lim[1] else lim[0]
                    ax.plot(xe, i + off, ">" if med > lim[1] else "<", ms=4.5, color=col, alpha=al)
                    ax.text(xe, i + off, f" {med:.0f} " if abs(med) >= 10 else f" {med:.1f} ", fontsize=7.5, color=col, alpha=al,
                            va="center", ha="right" if med > lim[1] else "left")
        ax.axvline(0, color=C["axis"], lw=0.6)
        if q == "alpha": ax.axvline(1, color=C["axis"], lw=0.6, ls=":")
        if lim: ax.set_xlim(*lim)
        ax.set_title(label, loc="left", fontsize=11.5, color=C["text"])
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(keys)))
    axes[0].set_yticklabels([NAME[k] + (" *" if k in comp.index and not comp.loc[k, "passes"] else "") + (" †" if k in noisy else "")
                             for k in keys], fontsize=10, color=C["text2"])
    axes[0].set_ylim(len(keys) - 0.5, -0.8)
    for arm, (fl, col, off) in FRAME.items():
        axes[2].text(axes[2].get_xlim()[1], -0.55 + off * 1.8, fl, color=col, fontsize=9.5, ha="right", va="center")
    fig.text(0.02, 1.02, "Rivals count for less than allies, inertia appears once positions have content, and summing vs averaging depends on the model",
             fontsize=15, color=C["text"])
    fig.text(0.02, 0.985, "Ratios to β do not change with sampling temperature (every coefficient scales as 1/T). Dots: bootstrap median; bars: 95% CI.",
             fontsize=11, color=C["text2"])
    note(fig, "Source: OpenRouter, pinned endpoints, first-token logprobs at T = 1, two calls per prompt; 209 neighbour multisets × own position per framing, label/order controls; "
              "CIs from 1,000 cluster-bootstrap draws. Sampled endpoints: 30 samples per prompt, k ≤ 4, neutral + political only.\n* = fails comprehension check (≥ 90%); † = fails 0.3-nat repeat-noise gate; both faded, descriptive only. α is interpretable only where H3 is negative (Amendment 1.4).", y=-0.01)
    fig.savefig(os.path.join(FIG, "llm_stage1_invariants.png")); plt.close(fig)


def fig_H4(F, comp):
    from summarize import pooled_sd, NOISE_GATE
    noisy = {k for k in F if pooled_sd(F[k]) > NOISE_GATE}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    for ax, (arm, (fl, col, _)) in zip(axes, FRAME.items()):
        for j, (s, l) in enumerate(H4_PAIRS):
            if not all(k in F and arm in F[k] for k in (s, l)): continue
            y = []
            for x, k in ((0, s), (1, l)):
                lo, med, hi = ci(F[k][arm]["draws"]["beta"]); y.append(F[k][arm]["coef"]["beta"]["est"])
                ax.plot([x, x], [lo, hi], color=C["series"], lw=1.1)
            gated = [k for k in (s, l) if (k in comp.index and not comp.loc[k, "passes"]) or k in noisy]
            mark = (" *" if any(k in comp.index and not comp.loc[k, "passes"] for k in (s, l)) else "") + (" †" if any(k in noisy for k in (s, l)) else "")
            c = C["text3"] if gated else C["text"]
            ax.plot([0, 1], y, marker="o", ms=3.5, color=c, lw=1.2, ls="--" if gated else "-")
            ax.text(1.06, y[1], NAME[l].split(" ")[0] + mark, color=c, fontsize=9.5, va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["small", "large"]); ax.set_xlim(-0.2, 1.45)
        ax.set_title(fl, loc="left", fontsize=12, color=col)
    axes[0].set_ylabel("β at T = 1 (logit units per neighbour)")
    fig.text(0.02, 1.03, "H4: β is higher for the larger model in every family, but only Qwen passes both the comprehension and noise gates", fontsize=15, color=C["text"])
    fig.text(0.02, 0.98, "β is a sampling knob (β(T) = β(1)/T); a size effect at T = 1 can also reflect task comprehension. Bars: 95% CI.",
             fontsize=11, color=C["text2"])
    note(fig, "Source: OpenRouter, pinned endpoints; pairs: Llama 3.1-8B→3.3-70B, Qwen3.5 9B→122B-A10B, Gemma-4 26B-A4B→31B, Mistral Nemo→Large 4 "
              "(confounds listed in PREREG §6).\nDashed = excluded from H4: * a member fails the comprehension check; † a member fails the 0.3-nat repeat-noise gate (no passing backup). "
              "H4 required ≥ 3 of 4 families; 1 was testable, so H4 is not supported.", y=-0.02)
    fig.savefig(os.path.join(FIG, "llm_stage1_H4.png")); plt.close(fig)


if __name__ == "__main__":
    F, comp = load()
    fig_comprehension(comp); fig_invariants(F, comp); fig_H4(F, comp); print("ok")
