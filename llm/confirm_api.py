"""Confirmatory amplification phase, API side (plan: llm/AMPLIFICATION_CONFIRM_PLAN.md + amendment 1).

  python llm/confirm_api.py prompts                 # K_avg prompt set per model + mapping-symmetry assertion (no calls)
  python llm/confirm_api.py all [--budget 6.0]      # Steps 1-2 in plan order (cached; resumable; one process = one budget)

Outputs: results/llm/confirm/ (selection.json, probes/*.json, comprehension_*.csv/json, temp_*.json, kavg_*.csv).
"""
import argparse, hashlib, itertools, json, math, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery, client, stage3, kernel_test as K
from endpoints import E
from run_stage1 import parse_lp, parse_sample

OUT = os.path.join(client.ROOT, "results", "llm", "confirm")
N_ORDERS = 8
SAMPLED = {"gemini": dict(key="gemini-lite", extra={"top_p": 1.0}, label="confirmatory"),
           "luna": dict(key="gpt6-luna", extra={}, label="exploratory")}
SAMPLES_KAVG, SAMPLES_COMP = 50, 10
PROBE_BUDGET = 2.0
CAND = os.path.join(client.ROOT, "results/llm/catalog/confirm_candidates_2026-10-14.json")
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------- prompts
SWAP = {"aP": "aQ", "aQ": "aP", "rP": "rQ", "rQ": "rP"}


def kavg_prompts(model_key):
    """All k = 4 configurations at rho = 0.25 (35 multisets x own position), mapping 0, <= 8 contact orders x 2 option
    orders. Neutral: mapping 1 of (c, s0) == mapping 0 of the P<->Q-relabelled (c', -s0) (asserted)."""
    rows = []
    for c, s0 in K.configs(4, 0.25):
        types = [t for t, n in zip(battery.TYPES, c) for _ in range(n)]
        perms = sorted(set(itertools.permutations(types)))
        rng = random.Random(stage3._seed("confirm-kavg-orders", model_key, c, s0))
        chosen = perms if len(perms) <= N_ORDERS else rng.sample(perms, N_ORDERS)
        for seq in chosen:
            for o in (0, 1):
                p = stage3.render_live("neutral", c, s0, 0, o, list(seq), True)
                stage3.check_prompt(p, "neutral", c, s0, 0, o, True)
                rows.append(dict(counts="".join(map(str, c)), s0=s0, mapping=0, order=o, seq="".join(seq), prompt=p))
    return pd.DataFrame(rows)


def assert_mapping_symmetry(n=40):
    rng = random.Random(1)
    for c, s0 in rng.sample(K.configs(4, 0.25), n):
        types = [t for t, k in zip(battery.TYPES, c) for _ in range(k)]
        seq = rng.sample(types, len(types)); o = rng.randint(0, 1)
        a = stage3.render_live("neutral", c, s0, 1, o, seq, True)
        b = stage3.render_live("neutral", (c[1], c[0], c[3], c[2]), -s0, 0, o, [SWAP[t] for t in seq], True)
        assert a == b, (c, s0, o)
    return n


# ---------------------------------------------------------------- generic call helpers
def lp_body(ep, prompt, T=1.0):
    return battery.body(ep["model"], ep["tag"], prompt, "lp", ep["roff"], T)


def sample_body(ep, prompt, T=1.0):
    b = battery.body(ep["model"], ep["tag"], prompt, "sample", ep["roff"], T); b.update(ep.get("extra", {}))
    return b


def run_calls(jobs, budget, workers=16):
    """jobs: list of (body, rep, meta). Returns list of (meta, rec or exception)."""
    def one(j):
        b, rep, meta = j
        try:
            return meta, client.chat(b, rep=rep, budget_usd=budget)
        except client.BudgetExceeded:
            raise
        except Exception as e:
            return meta, e
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, jobs))


def rtokens(rec):
    u = (rec["resp"].get("usage") or {}); d = u.get("completion_tokens_details") or {}
    return d.get("reasoning_tokens") or 0


def cost(rec):
    return ((rec["resp"].get("usage") or {}).get("cost")) or 0.0


def ep_from_sampled(name):
    key = SAMPLED[name]["key"]; model, tag, mode, roff, _ = E[key]
    return dict(name=name, key=key, model=model, tag=tag, mode="sample", roff=roff, extra=SAMPLED[name]["extra"])


# ---------------------------------------------------------------- kappa fit (temperature exponent)
def _fit_l(nA, n, s):
    """Per-prompt MLE of l given scale s = T^-kappa for each observation group (arrays over T)."""
    l = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-l * s)); g = np.sum(s * (nA - n * p)); h = np.sum(s * s * n * p * (1 - p)) + 1e-9
        st = g / h; l += st
        if abs(st) < 1e-9:
            break
    return l


def kappa_profile(df, fixed_l=None):
    """df rows: prompt, T, nA, n. Returns kappa_hat, 95% profile CI, LR GOF (p, df). fixed_l: dict prompt -> logit."""
    grid = np.arange(-1.0, 3.0001, 0.005); prompts = sorted(df.prompt.unique()); ll = []
    for k in grid:
        tot = 0.0
        for pr in prompts:
            g = df[df.prompt == pr]; s = g["T"].values ** (-k); nA, n = g.nA.values, g.n.values
            l = fixed_l[pr] if fixed_l is not None else _fit_l(nA, n, s)
            p = np.clip(1 / (1 + np.exp(-l * s)), 1e-12, 1 - 1e-12)
            tot += float(np.sum(nA * np.log(p) + (n - nA) * np.log(1 - p)))
        ll.append(tot)
    ll = np.array(ll); j = int(ll.argmax()); inside = grid[ll >= ll[j] - 1.920729]
    ps = np.clip(df.nA / df.n, 1e-12, 1 - 1e-12)
    llsat = float(np.sum(df.nA * np.log(ps) + (df.n - df.nA) * np.log(1 - ps)))
    dfree = len(df) - (len(prompts) if fixed_l is None else 0) - 1
    from scipy.stats import chi2
    lr = 2 * (llsat - ll[j])
    return dict(kappa=float(grid[j]), ci=[float(inside.min()), float(inside.max())], ci_at_grid_edge=bool(inside.min() <= grid[0] or inside.max() >= grid[-1]),
                lr_gof=float(lr), df=int(dfree), p_gof=float(chi2.sf(lr, dfree)))


# ---------------------------------------------------------------- Step 1 pieces
def lp_logprob_probe(ep, prompts, budget, tag):
    """2 repeats per prompt in lp mode. Returns stats dict."""
    jobs = [(lp_body(ep, p), f"confirm-probe:{tag}:{i}:{r}", (i, r)) for i, p in enumerate(prompts) for r in (0, 1)]
    res = run_calls(jobs, budget); rows = []
    for (i, r), rec in res:
        if isinstance(rec, Exception):
            rows.append(dict(i=i, r=r, err=str(rec)[:200])); continue
        first = rec["resp"].get("first"); q = parse_lp(first)
        has_ab = bool(first and first.get("top") and any(t and t.strip() in ("A", "B") for t, _ in first["top"]))
        rows.append(dict(i=i, r=r, pA=q["pA"], leak=q["leak"], has_ab=has_ab, rt=rtokens(rec), cost=cost(rec),
                         provider=rec["resp"].get("provider")))
    d = pd.DataFrame(rows)
    out = dict(calls=len(d), errors=int(d.get("err", pd.Series(dtype=str)).notna().sum()) if "err" in d else 0)
    if out["errors"]:
        out["first_error"] = str(d.err.dropna().iloc[0])
    ok = d[d.get("has_ab", pd.Series(False, index=d.index)).fillna(False).astype(bool)] if "has_ab" in d else d.iloc[0:0]
    out["share_logprobs_ab"] = float(len(ok) / max(len(d), 1))
    if len(ok):
        y = np.clip(np.log(ok.pA / (1 - ok.pA)), -6, 6); ok = ok.assign(y=y)
        w = ok.pivot_table(index="i", columns="r", values="y")
        w = w.dropna()
        out["repeat_sd"] = float(np.sqrt(np.mean(((w[0] - w[1]) / np.sqrt(2)) ** 2))) if len(w) else None
        out["mean_leak"] = float(ok.leak.mean()); out["reasoning_tokens_max"] = int(d.rt.fillna(0).max())
        out["providers"] = d.provider.dropna().value_counts().to_dict()
        out["meanP"] = ok.groupby("i").pA.mean().to_dict()
    out["cost"] = float(d.get("cost", pd.Series(dtype=float)).fillna(0).sum())
    return out, d


def temp_probe_sampled(ep, prompt, budget, tag):
    """require_parameters + temperature: does any endpoint accept it?"""
    res = {}
    for T in (0.5, 1.5):
        b = sample_body(ep, prompt, T); b["provider"] = dict(b["provider"], require_parameters=True)
        try:
            rec = client.chat(b, rep=f"confirm-tprobe:{tag}:{T}", budget_usd=budget)
            res[str(T)] = dict(ok=True, content=rec["resp"].get("content"), provider=rec["resp"].get("provider"), rt=rtokens(rec))
        except client.BudgetExceeded:
            raise
        except Exception as e:
            res[str(T)] = dict(ok=False, error=str(e)[:300])
    res["accepted"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
    return res


def sampled_kappa_check(ep, budget, tag):
    """Plan §2.3: 20 Stage 1 neutral_own prompts (sampled P in [0.15, 0.85], SHA order); 100 @ T=1, 50 @ 0.5, 50 @ 1.5."""
    d = pd.read_csv(os.path.join(client.ROOT, f"results/llm/stage1/main/{ep['key']}.csv"), dtype={"counts": str})
    d = d[(d.arm == "neutral_own") & (d.perm == 0) & (d.mapping == 0)].copy(); d["counts"] = d.counts.str.zfill(4)
    g = d.groupby(["counts", "s0", "order"]).agg(nA=("answer", lambda x: (x == "A").sum()), n=("answer", lambda x: x.notna().sum())).reset_index()
    g["p"] = g.nA / g.n; g["ident"] = [sha(f"{c}|{s}|{o}") for c, s, o in zip(g.counts, g.s0, g.order)]
    q = g[(g.p >= 0.15) & (g.p <= 0.85)].sort_values("ident").head(20)
    jobs = []
    for _, r in q.iterrows():
        text, _ = battery.render("neutral", tuple(int(x) for x in r.counts), int(r.s0), 0, int(r.order), 0)
        for T, n in ((1.0, 100), (0.5, 50), (1.5, 50)):
            jobs += [(sample_body(ep, text, T), f"confirm-temp:{tag}:{i}", (r.ident, T)) for i in range(n)]
    res = run_calls(jobs, budget)
    rows = [dict(prompt=m[0], T=m[1], ans=(None if isinstance(rec, Exception) else parse_sample(rec["resp"].get("content"))),
                 err=isinstance(rec, Exception), cost=0 if isinstance(rec, Exception) else cost(rec)) for m, rec in res]
    S = pd.DataFrame(rows)
    inv = S.groupby("T").apply(lambda x: float(x.ans.isna().mean())).to_dict()
    A = S[S.ans.notna()].groupby(["prompt", "T"]).agg(nA=("ans", lambda x: (x == "A").sum()), n=("ans", "size")).reset_index()
    k = kappa_profile(A)
    k.update(n_prompts=int(q.shape[0]), invalid_rate_by_T={str(a): b for a, b in inv.items()}, cost=float(S.cost.sum()),
             passes=bool(k["ci"][0] <= 1 <= k["ci"][1] and k["p_gof"] > 0.01 and k["ci"][0] > 0))
    S.to_csv(os.path.join(OUT, f"temp_{tag}_samples.csv"), index=False)
    return k


def lp_temp_check(ep, kavg_lp, budget, tag):
    """§2.1 for the selected logprob model: 20 prompts from its K_avg (P in [0.1, 0.9], SHA order), 50 samples at
    T = 0.5 and 1.5 against the logprob prediction; slope CI contains 1/T and chi2 p > 0.01 (temp_check method)."""
    import temp_check as TC
    from analysis import fit_ce
    from scipy.stats import chi2
    g = kavg_lp.groupby("prompt_id").agg(pA=("pA", "mean"), prompt=("prompt", "first")).reset_index()
    g["ident"] = g.prompt.map(sha)
    q = g[(g.pA >= 0.1) & (g.pA <= 0.9)].sort_values("ident").head(20)
    jobs = [(sample_body(ep, r.prompt, T), f"confirm-temp:{tag}:{i}", (r.ident, float(r.pA), T)) for _, r in q.iterrows() for T in (0.5, 1.5) for i in range(50)]
    res = run_calls(jobs, budget)
    S = pd.DataFrame([dict(ident=m[0], p1=m[1], T=m[2], answer=None if isinstance(rec, Exception) else parse_sample(rec["resp"].get("content")),
                           cost=0 if isinstance(rec, Exception) else cost(rec)) for m, rec in res])
    S.to_csv(os.path.join(OUT, f"temp_{tag}_samples.csv"), index=False)
    out = {}
    for T, gg in S.groupby("T"):
        inv = float(gg.answer.isna().mean()); gg = gg[gg.answer.notna()]
        agg = gg.groupby("ident").agg(p1=("p1", "first"), n=("answer", "size"), nA=("answer", lambda x: (x == "A").sum())).reset_index()
        l1 = np.log(agg.p1 / (1 - agg.p1)).values; pred = 1 / (1 + np.exp(-l1 / T))
        x2 = float((((agg.nA - agg.n * pred) ** 2) / (agg.n * pred * (1 - pred))).sum())
        X = np.column_stack([l1]); y = agg.nA.values / agg.n.values; w = agg.n.values.astype(float)
        slope = float(fit_ce(X, y, w)[0]); rng = np.random.default_rng(0); bs = []
        for _ in range(1000):
            idx = rng.integers(0, len(agg), len(agg)); nA = rng.binomial(agg.n.values[idx], y[idx])
            bs.append(fit_ce(X[idx], nA / agg.n.values[idx], w[idx])[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[str(T)] = dict(slope=slope, slope_ci=[float(lo), float(hi)], expected=1 / T, chi2=x2, df=len(agg), p_chi2=float(chi2.sf(x2, len(agg))),
                           invalid_rate=inv, passes=bool(lo <= 1 / T <= hi and chi2.sf(x2, len(agg)) > 0.01))
    out["passes"] = all(v["passes"] for v in out.values()); out["cost"] = float(S.cost.sum())
    return out


# ---------------------------------------------------------------- comprehension
def comprehension(ep, budget, tag):
    items = list(battery.comprehension_items()); rows = []
    if ep["mode"] == "lp":
        res = run_calls([(lp_body(ep, it["prompt"]), f"confirm-comp:{tag}", j) for j, it in enumerate(items)], budget)
        for j, rec in res:
            it = items[j]; q = parse_lp(None if isinstance(rec, Exception) else rec["resp"].get("first"))
            rows.append(dict(key=tag, model=ep["model"], tag=ep["tag"], question=it["question"], counts="".join(map(str, it["counts"])),
                             correct=it["correct"], pA=q["pA"], leak=q["leak"]))
        d = pd.DataFrame(rows); path = os.path.join(OUT, f"comprehension_{tag}.csv"); d.to_csv(path, index=False)
        import comprehension as C
        r = C.score(path)
    else:
        jobs = [(sample_body(ep, it["prompt"]), f"confirm-comp:{tag}:{s}", (j, s)) for j, it in enumerate(items) for s in range(SAMPLES_COMP)]
        res = run_calls(jobs, budget)
        for (j, s), rec in res:
            it = items[j]
            rows.append(dict(j=j, question=it["question"], counts="".join(map(str, it["counts"])), correct=it["correct"],
                             ans=None if isinstance(rec, Exception) else parse_sample(rec["resp"].get("content"))))
        d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, f"comprehension_{tag}.csv"), index=False)
        per = d.groupby(["j", "question"]).apply(lambda x: int((x.ans == x.correct).sum() >= 6)).rename("ok").reset_index()
        r = {"key": tag, "n": int(per.shape[0]), "invalid_rate": float(d.ans.isna().mean())}
        for qn, gg in per.groupby("question"):
            r[f"acc_{qn}"] = round(float(gg.ok.mean()), 4)
        r["passes"] = all(r.get(f"acc_{q}", 0) >= 0.9 for q in ("own", "ally_majority", "rival_majority"))
    json.dump(r, open(os.path.join(OUT, f"comprehension_{tag}.json"), "w"), indent=1, default=str)
    return r


# ---------------------------------------------------------------- K_avg
def kavg(ep, budget, tag, reps):
    P = kavg_prompts(ep["key"]); P["prompt_id"] = range(len(P)); jobs = []
    for _, r in P.iterrows():
        for i in range(reps):
            b = lp_body(ep, r.prompt) if ep["mode"] == "lp" else sample_body(ep, r.prompt)
            jobs.append((b, f"confirm-kavg:{tag}:{r.counts}:{r.s0}:{r.order}:{r.seq}:{i}", (r.prompt_id, i)))
    res = run_calls(jobs, budget); rows = []
    for (pid, i), rec in res:
        r = P.iloc[pid]; base = dict(prompt_id=pid, counts=r.counts, s0=r.s0, mapping=0, order=r.order, seq=r.seq, rep=i)
        if isinstance(rec, Exception):
            rows.append(dict(base, valid=False, err=str(rec)[:120])); continue
        if ep["mode"] == "lp":
            q = parse_lp(rec["resp"].get("first")); valid = q["pA"] is not None and q["leak"] <= stage3.LEAK_MAX
            rows.append(dict(base, pA=q["pA"], leak=q["leak"], valid=valid, rt=rtokens(rec), provider=rec["resp"].get("provider"), cost=cost(rec)))
        else:
            a = parse_sample(rec["resp"].get("content"))
            rows.append(dict(base, pA=(1.0 if a == "A" else 0.0) if a else None, valid=a is not None, rt=rtokens(rec),
                             provider=rec["resp"].get("provider"), cost=cost(rec)))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, f"kavg_{tag}.csv"), index=False)
    P[["prompt_id", "counts", "s0", "order", "seq", "prompt"]].to_csv(os.path.join(OUT, f"kavg_prompts_{tag}.csv"), index=False)
    return dict(calls=len(d), invalid=float((~d.valid).mean()), cost=float(d.get("cost", pd.Series(dtype=float)).fillna(0).sum()),
                providers=d.get("provider", pd.Series(dtype=str)).value_counts().to_dict(), reasoning_tokens_max=int(d.get("rt", pd.Series([0])).fillna(0).max())), d.merge(P[["prompt_id", "prompt"]], on="prompt_id")


# ---------------------------------------------------------------- candidate loop
FIRST = {"deepseek": "deepseek", "openai": "openai"}
RANK = {"bf16": 0, "fp16": 0, "fp8": 1, "unknown": 2, None: 2, "fp4": 3}


def repin(c, blocked):
    """Amendment 1 pin rule over the endpoints listing logprobs + temperature, minus providers this account's
    OpenRouter guardrails remove (amendment 2)."""
    eps = [e for e in c["endpoints"] if e["logprobs"] and e["temperature"] and e["tag"] not in blocked]
    if not eps:
        return None
    fp = [e for e in eps if e["tag"] == FIRST.get(c["id"].split("/")[0])]
    return (fp[0] if fp else sorted(eps, key=lambda e: (RANK.get(e["quant"], 2), e["prompt"], e["tag"]))[0])["tag"]
def select_model(budget):
    path = os.path.join(OUT, "selection.json")
    sel = json.load(open(path)) if os.path.exists(path) else {"tried": []}
    if sel.get("selected"):
        return sel
    C = json.load(open(CAND)); probe_spent = sum(t.get("cost", 0) for t in sel["tried"])
    done = {t["id"] for t in sel["tried"]}
    for i, c in enumerate(C, 1):
        if c["id"] in done:
            if any(t["id"] == c["id"] and t.get("passes") and t.get("comprehension_passes") for t in sel["tried"]):
                break
            continue
        if probe_spent >= PROBE_BUDGET:
            log("probe budget exhausted"); break
        pin = repin(c, sel.setdefault("guardrail_blocked", {}).get(c["id"], []))
        if pin is None:
            t = dict(rank=i, id=c["id"], passes=False, error="no reachable endpoint after guardrail exclusions"); sel["tried"].append(t); continue
        e = next(x for x in c["endpoints"] if x["tag"] == pin)
        ep = dict(name=f"cand{i}", key=c["id"], model=c["id"], tag=pin, mode="lp", roff=bool(e["reasoning"]), extra={})
        prompts = kavg_prompts("confirm-lp").assign(h=lambda x: x.prompt.map(sha)).sort_values("h").prompt.head(20).tolist()
        t = dict(rank=i, id=c["id"], pin=pin, pin_rule_original=c["pin"], reasoning_param=bool(e["reasoning"]))
        try:
            a, _ = lp_logprob_probe(ep, prompts, budget, f"cand{i}")
            t["lp"] = {k: v for k, v in a.items() if k != "meanP"}; t["cost"] = a["cost"]
            if a["errors"] == a["calls"] and "Filter by Guardrails removed" in a.get("first_error", "") and pin in a["first_error"].split("Filter by Guardrails removed")[1].split(";")[0]:
                sel["guardrail_blocked"].setdefault(c["id"], []).append(pin)            # amendment 2: re-pin, probe again
                log("guardrail blocks", c["id"], pin, "-> re-pin"); json.dump(sel, open(path, "w"), indent=1, default=str)
                return select_model(budget)
            pass_a = (a["share_logprobs_ab"] >= 0.95 and a.get("repeat_sd") is not None and a["repeat_sd"] <= 0.3
                      and a.get("mean_leak", 1) < 0.05 and (not e["reasoning"] or a.get("reasoning_tokens_max", 1) == 0))
            t["pass_a"] = bool(pass_a)
            if pass_a:
                mp = pd.Series(a["meanP"]); top5 = (mp - 0.5).abs().sort_values().index[:5]
                jobs = [(sample_body(ep, prompts[j], T), f"confirm-probeT:cand{i}:{s}", (j, T)) for j in top5 for T in (0.5, 1.5) for s in range(40)]
                res = run_calls(jobs, budget)
                S = pd.DataFrame([dict(prompt=m[0], T=m[1], ans=None if isinstance(rec, Exception) else parse_sample(rec["resp"].get("content")),
                                       c=0 if isinstance(rec, Exception) else cost(rec)) for m, rec in res])
                t["cost"] += float(S.c.sum())
                A = S[S.ans.notna()].groupby(["prompt", "T"]).agg(nA=("ans", lambda x: (x == "A").sum()), n=("ans", "size")).reset_index()
                fl = {j: float(np.clip(np.log(mp[j] / (1 - mp[j])), -6, 6)) for j in top5}
                k = kappa_profile(A, fixed_l=fl); t["temp"] = dict(k, invalid_rate=float(S.ans.isna().mean()))
                t["pass_b"] = bool(k["ci"][0] > 0)
            t["passes"] = bool(t.get("pass_a") and t.get("pass_b"))
        except client.BudgetExceeded:
            raise
        except Exception as ex:
            t["error"] = repr(ex)[:300]; t["passes"] = False
        probe_spent += t.get("cost", 0); t["probe_spent_cum"] = probe_spent
        if t["passes"]:
            r = comprehension(ep, budget, f"cand{i}"); t["comprehension_passes"] = bool(r["passes"]); t["comprehension"] = r
        sel["tried"].append(t); json.dump(sel, open(path, "w"), indent=1, default=str); log("candidate", i, c["id"], {k: t.get(k) for k in ("pass_a", "pass_b", "comprehension_passes", "cost")})
        if t["passes"] and t.get("comprehension_passes"):
            sel["selected"] = dict(ep, rank=i); break
    if not sel.get("selected"):
        model, tag, mode, roff, _ = E["gpt4o-mini"]
        sel["selected"] = dict(name="fallback", key="gpt4o-mini", model=model, tag=tag, mode="lp", roff=roff, extra={}, fallback=True)
        log("FALLBACK to GPT-4o-mini")
    json.dump(sel, open(path, "w"), indent=1, default=str)
    return sel


# ---------------------------------------------------------------- main
def all_steps(budget):
    os.makedirs(OUT, exist_ok=True); R = {}
    def save():
        R["spend"] = client.spend(); json.dump(R, open(os.path.join(OUT, "api_summary.json"), "w"), indent=1, default=str)
    # Step 1: 1/T checks for P1 and P4 (exactly the Qwen-9B check)
    import temp_check as TC
    for key in ("qwen-122b", "nemo-12b"):
        f = os.path.join(client.ROOT, f"results/llm/temperature_verify_{key}.json")
        if not os.path.exists(f):
            log("1/T verify", key); TC.verify(key)
        R[f"temp_{key}"] = json.load(open(f)); save()
    # probes for the sampled models: logprobs + temperature acceptance
    for nm in ("gemini", "luna"):
        ep = ep_from_sampled(nm); p0 = kavg_prompts(ep["key"]).prompt.iloc[0]
        lpb = dict(ep, mode="lp")
        try:
            a, _ = lp_logprob_probe(lpb, [p0], budget, f"{nm}-lpprobe")
        except client.BudgetExceeded:
            raise
        R[f"{nm}_logprob_probe"] = {k: v for k, v in a.items() if k != "meanP"}
        R[f"{nm}_temp_probe"] = temp_probe_sampled(ep, p0, budget, nm); save()
        log(nm, "probes", R[f"{nm}_logprob_probe"].get("share_logprobs_ab"), R[f"{nm}_temp_probe"]["accepted"])
    # selection loop (includes comprehension of the selected model)
    sel = select_model(budget); R["selection"] = sel; save(); ep = sel["selected"]
    if sel["selected"].get("fallback"):
        R["comprehension_selected"] = comprehension(ep, budget, "fallback"); save()
    # comprehension for the sampled models
    for nm in ("gemini", "luna"):
        f = os.path.join(OUT, f"comprehension_{nm}.json")
        R[f"comprehension_{nm}"] = json.load(open(f)) if os.path.exists(f) else comprehension(ep_from_sampled(nm), budget, nm); save()
        log("comprehension", nm, R[f"comprehension_{nm}"])
    # K_avg + 1/T for the selected logprob model
    tag = "selected"; s, dsel = kavg(ep, budget, tag, 2); R["kavg_selected"] = s; save(); log("kavg selected", s)
    R["temp_selected"] = lp_temp_check(ep, dsel[dsel.valid], budget, tag); save(); log("1/T selected", R["temp_selected"]["passes"])
    # sampled models: 1/T (kappa) check where temperature is accepted, then K_avg
    for nm in ("gemini", "luna"):
        e2 = ep_from_sampled(nm)
        R[f"temp_{nm}"] = sampled_kappa_check(e2, budget, nm) if R[f"{nm}_temp_probe"]["accepted"] else dict(passes=False, reason="temperature not accepted (require_parameters refused)")
        save(); log("1/T", nm, R[f"temp_{nm}"].get("passes"), R[f"temp_{nm}"].get("kappa"))
        s, _ = kavg(e2, budget, nm, SAMPLES_KAVG); R[f"kavg_{nm}"] = s; save(); log("kavg", nm, s)
    log("DONE spend", client.spend())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--budget", type=float, default=6.0)
    a = ap.parse_args()
    if a.cmd == "prompts":
        print("mapping symmetry asserted on", assert_mapping_symmetry(), "random cells")
        for k in ("confirm-lp", "gemini-lite", "gpt6-luna", "gemma-31b-local"):
            print(k, len(kavg_prompts(k)), "prompts")
    elif a.cmd == "all":
        assert_mapping_symmetry(); all_steps(a.budget)
