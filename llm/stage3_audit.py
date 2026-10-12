"""Stage 3 harness audit (plan: llm/STAGE3_AUDIT_PLAN.md). Audit-only: imports the harness (llm/stage3.py) and the
simulator (llm/surrogate.py, llm/stage2.py) unchanged; never makes an API call (the replay uses a cache-only lookup
that raises on a miss).

  python llm/stage3_audit.py logs              # B1–B10 and C from the chain logs + raw call cache
  python llm/stage3_audit.py mock [--reps 40]  # A: surrogate mock agent through the live harness (resumable)
  python llm/stage3_audit.py surrogate         # B11
  python llm/stage3_audit.py report            # pass/fail table -> results/llm/stage3_audit/audit_table.md

Outputs: results/llm/stage3_audit/.
"""
import argparse, collections, inspect, json, math, os, random, re, subprocess, sys, textwrap
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery, client, stage2, stage3, surrogate as S
from endpoints import E
from run_stage1 import parse_sample

ROOT = stage3.ROOT
OUT = os.path.join(ROOT, "results", "llm", "stage3_audit")
POINTS = list(stage3.POINTS)
MEAS = ["abs_m", "m", "persistence", "excess_unsat", "flip_rate", "abs_q"]
FLOOR = None


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    json.dump(obj, open(os.path.join(OUT, name), "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def floor_for(rho):
    global FLOOR
    if FLOOR is None:
        f = pd.read_csv(os.path.join(ROOT, "results/llm/stage2/floor.csv"))
        FLOOR = f[(f.graph == "rrg") & (f.N == stage3.N)].groupby("rho").unsat_floor.mean()
    return float(FLOOR.loc[rho])


# ======================================================================= raw cache
def load_cache(model):
    """All Stage 3 records (tag -> record) and the set of Stage 1 body signatures (all fields except messages)."""
    recs, sig1 = {}, collections.Counter()
    with open(client._path(model)) as f:
        for line in f:
            r = json.loads(line)
            if isinstance(r["rep"], str) and r["rep"].startswith("stage3:"):
                recs[r["rep"]] = r
            else:
                b = {k: v for k, v in r["body"].items() if k != "messages"}
                sig1[json.dumps(b, sort_keys=True)] += 1
    return recs, sig1


# ======================================================================= independent prompt parser
RE_LAST = re.compile(r"Which position do you take now\? Reply with exactly one letter, ([AB]) or ([AB]), and nothing else\.")
RE_OWN = re.compile(r"You currently hold position ([AB])(?: \((.+)\))?\.")
RE_CON = re.compile(r"- (An ally|A rival) holds ([AB])(?: \((.+)\))?\.")


def parse_prompt(text, framing):
    """Return dict(order, own_letter, contacts=[(is_ally, letter)], letter_of_P or None, gloss_ok)."""
    L = text.split("\n")
    m = RE_LAST.fullmatch(L[-1]); X, Y = m.group(1), m.group(2)
    assert {X, Y} == {"A", "B"}
    own = [RE_OWN.fullmatch(l) for l in L]; own = [o for o in own if o]
    assert len(own) == 1
    cons = [RE_CON.fullmatch(l) for l in L]; cons = [c for c in cons if c]
    out = dict(order=0 if X == "A" else 1, own_letter=own[0].group(1),
               contacts=[(c.group(1) == "An ally", c.group(2)) for c in cons], letter_of_P=None, gloss_ok=True)
    F = battery.FRAMINGS[framing]["content"]
    if F is not None:                                      # content gloss: which letter carries content P
        intro = L[0]
        gA = re.search(r"\bA \(([^)]+)\)", intro).group(1); gB = re.search(r"\bB \(([^)]+)\)", intro).group(1)
        assert {gA, gB} == {F["P"], F["Q"]}
        out["letter_of_P"] = "A" if gA == F["P"] else "B"
        g = {"A": gA, "B": gB}
        ok = own[0].group(2) == g[own[0].group(1)] and all(c.group(3) == g[c.group(2)] for c in cons)
        out["gloss_ok"] = bool(ok)
    else:
        out["gloss_ok"] = own[0].group(2) is None and all(c.group(3) is None for c in cons)
    return out


def pA_from_raw(first):
    """Independent re-implementation of the renormalised P(A) and leak from the raw first-token top-5."""
    if not first or not first.get("top"):
        return None, None
    top = first["top"]
    pa = sum(math.exp(l) for t, l in top if t is not None and t.strip() == "A")
    pb = sum(math.exp(l) for t, l in top if t is not None and t.strip() == "B")
    leak = max(0.0, 1.0 - pa - pb)
    fl = math.exp(min(l for _, l in top))
    pa2, pb2 = (pa or fl), (pb or fl)
    return pa2 / (pa2 + pb2), leak


# ======================================================================= simulator's own measurement code, verbatim
def replay_measure_fn():
    """surrogate.run with ONLY its dynamics replaced by a replay of logged updates; every measurement line is the
    simulator's own source, executed unchanged (asserted by exact-substring replacement)."""
    src = textwrap.dedent(inspect.getsource(S.run.py_func))
    rep = [
        ("@njit(cache=True)\n", ""),
        ("def run(indptr, nbr, tie, typ, par, n_warm, n_meas, seed, explicit_jitter, init_bias):",
         "def run_replay(indptr, nbr, tie, S0, AG, NEW, n_warm, n_meas):"),
        ("    np.random.seed(seed)\n", ""),
        ("            s[r, i] = 1 if np.random.random() < 0.5 + init_bias else -1", "            s[r, i] = S0[r, i]"),
    ]
    for a, b in rep:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    a = src.index("                i = np.random.randint(N)\n")
    b = src.index("                s[r, i] = new\n")
    src = src[:a] + ("                i = AG[r, t, _u]\n                old = s[r, i]\n                new = NEW[r, t, _u]\n") + src[b:]
    ns = {"np": np, "_sig": S._sig.py_func}
    exec(src, ns)
    return ns["run_replay"], src


# ======================================================================= B1–B10, C
def audit_logs():
    run_replay, replay_src = replay_measure_fn()
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "replay_measure_source.py"), "w").write(replay_src)
    reported = pd.read_csv(os.path.join(ROOT, "results/llm/stage3/measures_by_graph.csv"))
    summary = pd.read_csv(os.path.join(ROOT, "results/llm/stage3/summary.csv")).set_index("point")
    res = {k: {} for k in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "C"]}
    calib_rows, init_z, b10_rows, all_inv = [], [], [], []
    caches = {}
    for pid in POINTS:
        key, framing, rho, contacts = stage3.POINTS[pid]
        model, tag, mode, roff, _ = E[key]
        if model not in caches:
            caches[model] = load_cache(model)
        recs, sig1 = caches[model]
        ref_body = {k: v for k, v in battery.body(model, tag, "", mode, roff, 1.0).items() if k != "messages"}
        ref_sig = json.dumps(ref_body, sort_keys=True)
        battery_pA = None
        if contacts:
            d1 = pd.read_csv(os.path.join(ROOT, f"results/llm/stage1/main/{key}.csv"), dtype={"counts": str})
            d1 = d1[(d1.arm == stage2.FRAME_ARM[framing]) & (d1.perm == 0)]
            d1["counts"] = d1["counts"].str.zfill(4)
            g1 = d1.groupby(["counts", "s0", "mapping", "order"])
            battery_pA = pd.DataFrame({"pA": g1.pA.mean(), "leak_max": g1.leak.max()})
            served1 = collections.Counter(d1["served"])
        c = collections.Counter()
        prov = collections.Counter()
        tz, sweeps_ok, per_agent_sweep = [], True, collections.Counter()
        agent_chi_p, rho_real, sym_ok = [], [], True
        order_pos = np.zeros((4, 3))           # [listed position j] -> (observed hits, expected, variance) for CSR-first type
        bal = collections.Counter(); inv_rows = []
        draw_z = [0.0, 0.0]
        for g in range(stage3.N_GRAPHS):
            ip, nb, tie = stage3.graph(pid, g)
            recv = np.repeat(np.arange(stage3.N), np.diff(ip))
            # B5: symmetry and realised rho
            em = {(int(a), int(b)): int(t) for a, b, t in zip(recv, nb, tie)}
            sym_ok &= all(em[(b, a)] == t for (a, b), t in em.items())
            rho_real.append(float(np.mean([t < 0 for (a, b), t in em.items() if a < b])))
            st_all, upd_all = [], []
            for r in (0, 1):
                df = pd.read_csv(os.path.join(stage3.OUT, "chains", pid, f"g{g}_r{r}.csv.gz"), dtype={"counts": str})
                st = np.load(os.path.join(stage3.OUT, "chains", pid, f"g{g}_r{r}_states.npy")).astype(np.int64)
                df["counts"] = df["counts"].str.zfill(4)
                # ---- B1 update scheme
                per_sweep = df.groupby("sweep").size()
                sweeps_ok &= (list(per_sweep.index) == list(range(stage3.N_WARM + stage3.N_MEAS))
                              and (per_sweep == stage3.N).all()
                              and all((grp.step.values == np.arange(stage3.N)).all() for _, grp in df.groupby("sweep")))
                agent_chi_p.append(stats.chisquare(np.bincount(df.agent, minlength=stage3.N)).pvalue)
                for _, grp in df.groupby("sweep"):
                    for n_ in np.bincount(grp.agent, minlength=stage3.N):
                        per_agent_sweep[min(int(n_), 3)] += 1
                init_z.append(st[0].sum() / math.sqrt(stage3.N))
                # ---- replay
                s = st[0].copy(); t_prev = -1.0
                for row in df.itertuples(index=False):
                    i = row.agent
                    c["rows"] += 1
                    if row.s0 != s[i]: c["B2_s0_mismatch"] += 1
                    seq0 = [("a" if tie[q] > 0 else "r") + ("P" if s[nb[q]] > 0 else "Q") for q in range(ip[i], ip[i + 1])]
                    counts = tuple(seq0.count(x) for x in battery.TYPES)
                    if "".join(map(str, counts)) != row.counts: c["B5_logged_counts_mismatch"] += 1
                    first_valid = None
                    for att in range(1, row.attempts + 1):
                        rec = recs.get(f"stage3:{pid}:{g}:{r}:{row.sweep}:{row.step}:{att}")
                        if rec is None:
                            c["missing_cache_record"] += 1; continue
                        c["calls"] += 1
                        if rec["t"] <= t_prev: c["B2_nonmonotone_time"] += 1
                        t_prev = rec["t"]
                        prompt = rec["body"]["messages"][0]["content"]
                        P_ = parse_prompt(prompt, framing)
                        if not P_["gloss_ok"]: c["B3_gloss_inconsistent"] += 1
                        # letter of position P: from the content gloss, or (neutral) from the own line + replayed state
                        lp_ = P_["letter_of_P"] or (P_["own_letter"] if s[i] > 0 else ("B" if P_["own_letter"] == "A" else "A"))
                        mapping_rec = 0 if lp_ == "A" else 1
                        pos = lambda letter: 1 if letter == lp_ else -1
                        # B4 own position (independent in political/workplace; consistency in neutral)
                        if pos(P_["own_letter"]) != s[i]: c["B4_own_mismatch"] += 1
                        # B2/B5 contacts shown vs replayed neighbour states and tie signs
                        shown = sorted(("a" if al else "r") + ("P" if pos(le) > 0 else "Q") for al, le in P_["contacts"])
                        if contacts:
                            if shown != sorted(seq0): c["B2B5_contacts_mismatch"] += 1
                        elif shown: c["nc_contacts_present"] += 1
                        # B6 template + body
                        try:
                            stage3.check_prompt(prompt, framing, counts, int(s[i]), mapping_rec, P_["order"], contacts)
                        except AssertionError:
                            c["B6_check_prompt_fail"] += 1
                        b = {k: v for k, v in rec["body"].items() if k != "messages"}
                        if json.dumps(b, sort_keys=True) != ref_sig: c["B6_body_differs"] += 1
                        bal[(mapping_rec, P_["order"], int(s[i]))] += 1
                        if contacts and att == 1:
                            listed = [("a" if al else "r") + ("P" if pos(le) > 0 else "Q") for al, le in P_["contacts"]]
                            for j in range(len(listed)):
                                order_pos[j, 0] += listed[j] == seq0[0]
                                q_ = seq0.count(seq0[0]) / len(seq0)
                                order_pos[j, 1] += q_; order_pos[j, 2] += q_ * (1 - q_)
                        # B7
                        prov[rec["resp"].get("provider")] += 1
                        # raw reply -> validity, pA / letter
                        if mode == "lp":
                            pA, leak = pA_from_raw(rec["resp"].get("first"))
                            valid = pA is not None and leak <= stage3.LEAK_MAX
                        else:
                            letter_raw = parse_sample(rec["resp"].get("content")); valid = letter_raw is not None
                        if first_valid is None:
                            first_valid = valid
                        if not valid:
                            inv_rows.append(dict(point=pid, s0=int(s[i]), mapping=mapping_rec, order=P_["order"],
                                                 counts=row.counts, content=rec["resp"].get("content"),
                                                 top=str((rec["resp"].get("first") or {}).get("top"))[:200]))
                        if att == row.attempts:          # the attempt whose reply was applied
                            if mapping_rec != row.mapping: c["B3_mapping_logged_vs_prompt"] += 1
                            if P_["order"] != row.order: c["B3_order_logged_vs_prompt"] += 1
                            if mode == "lp":
                                if valid and abs(pA - row.pA) > 1e-12: c["B3_pA_mismatch"] += 1
                                if valid:
                                    draw_z[0] += (row.letter == "A") - pA; draw_z[1] += pA * (1 - pA)
                            else:
                                if valid and letter_raw != row.letter: c["B3_letter_mismatch"] += 1
                            expect_new = pos(row.letter) if valid else int(s[i])
                            if expect_new != row.new: c["B3_applied_mismatch"] += 1
                            if valid and contacts:
                                ref_cell = (counts, int(s[i]), mapping_rec)
                                if framing == "neutral" and mapping_rec == 1:
                                    ref_cell = ((counts[1], counts[0], counts[3], counts[2]), -int(s[i]), 0)
                                kk = ("".join(map(str, ref_cell[0])), ref_cell[1], ref_cell[2], P_["order"])
                                if kk in battery_pA.index:
                                    bp = battery_pA.loc[kk]
                                    calib_rows.append(dict(point=pid, battery_pA=bp.pA, battery_leak=bp.leak_max,
                                                           live_A=int(row.letter == "A"), live_pA=pA if mode == "lp" else np.nan))
                                else:
                                    c["C_cell_not_in_battery"] += 1
                    if first_valid is not None and int(not first_valid) != row.invalid_first: c["B8_invalid_flag_mismatch"] += 1
                    s[i] = row.new
                    if row.step == stage3.N - 1 and not (s == st[row.sweep + 1]).all():
                        c["B2_sweep_state_mismatch"] += 1
                st_all.append(st); upd_all.append(df)
                # ---- exact replay of the chain from the cache through the unchanged harness (B3)
                def cache_only(body, rep, _recs=recs):
                    rec = _recs.get(rep)
                    if rec is None or rec["key"] != client.request_key(body, rep):
                        raise RuntimeError("cache miss in replay: " + rep)
                    return rec
                df2, st2 = stage3.chain(pid, g, r, call=cache_only)
                same = (st2.astype(np.int64) == st).all()
                cols = ["agent", "s0", "new", "attempts", "mapping", "order", "letter", "invalid_first"]
                same &= all((df2[k].values == df[k].values).all() for k in cols)
                same &= np.allclose(df2.pA.astype(float).values, df.pA.astype(float).values, equal_nan=True, rtol=0, atol=1e-12)
                if not same: c["B3_exact_replay_differs"] += 1
                c["chains_replayed"] += 1
            # ---- B10: simulator's own measurement code on the replayed trajectories
            ST = np.stack(st_all)
            AG = np.zeros((2, stage3.N_WARM + stage3.N_MEAS, stage3.N), np.int64); NEW = AG.copy()
            for r, df in enumerate(upd_all):
                AG[r, df.sweep.values, df.step.values] = df.agent.values; NEW[r, df.sweep.values, df.step.values] = df.new.values
            out = run_replay(ip, nb, tie, ST[:, 0], AG, NEW, stage3.N_WARM, stage3.N_MEAS)
            sim = dict(zip(S.MEASURES, out)); sim["excess_unsat"] = sim["unsatisfied"] - floor_for(rho)
            rep_row = reported[(reported.point == pid) & (reported.graph == g)].iloc[0]
            harness = stage3.graph_measures(ip, nb, tie, ST, pd.concat(upd_all))
            for m in S.MEASURES + ["excess_unsat"]:
                b10_rows.append(dict(point=pid, graph=g, measure=m, simulator_code=sim[m], reported=rep_row[m],
                                     harness_recomputed=harness.get(m, harness.get("unsatisfied") - floor_for(rho) if m == "excess_unsat" else np.nan)))
        # per-point summaries
        res["B1"][pid] = dict(sweeps_and_steps_complete=bool(sweeps_ok), min_agent_chi2_p=float(min(agent_chi_p)),
                              min_agent_chi2_p_x_chains=float(min(1, min(agent_chi_p) * len(agent_chi_p))),
                              updates_per_agent_per_sweep={str(k): v for k, v in sorted(per_agent_sweep.items())},
                              expected_binomial_100_001={str(k): float(stats.binom.pmf(k, 100, 0.01) if k < 3 else stats.binom.sf(2, 100, 0.01)) for k in range(4)})
        res["B2"][pid] = dict(calls=c["calls"], s0_mismatch=c["B2_s0_mismatch"], contacts_shown_vs_replayed_state_mismatch=c["B2B5_contacts_mismatch"],
                              sweep_state_mismatch=c["B2_sweep_state_mismatch"], nonmonotone_cache_time_within_chain=c["B2_nonmonotone_time"],
                              missing_cache_records=c["missing_cache_record"])
        res["B3"][pid] = dict(mapping_logged_vs_prompt=c["B3_mapping_logged_vs_prompt"], order_logged_vs_prompt=c["B3_order_logged_vs_prompt"],
                              gloss_inconsistent=c["B3_gloss_inconsistent"], pA_mismatch=c["B3_pA_mismatch"], letter_mismatch=c["B3_letter_mismatch"],
                              applied_position_mismatch=c["B3_applied_mismatch"], exact_replay_chains_differing=c["B3_exact_replay_differs"],
                              chains_replayed=c["chains_replayed"],
                              letter_draw_z=(draw_z[0] / math.sqrt(draw_z[1]) if draw_z[1] > 0 else None))
        res["B4"][pid] = dict(own_line_mismatch=c["B4_own_mismatch"], calls=c["calls"],
                              independent=framing != "neutral")
        res["B5"][pid] = dict(ties_symmetric=bool(sym_ok), rho_design=rho, rho_realised_per_graph=rho_real,
                              rho_in_binomial_99=[bool(stats.binom.ppf(0.005, 2 * stage3.N, rho) <= x * 2 * stage3.N <= stats.binom.ppf(0.995, 2 * stage3.N, rho)) for x in rho_real],
                              logged_counts_mismatch=c["B5_logged_counts_mismatch"], contacts_mismatch=c["B2B5_contacts_mismatch"],
                              nc_prompts_with_contacts=c["nc_contacts_present"])
        nmap = sum(v for (mp, o, s0), v in bal.items() if mp == 0); nord = sum(v for (mp, o, s0), v in bal.items() if o == 0); ntot = sum(bal.values())
        tab = lambda f: np.array([[sum(v for k, v in bal.items() if f(k) == (a, b)) for b in (-1, 1)] for a in (0, 1)])
        ct = lambda f: float(stats.chi2_contingency(tab(f))[1])
        res["B6"][pid] = dict(check_prompt_fail=c["B6_check_prompt_fail"], body_differs_from_battery_body=c["B6_body_differs"],
                              body_signature=json.loads(ref_sig), stage1_records_with_same_body_signature=int(sig1.get(ref_sig, 0)),
                              share_mapping0=nmap / ntot, p_mapping=float(stats.binomtest(nmap, ntot).pvalue),
                              share_order0=nord / ntot, p_order=float(stats.binomtest(nord, ntot).pvalue),
                              p_mapping_x_own=ct(lambda k: (k[0], k[2])), p_order_x_own=ct(lambda k: (k[1], k[2])),
                              contact_order_first_csr_type_by_listed_position=(
                                  [dict(listed_position=j, observed=int(order_pos[j, 0]), expected=round(order_pos[j, 1], 1),
                                        z=(order_pos[j, 0] - order_pos[j, 1]) / math.sqrt(max(order_pos[j, 2], 1e-9)))
                                   for j in range(4)] if contacts else None))
        res["B7"][pid] = dict(pinned_tag=tag, allow_fallbacks_false=c["B6_body_differs"] == 0, served=dict(prov),
                              stage1_main_served=dict(served1) if contacts else "see parent point")
        nI = len([x for x in inv_rows if x["point"] == pid])
        lg = pd.concat([pd.read_csv(os.path.join(stage3.OUT, "chains", pid, f"g{g}_r{r}.csv.gz")) for g in range(stage3.N_GRAPHS) for r in (0, 1)])
        res["B8"][pid] = dict(updates=len(lg), first_attempt_invalid=int(lg.invalid_first.sum()), rate=float(lg.invalid_first.mean()),
                              resampled_calls=int((lg.attempts - 1).sum()), kept_after_invalid=int(lg.kept_after_invalid.sum()),
                              invalid_calls_in_cache=nI, flag_mismatch=c["B8_invalid_flag_mismatch"],
                              invalid_examples=[x for x in inv_rows if x["point"] == pid][:5],
                              clustering="n/a (< 20 invalid)" if nI < 20 else "see invalid.csv")
        res["C"][pid] = dict(cells_not_in_battery=c["C_cell_not_in_battery"])
        all_inv += inv_rows
        print(pid, dict(c), flush=True)
    # B9
    z = np.array(init_z)
    res["B9"] = dict(chains=len(z), initial_m_z_mean=float(z.mean()), initial_m_z_var=float(z.var()),
                     chi2_sum_z2_p=float(2 * min(stats.chi2.sf((z ** 2).sum(), len(z)), stats.chi2.cdf((z ** 2).sum(), len(z)))),
                     harness=dict(N_WARM=stage3.N_WARM, N_MEAS=stage3.N_MEAS, init="independent 50/50 per agent (rng.random() < 0.5)",
                                  persistence_ref="states after sweep N_WARM-1 (st[:, N_WARM])"),
                     simulator=dict(stage3_jobs=sorted({(j[7], j[8]) for j in stage2.jobs_stage3(1)}), init="np.random.random() < 0.5 + init_bias, init_bias = 0",
                                    persistence_ref="copied when t == n_warm - 1"),
                     stage2_csv_n_warm_n_meas=sorted(map(list, pd.read_csv(os.path.join(ROOT, "results/llm/stage2/stage3.csv"),
                                                                            usecols=["n_warm", "n_meas"]).drop_duplicates().values.tolist())))
    # B10
    b10 = pd.DataFrame(b10_rows)
    b10["d_reported"] = (b10.simulator_code - b10.reported).abs(); b10["d_harness"] = (b10.simulator_code - b10.harness_recomputed).abs()
    b10.to_csv(os.path.join(OUT, "B10_measures_recomputed.csv"), index=False)
    pm = b10[b10.measure.isin(MEAS)].groupby(["point", "measure"]).simulator_code.mean().unstack()
    d_sum = float(np.nanmax(np.abs(pm.values - summary.loc[pm.index, pm.columns].values)))
    res["B10"] = dict(max_abs_diff_vs_measures_by_graph=float(b10.d_reported.max()),
                      max_abs_diff_vs_harness_graph_measures=float(b10.d_harness.max()),
                      max_abs_diff_point_means_vs_summary_csv=d_sum,
                      code="results/llm/stage3_audit/replay_measure_source.py (surrogate.run, dynamics replaced by replay)")
    # C
    cal = pd.DataFrame(calib_rows); cal.to_csv(os.path.join(OUT, "C_calls.csv.gz"), index=False)
    tabs = []
    for pid, g in cal.groupby("point"):
        g = g.copy(); g["bin"] = np.minimum((g.battery_pA * 10).astype(int), 9)
        t = g.groupby("bin").agg(n=("live_A", "size"), battery_pA=("battery_pA", "mean"), observed_A=("live_A", "mean"),
                                 live_pA=("live_pA", "mean")).reset_index(); t.insert(0, "point", pid); tabs.append(t)
        x = np.log(np.clip(g.battery_pA, 1e-4, 1 - 1e-4) / (1 - np.clip(g.battery_pA, 1e-4, 1 - 1e-4)))
        b, se = logit_fit(x.values, g.live_A.values)
        lp = g.dropna(subset=["live_pA"])
        rms = None
        if len(lp):
            cl = lambda p: np.clip(np.log(np.clip(p, 1e-12, 1) / np.clip(1 - p, 1e-12, 1)), -6, 6)
            rms = float(np.sqrt(np.mean((cl(lp.live_pA.values) - cl(lp.battery_pA.values)) ** 2)))
        res["C"][pid].update(n=len(g), mean_battery_pA=float(g.battery_pA.mean()), mean_observed_A=float(g.live_A.mean()),
                             calibration_in_the_large=float(g.live_A.mean() - g.battery_pA.mean()),
                             slope=float(b[1]), slope_ci=[float(b[1] - 1.96 * se[1]), float(b[1] + 1.96 * se[1])],
                             intercept=float(b[0]), rms_logit_live_vs_battery_pA=rms,
                             share_battery_cells_leak_gt_5pct=float((g.battery_leak > 0.05).mean()) if g.battery_leak.notna().any() else None)
    pd.concat(tabs).to_csv(os.path.join(OUT, "C_calibration_table.csv"), index=False)
    if all_inv:
        pd.DataFrame(all_inv).to_csv(os.path.join(OUT, "B8_invalid.csv"), index=False)
    save("logs_checks.json", res)
    print(json.dumps({k: v for k, v in res.items() if k in ("B9", "B10")}, indent=1, default=str))


def logit_fit(x, y, iters=50):
    """Logistic regression of y on [1, x] by Newton; returns (coef, se)."""
    X = np.column_stack([np.ones_like(x), x]); b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        H = X.T @ (X * W[:, None]); b = b + np.linalg.solve(H, X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ b)); H = X.T @ (X * (p * (1 - p))[:, None])
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


# ======================================================================= A: surrogate mock agent
RE_X = re.compile(r"exactly one letter, ([AB]) or")


def mock_agent(row, framing, mode, salt):
    """Answer with the fitted surrogate's P(choice | prompt), in letter space, reading only the prompt text."""
    Pc = (battery.FRAMINGS[framing]["content"] or {}).get("P")

    def call(body, rep):
        p = body["messages"][0]["content"]; lines = p.split("\n")
        X = RE_X.search(lines[-1]).group(1)
        own = RE_OWN.search(p).group(1)
        cons = [(m.group(1) == "An ally", m.group(2)) for m in (RE_CON.fullmatch(l) for l in lines) if m]
        k = len(cons)
        ma = sum(1 if le == "A" else -1 for al, le in cons if al)
        mr = sum(1 if le == "A" else -1 for al, le in cons if not al)
        cA = 0.0 if Pc is None else (1.0 if f"A ({Pc})" in lines[0] else -1.0)
        eta = ((k ** -row[S.ALPHA]) * (row[S.B_A] * ma - row[S.B_R] * mr) if k else 0.0) \
            + row[S.GAMMA] * (1 if own == "A" else -1) + row[S.H_C] * cA + row[S.H_L] + row[S.H_O] * (1 if X == "A" else -1)
        pa = 1 / (1 + math.exp(-eta))
        if k and row[S.P_VOTER] > 0:            # MF: follow the first-listed contact with probability p
            al, le = cons[0]
            fA = 1.0 if (le == "A") == al else 0.0
            pa = row[S.P_VOTER] * fA + (1 - row[S.P_VOTER]) * pa
        if mode == "lp":
            pa = min(max(pa, 1e-15), 1 - 1e-15)
            return {"resp": {"first": {"top": [["A", math.log(pa)], ["B", math.log(1 - pa)]]}, "usage": {"cost": 0.0}}}
        u = random.Random(stage3._seed("audit-mock-sample", salt, rep)).random()
        return {"resp": {"content": "A" if u < pa else "B", "usage": {"cost": 0.0}}}
    return call


def mock_task(job):
    rep, pid = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    key, framing, rho, contacts = stage3.POINTS[pid]
    mode = E[key][2]
    fit = stage2.load_fit(key, framing)
    orig = stage3._seed
    if rep > 0:                                   # fresh chain seeds; graphs unchanged ("stage3-graph" seeds untouched)
        stage3._seed = lambda *parts: orig(*parts, "audit-rep", rep) if parts[0] == "stage3-chain" else orig(*parts)
    try:
        per = []
        for g in range(stage3.N_GRAPHS):
            d = int(np.random.default_rng(orig("audit-draw", rep, pid.split("-")[0], g)).integers(len(fit["draws"])))
            row = stage2.param_row(fit, framing, d, "fitted")
            ch = [stage3.chain(pid, g, r, call=mock_agent(row, framing, mode, (rep, pid, g, r))) for r in (0, 1)]
            st = np.stack([x[1] for x in ch]).astype(np.int64); upd = pd.concat([x[0] for x in ch])
            m = stage3.graph_measures(*stage3.graph(pid, g), st, upd)
            m["excess_unsat"] = m["unsatisfied"] - floor_for(rho)
            per.append(dict(graph=g, draw=d, invalid=int(upd.invalid_first.sum()), **{k: float(m[k]) for k in MEAS}))
    finally:
        stage3._seed = orig
    return dict(rep=rep, point=pid, per_graph=per, mean5={k: float(np.mean([x[k] for x in per])) for k in MEAS})


def audit_mock(reps, workers, points=None, first=0):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "A_mock_runs.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            j = json.loads(l); done.add((j["rep"], j["point"]))
    jobs = [(r, p) for r in range(first, reps + 1) for p in (points or POINTS) if (r, p) not in done]
    print(f"mock: {len(jobs)} tasks to run ({len(done)} done earlier)", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(mock_task, j) for j in jobs]
        for f in as_completed(futs):
            j = f.result()
            with open(path, "a") as fo:
                fo.write(json.dumps(j) + "\n")
            print(f"  rep {j['rep']:2d} {j['point']:6s} " + " ".join(f"{k} {v:.3f}" for k, v in j["mean5"].items()), flush=True)
    summarise_mock()


def predictions():
    pred = {p["id"].split(" ")[0]: (p, "fitted") for p in json.load(open(os.path.join(ROOT, "results/llm/stage2/stage3_predictions.json")))}
    for p in json.load(open(os.path.join(ROOT, "results/llm/stage2/stage3_predictions_nocontacts.json"))):
        pred[p["id"]] = (p, "predicted")
    return pred


def summarise_mock():
    pred = predictions()
    runs = [json.loads(l) for l in open(os.path.join(OUT, "A_mock_runs.jsonl"))]
    out = {}
    for pid in POINTS:
        p, mode = pred[pid]
        R = [x for x in runs if x["point"] == pid]
        r0 = [x for x in R if x["rep"] == 0]; rr = [x for x in R if 0 < x["rep"] <= 40]     # pre-stated rule: reps 1..40
        meas = {}
        for m in MEAS:
            lo, hi = p[f"{mode}_{m}_mean5_lo"], p[f"{mode}_{m}_mean5_hi"]
            vals = [x["mean5"][m] for x in rr]
            meas[m] = dict(primary=m in p["primary"], interval=[lo, hi],
                           live_seeds_run=r0[0]["mean5"][m] if r0 else None,
                           live_seeds_inside=bool(lo <= r0[0]["mean5"][m] <= hi) if r0 else None,
                           reps=len(vals), inside_reps=int(sum(lo <= v <= hi for v in vals)),
                           reps_mean=float(np.mean(vals)) if vals else None,
                           reps_5_95=[float(np.percentile(vals, 5)), float(np.percentile(vals, 95))] if vals else None)
        prim = [meas[m] for m in p["primary"]]
        out[pid] = dict(measures=meas, invalid_total=sum(g["invalid"] for x in R for g in x["per_graph"]),
                        passes=all(x["live_seeds_inside"] for x in prim) and all(x["inside_reps"] >= 30 * x["reps"] / 40 for x in prim))
    save("A_mock_summary.json", out)
    for pid, v in out.items():
        print(pid, "PASS" if v["passes"] else "FAIL", {m: (round(x["live_seeds_run"], 3), f"{x['inside_reps']}/{x['reps']}", [round(y, 2) for y in x["interval"]])
                                                       for m, x in v["measures"].items() if x["primary"]})
    return out


# ======================================================================= A, post-hoc diagnostic (not in the plan)
def sim_same_graphs(reps=400):
    """Post-hoc diagnostic for A: the simulator itself (surrogate.run) on the SAME 5 live graphs with the SAME
    per-graph parameter draws as the mock repetitions (draw scheme of mock_task). If the mock (harness) and the
    simulator agree here, a coverage shortfall of the mock against the committed intervals comes from conditioning on
    these 5 graphs / the 40-run interval estimate, not from the harness."""
    pred = predictions()
    mock = [json.loads(l) for l in open(os.path.join(OUT, "A_mock_runs.jsonl"))]
    out, rows = {}, []
    for pid in POINTS:
        key, framing, rho, contacts = stage3.POINTS[pid]
        fit = stage2.load_fit(key, framing)
        mode = "fitted" if contacts else "fields_only"
        G = [stage3.graph(pid, g) for g in range(stage3.N_GRAPHS)]
        sim = []
        for rep in range(1, reps + 1):
            per = []
            for g in range(stage3.N_GRAPHS):
                d = int(np.random.default_rng(stage3._seed("audit-draw", rep, pid.split("-")[0], g)).integers(len(fit["draws"])))
                row = stage2.param_row(fit, framing, d, mode)
                r = dict(zip(S.MEASURES, S.run(*G[g], np.zeros(stage3.N, np.int64), row[None, :], stage3.N_WARM, stage3.N_MEAS,
                                               int(stage3._seed("audit-sim", rep, pid, g) % 2**31), False, 0.0)))
                r["excess_unsat"] = r["unsatisfied"] - floor_for(rho); per.append(r)
            sim.append({m: float(np.mean([x[m] for x in per])) for m in MEAS})
        mk = [x["mean5"] for x in sorted(mock, key=lambda x: x["rep"]) if x["point"] == pid and x["rep"] > 0]   # reps 1..n, same draws as sim reps 1..n
        p, pm = pred[pid]
        res = {}
        for m in MEAS:
            a = np.array([x[m] for x in mk]); b = np.array([x[m] for x in sim]); b40 = b[:len(a)]
            lo, hi = p[f"{pm}_{m}_mean5_lo"], p[f"{pm}_{m}_mean5_hi"]
            se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            res[m] = dict(primary=m in p["primary"], interval=[lo, hi], mock_mean=float(a.mean()), sim_mean=float(b.mean()),
                          mock_sd=float(a.std(ddof=1)), sim_sd=float(b.std(ddof=1)), z_mock_minus_sim=float((a.mean() - b.mean()) / se) if se > 0 else 0.0,
                          paired_mean_diff=float((a - b40).mean()),
                          paired_z=float((a - b40).mean() / ((a - b40).std(ddof=1) / math.sqrt(len(a)))),
                          mock_inside=f"{int(((a >= lo) & (a <= hi)).sum())}/{len(a)}", sim_inside=f"{int(((b >= lo) & (b <= hi)).sum())}/{len(b)}",
                          sim_inside_share=float(((b >= lo) & (b <= hi)).mean()))
            rows.append(dict(point=pid, measure=m, **{k: v for k, v in res[m].items() if k != "interval"}, lo=lo, hi=hi))
        out[pid] = res
        print(pid, {m: (round(v["mock_mean"], 3), round(v["sim_mean"], 3), round(v["z_mock_minus_sim"], 2), v["mock_inside"], v["sim_inside"])
                    for m, v in res.items() if v["primary"]}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "A_diag_sim_same_graphs.csv"), index=False)
    save("A_diag_sim_same_graphs.json", out)


# ======================================================================= B11
def audit_surrogate():
    pred = predictions()
    S3 = pd.read_csv(os.path.join(ROOT, "results/llm/stage2/stage3.csv"))
    summ = pd.read_csv(os.path.join(ROOT, "results/llm/stage2/summary_stage3.csv"))
    import stage2_report as R
    resum = R.summarise(R.add_floor(S3.copy()))
    out = {}
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    pred_commit = git("log", "-1", "--format=%h", "--", "results/llm/stage2/stage3_predictions.json")
    for pid in POINTS:
        key, framing, rho, contacts = stage3.POINTS[pid]
        p, pmode = pred[pid]
        mode = "fitted" if contacts else "fields_only"
        sel = (S3["pop"] == key) & (S3.framing == framing) & (S3.rho == rho) & (S3.graph == "rrg") & (S3.eps == 0) & (S3["mode"] == mode)
        runs = S3[sel]
        q = lambda D: D[(D["pop"] == key) & (D.framing == framing) & (D.rho == rho) & (D.graph == "rrg") & (D.eps == 0) & (D["mode"] == mode)].iloc[0]
        a, b = q(summ), q(resum)
        dmax = max(max(abs(a[m + s] - b[m + s]), abs(a[m + s] - p[f"{pmode}_{m}_{t}"]))
                   for m in MEAS for s, t in (("_mean5_lo", "mean5_lo"), ("_mean5_hi", "mean5_hi")))
        fit = stage2.load_fit(key, framing)
        row = stage2.param_row(fit, framing, None, mode)
        fit_file = f"results/llm/fits/surrogate/{key}.json"
        out[pid] = dict(endpoint=key, framing_arm=stage2.FRAME_ARM[framing], rho=rho, mode=mode, runs=int(len(runs)),
                        runs_N=sorted(runs.N.unique().tolist()), runs_n_warm_meas=sorted({(int(x), int(y)) for x, y in zip(runs.n_warm, runs.n_meas)}),
                        explicit_jitter=sorted(runs.explicit_jitter.unique().tolist()),
                        surrogate_choice=fit["choice"], params_point_estimate=dict(zip(["p_voter", "alpha", "b_a", "b_r", "gamma", "h_C", "h_L", "h_O"], map(float, row[:8]))),
                        neutral_h_applied_as_letter_field=bool((framing != "neutral") or (row[S.H_C] == 0 and row[S.H_L] == fit["est"][fit["names"].index("h")])),
                        fit_file_changed_since_predictions_commit=bool(git("diff", "--stat", pred_commit, "HEAD", "--", fit_file)),
                        stage3_csv_changed_since_predictions_commit=bool(git("diff", "--stat", pred_commit, "HEAD", "--", "results/llm/stage2/stage3.csv")),
                        max_abs_diff_interval_recomputed_vs_committed=float(dmax))
    src = inspect.getsource(S.run.py_func)
    common = dict(predictions_commit=pred_commit,
                  per_call_L_O_redrawn=("L = 1.0 if np.random.random() < 0.5 else -1.0" in src and "O = 1.0 if np.random.random() < 0.5 else -1.0" in src),
                  nocontacts_prediction="fields_only rows (social terms and voter step zeroed) = the k = 0 update")
    save("B11_surrogate.json", dict(points=out, common=common))
    print(json.dumps(dict(points=out, common=common), indent=1, default=str))


# ======================================================================= report
def report():
    L = json.load(open(os.path.join(OUT, "logs_checks.json")))
    A = json.load(open(os.path.join(OUT, "A_mock_summary.json")))
    B11 = json.load(open(os.path.join(OUT, "B11_surrogate.json")))
    rows = []
    def add(chk, ok, ev):
        rows.append((chk, "PASS" if ok else "FAIL", ev))
    for pid in POINTS:
        a = A[pid]; ev = "; ".join(f"{m}: live-seeds {x['live_seeds_run']:.3f} in [{x['interval'][0]:.2f}, {x['interval'][1]:.2f}] "
                                   f"{'yes' if x['live_seeds_inside'] else 'NO'}, reps inside {x['inside_reps']}/{x['reps']}"
                                   for m, x in a["measures"].items() if x["primary"])
        add(f"A {pid}", a["passes"], ev)
    B = {k: L[k] for k in L}
    add("B1", all(v["sweeps_and_steps_complete"] and v["min_agent_chi2_p_x_chains"] > 0.001 for v in B["B1"].values()),
        "all chains 20 sweeps x 100 updates; agent draw χ² min p (Bonferroni) " +
        ", ".join(f"{p} {v['min_agent_chi2_p_x_chains']:.2f}" for p, v in B["B1"].items()))
    add("B2", all(v["s0_mismatch"] == v["contacts_shown_vs_replayed_state_mismatch"] == v["sweep_state_mismatch"] == v["nonmonotone_cache_time_within_chain"] == v["missing_cache_records"] == 0 for v in B["B2"].values()),
        f"{sum(v['calls'] for v in B['B2'].values())} calls; stale/mismatched neighbour states " +
        str(sum(v["contacts_shown_vs_replayed_state_mismatch"] for v in B["B2"].values())))
    k3 = ["mapping_logged_vs_prompt", "order_logged_vs_prompt", "gloss_inconsistent", "pA_mismatch", "letter_mismatch", "applied_position_mismatch", "exact_replay_chains_differing"]
    add("B3", all(v[k] == 0 for v in B["B3"].values() for k in k3),
        "mismatches " + str({k: sum(v[k] for v in B["B3"].values()) for k in k3}) + "; chains replayed " + str(sum(v["chains_replayed"] for v in B["B3"].values())))
    add("B4", all(v["own_line_mismatch"] == 0 for v in B["B4"].values()), "own-line mismatches " + str(sum(v["own_line_mismatch"] for v in B["B4"].values())))
    add("B5", all(v["ties_symmetric"] and all(v["rho_in_binomial_99"]) and v["logged_counts_mismatch"] == 0 and v["contacts_mismatch"] == 0 and v["nc_prompts_with_contacts"] == 0 for v in B["B5"].values()),
        "; ".join(f"{p} ρ={v['rho_design']} realised {min(v['rho_realised_per_graph']):.3f}–{max(v['rho_realised_per_graph']):.3f}" for p, v in B["B5"].items()))
    add("B6", all(v["check_prompt_fail"] == 0 and v["body_differs_from_battery_body"] == 0 and v["stage1_records_with_same_body_signature"] > 0
                  and min(v["p_mapping"], v["p_order"], v["p_mapping_x_own"], v["p_order_x_own"]) > 0.001
                  and (v["contact_order_first_csr_type_by_listed_position"] is None or max(abs(x["z"]) for x in v["contact_order_first_csr_type_by_listed_position"]) < 3.3)
                  for v in B["B6"].values()),
        "; ".join(f"{p} map0 {v['share_mapping0']:.3f} ord0 {v['share_order0']:.3f}" for p, v in B["B6"].items()))
    add("B7", all(len(v["served"]) == 1 and v["allow_fallbacks_false"] for v in B["B7"].values()),
        "; ".join(f"{p} {v['served']}" for p, v in B["B7"].items()))
    add("B8", all(v["rate"] <= 0.02 and v["flag_mismatch"] == 0 for v in B["B8"].values()),
        "; ".join(f"{p} {v['first_attempt_invalid']}/{v['updates']} kept {v['kept_after_invalid']}" for p, v in B["B8"].items()))
    b9 = B["B9"]
    add("B9", b9["simulator"]["stage3_jobs"] == [[10, 10]] and b9["stage2_csv_n_warm_n_meas"] == [[10, 10]] and b9["chi2_sum_z2_p"] > 0.001,
        f"warm-up/measured 10/10 both; initial m z: mean {b9['initial_m_z_mean']:.2f}, var {b9['initial_m_z_var']:.2f} over {b9['chains']} chains")
    b10 = B["B10"]
    add("B10", max(b10["max_abs_diff_vs_measures_by_graph"], b10["max_abs_diff_point_means_vs_summary_csv"]) < 1e-9,
        f"max |Δ| vs measures_by_graph {b10['max_abs_diff_vs_measures_by_graph']:.1e}, vs summary.csv {b10['max_abs_diff_point_means_vs_summary_csv']:.1e}")
    ok11 = all((not v["fit_file_changed_since_predictions_commit"]) and v["max_abs_diff_interval_recomputed_vs_committed"] < 1e-9 and v["runs_n_warm_meas"] == [[10, 10]]
               and v["runs_N"] == [100] and v["explicit_jitter"] == [False] and v["neutral_h_applied_as_letter_field"] for v in B11["points"].values()) and B11["common"]["per_call_L_O_redrawn"]
    add("B11", ok11, "; ".join(f"{p} {v['endpoint']}/{v['framing_arm']}/ρ{v['rho']}/{v['mode']} {v['runs']} runs" for p, v in B11["points"].items()))
    for pid, v in B["C"].items():
        if "n" not in v:
            add(f"C {pid}", True, "no battery cell (no contacts): n/a"); continue
        flag = abs(v["calibration_in_the_large"]) > 0.03 or v["slope_ci"][1] < 0.8 or v["slope_ci"][0] > 1.2
        add(f"C {pid}", not flag, f"n {v['n']}; observed A {v['mean_observed_A']:.3f} vs battery {v['mean_battery_pA']:.3f} "
            f"(Δ {v['calibration_in_the_large']:+.3f}); slope {v['slope']:.2f} [{v['slope_ci'][0]:.2f}, {v['slope_ci'][1]:.2f}]"
            + (f"; RMS logit live vs battery pA {v['rms_logit_live_vs_battery_pA']:.2f}" if v["rms_logit_live_vs_battery_pA"] is not None else ""))
    md = ["| check | result | evidence |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    open(os.path.join(OUT, "audit_table.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["logs", "mock", "mock-summary", "sim-diag", "surrogate", "report"])
    ap.add_argument("--reps", type=int, default=40); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--points", nargs="+"); ap.add_argument("--first", type=int, default=0)
    a = ap.parse_args()
    {"logs": audit_logs, "mock": lambda: audit_mock(a.reps, a.workers, a.points, a.first), "mock-summary": summarise_mock,
     "sim-diag": sim_same_graphs, "surrogate": audit_surrogate, "report": report}[a.what]()
