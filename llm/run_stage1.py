"""Run (or re-materialise from cache) the Stage 1 battery for one endpoint.

  python ../llm/run_stage1.py --key llama-8b --phase pilot
  python ../llm/run_stage1.py --key llama-8b --phase main --arms neutral_own political
  python ../llm/run_stage1.py --key haiku --phase main --samples 30 --kmax 4 --arms neutral_own political

Writes results/llm/stage1/<phase>/<key>.csv, one row per call. Every call is cached by client.py, so
re-running is free. Pilot uses replicate indices 100+ so pilot responses can never be reused by main runs.
"""
import argparse, csv, math, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client, battery
from endpoints import E

OUT = os.path.join(client.ROOT, "results", "llm", "stage1")
LETTERS = ("A", "B")


def parse_lp(first):
    """Renormalised P(A) over {A,B} from first-token top-5 (PREREG §4)."""
    if not first or not first.get("top"):
        return dict(pA=None, leak=None, censored=None)
    top = first["top"]
    pa = sum(math.exp(l) for t, l in top if t is not None and t.strip() == "A")
    pb = sum(math.exp(l) for t, l in top if t is not None and t.strip() == "B")
    leak = max(0.0, 1.0 - pa - pb)
    cens = 0
    floor = math.exp(min(l for _, l in top))
    if pa == 0: pa, cens = floor, 1
    if pb == 0: pb, cens = floor, 1
    return dict(pA=pa / (pa + pb), leak=leak, censored=cens)


def parse_sample(content):
    if content is None:
        return None
    s = content.strip().strip("*").strip().strip("'\"`").strip()
    if s.endswith("."):
        s = s[:-1]
    if s in LETTERS:
        return s
    if len(s) > 1 and s[0] in LETTERS and not s[1].isalpha():
        return s[0]
    return None


def pilot_filter():
    rng = random.Random("pilot-v1")
    by_k = {}
    for m in battery.multisets():
        by_k.setdefault(sum(m), []).append(m)
    pick = set()
    for k, want in zip(range(1, 7), (4, 4, 4, 4, 4, 5)):
        pick |= set(rng.sample(by_k[k], want))
    return lambda counts, s0: counts in pick


def kmax_filter(kmax, base=None):
    return lambda c, s0: sum(c) <= kmax and (base is None or base(c, s0))


def run(key, phase, arms, reps, samples=0, temperature=1.0, kmax=6, workers=16, budget=None, order_subset=True,
        cell_filter=None, tag_override=None, out_name=None):
    model, tag, mode, roff, _ = E[key]
    tag = tag_override or tag
    if samples:
        mode = "sample"
    filt = cell_filter
    if phase == "pilot":
        filt = pilot_filter()
    if kmax < 6:
        filt = kmax_filter(kmax, filt)
    jobs = []
    for arm in arms:
        its = list(battery.items(arm, filt))
        if order_subset and arm == "neutral_own" and not samples:
            its += list(battery.items(arm, filt, extra_perms=True))
        for it in its:
            b = battery.body(model, tag, it["prompt"], mode, roff, temperature)
            for r in (range(samples) if samples else reps):
                rr = r + (100 if phase == "pilot" else 0) + (1000 if samples else 0)
                jobs.append((it, b, rr))
    n_cached = sum(1 for _, b, r in jobs if client.cached(b, r) is not None)
    print(f"{key} [{model} @ {tag}] {phase}: {len(jobs)} calls ({n_cached} cached)", flush=True)
    rows, errs, t0 = [], 0, time.time()

    def one(job):
        it, b, r = job
        rec = client.chat(b, rep=r, budget_usd=budget)
        resp = rec["resp"]
        row = {k: v for k, v in it.items() if k != "prompt"}
        row.update(counts="".join(map(str, it["counts"])), key=key, model=model, tag=tag, rep=r,
                   temperature=temperature, served=resp.get("provider"), content=resp.get("content"),
                   cost=(resp.get("usage") or {}).get("cost"))
        if mode == "lp":
            row.update(parse_lp(resp.get("first")))
        else:
            ans = parse_sample(resp.get("content"))
            row.update(answer=ans, pA=None if ans is None else float(ans == "A"), leak=float(ans is None), censored=0)
        return row

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            try:
                rows.append(f.result())
            except client.BudgetExceeded as e:
                print("BUDGET STOP:", e); ex.shutdown(cancel_futures=True); break
            except Exception as e:
                errs += 1
                if errs <= 5: print("error:", str(e)[:200], flush=True)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(jobs)}  {time.time()-t0:.0f}s  spend ${client.spend()['usd']:.4f}", flush=True)
    os.makedirs(os.path.join(OUT, phase), exist_ok=True)
    path = os.path.join(OUT, phase, (out_name or key) + ".csv")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    rows.sort(key=lambda r: (r["arm"], r["counts"], r["s0"], r["mapping"], r["order"], r["perm"], r["rep"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {path}: {len(rows)} rows, {errs} errors, new spend ${client.spend()['usd']:.4f}", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--phase", choices=["pilot", "main"], required=True)
    ap.add_argument("--arms", nargs="+", default=list(battery.ARMS))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--budget", type=float, default=None, help="stop if session spend exceeds this (USD)")
    ap.add_argument("--tag", default=None, help="override provider tag (backup endpoint)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.key, a.phase, a.arms, range(a.reps), a.samples, a.temperature, a.kmax, a.workers, a.budget,
        tag_override=a.tag, out_name=a.out)
