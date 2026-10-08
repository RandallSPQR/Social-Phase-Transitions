"""Leak check of instruction v1 vs v2 across k, on 4 endpoints (neutral_own + political, 2 cells per k).
Output: results/llm/catalog/instruction_leak_<date>.csv"""
import csv, datetime, os, random, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client, battery
from endpoints import E
from run_stage1 import parse_lp
rng = random.Random("instr"); by_k = {}
for m in battery.multisets(): by_k.setdefault(sum(m), []).append(m)
pick = {m for k in by_k for m in rng.sample(by_k[k], 2)}
jobs = []
for key in ("llama-8b", "mistral-large", "qwen-9b", "gemma-31b"):
    model, tag, _, roff, _ = E[key]
    for v in (1, 2):
        for arm in ("neutral_own", "political"):
            for it in battery.items(arm, lambda c, s0: c in pick and s0 == 1, instruction=v):
                if it["order"] == 1 or it["mapping"] == 1: continue
                jobs.append((key, v, it, battery.body(model, tag, it["prompt"], "lp", roff)))
def one(j):
    key, v, it, b = j
    r = client.chat(b, rep=500)["resp"]
    return {"key": key, "instruction": v, "arm": it["arm"], "k": sum(it["counts"]), **parse_lp(r["first"]),
            "top1": (r["first"] or {}).get("top", [[None]])[0][0]}
with ThreadPoolExecutor(8) as ex: rows = list(ex.map(one, jobs))
with open(os.path.join(client.ROOT, "results/llm/catalog", f"instruction_leak_{datetime.date.today()}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import pandas as pd
d = pd.DataFrame(rows)
print(d.pivot_table(index=["key", "instruction"], columns="k", values="leak", aggfunc="mean").round(3))
print("spend", client.spend())
