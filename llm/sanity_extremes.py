"""Sanity check on unambiguous prompts (exploratory): neutral_own, mapping 0, A listed first.
Cases: all-ally unanimous for own position / against own position; all-rival likewise. k = 1, 3, 6.
Reports P(A) per endpoint. Output: results/llm/catalog/sanity_extremes_<date>.csv"""
import csv, datetime, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client, battery
from endpoints import E
from run_stage1 import parse_lp
cases = []
for k in (1, 3, 6):
    for name, counts in [("allies hold A", (k, 0, 0, 0)), ("allies hold B", (0, k, 0, 0)),
                         ("rivals hold A", (0, 0, k, 0)), ("rivals hold B", (0, 0, 0, k))]:
        for s0 in (+1, -1):
            cases.append((k, name, s0, counts))
keys = sys.argv[1:] or ["llama-8b", "llama-70b", "gpt4o-mini", "qwen-122b"]
def one(job):
    key, (k, name, s0, counts) = job
    model, tag, _, roff, _ = E[key]
    text, _ = battery.render("neutral", counts, s0, 0, 0, 0)
    r = client.chat(battery.body(model, tag, text, "lp", roff), rep=600)["resp"]
    return {"key": key, "k": k, "case": name, "own": "A" if s0 > 0 else "B", **parse_lp(r["first"])}
with ThreadPoolExecutor(8) as ex: rows = list(ex.map(one, [(kk, c) for kk in keys for c in cases]))
with open(os.path.join(client.ROOT, "results/llm/catalog", f"sanity_extremes_{datetime.date.today()}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import pandas as pd
d = pd.DataFrame(rows)
print(d.pivot_table(index=["case", "own"], columns=["key", "k"], values="pA").round(2).to_string())
print("spend", client.spend())
