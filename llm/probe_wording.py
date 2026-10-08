"""Exploratory: is Llama-8B's 'switch away from own position' a wording artifact? Same 24 unambiguous neutral
cases as sanity_extremes.py, four question wordings, on llama-8b and llama-70b.
Output: results/llm/catalog/wording_<date>.csv"""
import csv, datetime, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client, battery
from endpoints import E
from run_stage1 import parse_lp
TAIL = "Reply with exactly one letter, A or B, and nothing else."
WORDINGS = {"v2 take now": "Which position do you take now? " + TAIL,
            "hold now": "Which position do you hold now? " + TAIL,
            "choose after considering": "Having considered your contacts, which position do you choose? " + TAIL,
            "stay or switch": "You may keep your current position or switch. Which position do you hold after deciding? " + TAIL}
cases = [(k, n, s0, c) for k in (1, 3, 6) for n, c in [("allies A", (k, 0, 0, 0)), ("allies B", (0, k, 0, 0)), ("rivals A", (0, 0, k, 0)), ("rivals B", (0, 0, 0, k))] for s0 in (1, -1)]
def one(job):
    key, wname, (k, n, s0, c) = job
    model, tag, _, roff, _ = E[key]
    text, _ = battery.render("neutral", c, s0, 0, 0, 0)
    text = text.rsplit("\n", 1)[0] + "\n" + WORDINGS[wname]
    r = client.chat(battery.body(model, tag, text, "lp", roff), rep=700)["resp"]
    p = parse_lp(r["first"])
    # 'agree' = P(choose the position the contacts push toward): allies -> their letter, rivals -> the other letter
    push_A = (n == "allies A") or (n == "rivals B")
    return {"key": key, "wording": wname, "k": k, "case": n, "own": "A" if s0 > 0 else "B", "pA": p["pA"], "leak": p["leak"],
            "p_follow_push": p["pA"] if push_A else 1 - p["pA"], "p_keep_own": p["pA"] if s0 > 0 else 1 - p["pA"]}
jobs = [(k, w, c) for k in ("llama-8b", "llama-70b") for w in WORDINGS for c in cases]
with ThreadPoolExecutor(8) as ex: rows = list(ex.map(one, jobs))
with open(os.path.join(client.ROOT, "results/llm/catalog", f"wording_{datetime.date.today()}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import pandas as pd
d = pd.DataFrame(rows)
d["aligned"] = ((d.case.isin(["allies A", "rivals B"])) == (d.own == "A"))   # push agrees with own position
print(d.groupby(["key", "wording", "aligned"])[["p_follow_push", "p_keep_own", "leak"]].mean().round(2).to_string())
print("spend", client.spend())
