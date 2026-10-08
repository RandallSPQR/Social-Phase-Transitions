"""Verify, per (model, provider) endpoint, that a pinned 1-token request actually
returns first-token top_logprobs, whether the reply is a bare A/B letter, whether
the model emits reasoning, and whether two identical requests give identical
logprobs. Writes results/llm/catalog/probe_<date>.csv. Run from sim/ or anywhere.
"""
import csv, datetime, json, math, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client

ROOT = client.ROOT
PROMPT = ("You currently hold position A. Your three contacts currently hold these positions:\n"
          "- an ally who agrees with you on most things: B\n- a rival who usually opposes you: A\n"
          "- an ally who agrees with you on most things: B\n\n"
          "Which position do you take now? Reply with exactly one letter, A or B, and nothing else.")

CANDS = {  # model -> providers whose metadata claims logprobs (plus frontier spot-check endpoints)
    "meta-llama/llama-3.2-3b-instruct": ["parasail/bf16"],
    "meta-llama/llama-3.1-8b-instruct": ["novita/fp8", "coreweave/bf16"],
    "meta-llama/llama-3.3-70b-instruct": ["novita/bf16", "parasail/fp8", "akashml/fp8", "coreweave/fp16"],
    "qwen/qwen3-30b-a3b-instruct-2507": ["streamlake", "alibaba", "dekallm"],
    "qwen/qwen3-235b-a22b-2507": ["parasail/fp8", "alibaba", "gmicloud/fp8"],
    "qwen/qwen3.5-9b": ["parasail/bf16", "venice/fp8"],
    "qwen/qwen3.5-122b-a10b": None,
    "qwen/qwen3.5-27b": None,
    "google/gemma-3-27b-it": ["parasail/fp8"],
    "google/gemma-4-26b-a4b-it": ["parasail/bf16", "novita/bf16", "coreweave/bf16"],
    "google/gemma-4-31b-it": ["parasail/fp8", "coreweave/fp4"],
    "mistralai/mistral-nemo": ["parasail/fp8", "io-net/fp16", "dekallm/fp8"],
    "mistralai/mistral-small-3.2-24b-instruct": ["parasail/bf16"],
    "mistralai/mistral-large-4-0": ["mistral"],
    "anthropic/claude-haiku-5.5": ["anthropic"],
    "openai/gpt-6-luna": ["openai"],
    "openai/gpt-4o-mini": ["openai"],
    "google/gemini-3.1-flash-lite": ["google-ai-studio"],
    "google/gemini-2.5-flash-lite": ["google-ai-studio"],
}


def endpoints_with_lp(model):
    import requests
    r = requests.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints").json()
    return [e["tag"] for e in r.get("data", {}).get("endpoints", []) if "logprobs" in e.get("supported_parameters", [])]


def body(model, tag, lp=True):
    b = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
         "max_tokens": 1 if lp else 8, "temperature": 1.0,
         "provider": {"order": [tag], "allow_fallbacks": False, "require_parameters": lp}}
    if lp:
        b.update({"logprobs": True, "top_logprobs": 20})
    return b


def probe(model, tag):
    lp_claimed = tag is not None
    row = {"model": model, "provider_tag": tag}
    try:
        b = body(model, tag, lp=True)
        r1 = client.chat(b, rep=0)["resp"]; r2 = client.chat(b, rep=1)["resp"]
        f1, f2 = r1["first"], r2["first"]
        row.update(provider_served=r1["provider"], content=repr(r1["content"]),
                   reasoning=bool(r1["reasoning"]), logprobs=bool(f1 and f1["top"]))
        if f1 and f1["top"]:
            top = dict((t, l) for t, l in f1["top"])
            pa = sum(math.exp(l) for t, l in top.items() if t.strip().strip("*") == "A")
            pb = sum(math.exp(l) for t, l in top.items() if t.strip().strip("*") == "B")
            row.update(n_top=len(f1["top"]), pA=round(pa, 5), pB=round(pb, 5), mass_AB=round(pa + pb, 5),
                       top3=" | ".join(f"{t!r}:{l:.3f}" for t, l in f1["top"][:3]))
            if f2 and f2["top"]:
                t2 = dict((t, l) for t, l in f2["top"])
                common = set(top) & set(t2)
                row["repeat_maxdiff"] = round(max(abs(top[t] - t2[t]) for t in common), 5) if common else None
        u = r1.get("usage") or {}
        row.update(prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                   reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                   cost=u.get("cost"))
    except Exception as e:
        row["error"] = str(e)[:240]
    return row


def probe_sampling(model, tag):
    """Frontier spot-check endpoints: no logprobs, check a plain short reply and its cost."""
    row = {"model": model, "provider_tag": tag, "logprobs": False}
    try:
        b = body(model, tag, lp=False); b["max_tokens"] = 16
        if model.startswith("openai/gpt-6") or "gemini" in model:
            b["reasoning"] = {"effort": "minimal"}
        r = client.chat(b, rep=0)["resp"]
        u = r.get("usage") or {}
        row.update(provider_served=r["provider"], content=repr(r["content"]), reasoning=bool(r["reasoning"]),
                   prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                   reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                   cost=u.get("cost"), finish=r["finish"])
    except Exception as e:
        row["error"] = str(e)[:240]
    return row


if __name__ == "__main__":
    jobs = []
    for m, tags in CANDS.items():
        tags = tags if tags is not None else endpoints_with_lp(m)
        lp_tags = set(endpoints_with_lp(m))
        for t in tags:
            jobs.append((probe if t in lp_tags else probe_sampling, m, t))
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(lambda j: j[0](j[1], j[2]), jobs))
    out = os.path.join(ROOT, "results", "llm", "catalog", f"probe_{datetime.date.today()}.csv")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(json.dumps(r))
    print("spend", client.spend())
