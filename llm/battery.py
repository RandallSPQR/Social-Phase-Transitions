"""Stage 1 battery: configurations, prompt templates, request bodies (PREREG §3).

Spin coding: s = +1 for content P, −1 for content Q.
  neutral:   P = position A (letters are the content; mapping fixed)
  political: P = support the measure,  Q = oppose the measure
  workplace: P = adopt the new tool,   Q = keep the current tool
Neighbour types: aP, aQ (ally holding P/Q), rP, rQ (rival holding P/Q).
"""
import hashlib, itertools, random

TYPES = ("aP", "aQ", "rP", "rQ")

FRAMINGS = {
    "neutral": {
        "intro": ("You are one member of a group. Every member holds one of two positions, {X} or {Y}, and can "
                  "switch at any time. Some members are your allies (people you usually side with) and some are "
                  "your rivals (people you usually oppose)."),
        "content": None,
    },
    "political": {
        "intro": ("You are a voter in a mid-sized town. The town will vote on a ballot measure to raise the local "
                  "sales tax by half a percent to fund public transit. Every voter holds one of two positions, "
                  "{X} ({cX}) or {Y} ({cY}), and can switch at any time before the vote. Some voters are your "
                  "political allies (people who usually vote the way you do) and some are your political rivals "
                  "(people who usually vote against your side)."),
        "content": {"P": "support the measure", "Q": "oppose the measure"},
    },
    "workplace": {
        "intro": ("You work on a team that is deciding whether to switch to a new project-management tool. Every "
                  "team member holds one of two positions, {X} ({cX}) or {Y} ({cY}), and can switch at any time "
                  "before the decision. Some colleagues are your allies (people you usually back at work) and "
                  "some are your rivals (people you usually clash with at work)."),
        "content": {"P": "adopt the new tool", "Q": "keep the current tool"},
    },
}

ARMS = {  # arm: (framing, own position shown, mappings, orders)
    "neutral_own":   ("neutral", True, (0,), (0, 1)),
    "neutral_noown": ("neutral", False, (0,), (0, 1)),
    "political":     ("political", True, (0, 1), (0, 1)),
    "workplace":     ("workplace", True, (0, 1), (0, 1)),
}


def multisets(kmax=6):
    """All neighbour multisets as count tuples (n_aP, n_aQ, n_rP, n_rQ), k = 1..kmax (209 for kmax=6)."""
    out = []
    for k in range(1, kmax + 1):
        for combo in itertools.combinations_with_replacement(range(4), k):
            out.append(tuple(combo.count(t) for t in range(4)))
    return out


def cells(own=True, kmax=6):
    ms = multisets(kmax)
    return [(m, s0) for m in ms for s0 in ((+1, -1) if own else (0,))]


def _h(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def neighbour_order(counts, ident, perm):
    """Deterministic random order of the individual neighbours, seeded by prompt identity + permutation index."""
    seq = [t for t, n in zip(TYPES, counts) for _ in range(n)]
    random.Random(int(_h(ident, perm)[:16], 16)).shuffle(seq)
    return seq


def letters(mapping):
    """mapping 0: P→A, Q→B; mapping 1: P→B, Q→A."""
    return ({"P": "A", "Q": "B"} if mapping == 0 else {"P": "B", "Q": "A"})


def render(framing, counts, s0, mapping, order, perm):
    """Return (prompt text, neighbour sequence). order 0: A listed first; 1: B listed first."""
    F = FRAMINGS[framing]
    let = letters(mapping)
    X, Y = ("A", "B") if order == 0 else ("B", "A")
    inv = {v: k for k, v in let.items()}            # letter -> content key
    def gloss(ckey):
        return "" if F["content"] is None else f" ({F['content'][ckey]})"
    intro = F["intro"].format(X=X, Y=Y, cX=(F["content"] or {}).get(inv[X], ""), cY=(F["content"] or {}).get(inv[Y], ""))
    ident = (framing, counts, s0, mapping, order)
    seq = neighbour_order(counts, ident, perm)
    lines = []
    for t in seq:
        who = "An ally" if t[0] == "a" else "A rival"
        lines.append(f"- {who} holds {let[t[1]]}{gloss(t[1])}.")
    parts = [intro, ""]
    if s0 != 0:
        own = "P" if s0 > 0 else "Q"
        parts += [f"You currently hold position {let[own]}{gloss(own)}.", ""]
    parts += ["The people you are in contact with, and the position each currently holds:", *lines, "",
              f"Which position do you take now? Answer with a single letter: {X} or {Y}."]
    return "\n".join(parts), seq


def items(arm, cell_filter=None, extra_perms=False):
    """Yield one dict per prompt in an arm. extra_perms: the order-sensitivity subset (PREREG §3)."""
    framing, own, maps, orders = ARMS[arm]
    for counts, s0 in cells(own):
        if cell_filter is not None and not cell_filter(counts, s0):
            continue
        variants = [(m, o) for m in maps for o in orders]
        if extra_perms:
            if sum(counts) < 3:
                continue
            m, o = variants[int(_h("opick", arm, counts, s0)[:8], 16) % len(variants)]
            perms, variants = (1, 2), [(m, o)]
        else:
            perms = (0,)
        for m, o in variants:
            for perm in perms:
                text, seq = render(framing, counts, s0, m, o, perm)
                yield {"arm": arm, "framing": framing, "counts": counts, "s0": s0, "mapping": m, "order": o,
                       "perm": perm, "seq": "".join(t[0] + t[1] for t in seq), "prompt": text}


def body(model, tag, prompt, mode="lp", reasoning_off=False, temperature=1.0):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature,
         "provider": {"order": [tag], "allow_fallbacks": False}}
    if mode == "lp":
        b.update({"max_tokens": 1, "logprobs": True, "top_logprobs": 5})
        b["provider"]["require_parameters"] = True
    else:
        b["max_tokens"] = 8
    if reasoning_off:
        b["reasoning"] = {"enabled": False}
    return b


if __name__ == "__main__":
    import sys
    print(len(multisets()), "multisets;", {a: sum(1 for _ in items(a)) for a in ARMS},
          "order subset:", sum(1 for _ in items("neutral_own", extra_perms=True)))
    for arm in ARMS:
        it = list(items(arm))
        print("=" * 30, arm); print(it[len(it) // 2 + (3 if arm == "political" else 0)]["prompt"])
