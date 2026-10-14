# Substack visuals: captions and provenance

Script: `llm/substack_visuals.py`. The seed-selection rule and all settings were committed before the first render.
Later commits changed only labels. Statistics: `twins_stats.json`.

**Every caption carries this note:** *Simulated from each model's measured response table (order-averaged one-shot
battery), not live LLM calls. T is decision noise (every logit divided by T), not literal sampling temperature.*

**Shared setting:**
- One random-regular network: 400 agents, 4 contacts each, 25% rival ties, fingerprint `b29361d8478c7953`. The same
  network is used for every model.
- Every run lasts 1,000 sweeps, which is at least each model's confirmed equilibration length.
- Single runs use seed pair 0, fixed in advance.

## kymographs.png

*Who holds which position over time in four simulated populations.*
- Each pixel row is one agent: blue = A, orange = B. Agents are ordered by the network's **Fiedler vector** (spectral
  ordering). On this well-mixed random network that ordering produces little visible clustering.
- The line above each panel is |m(t)|, the size of the majority.
- **Qwen3.5-122B and Gemini 3.1 Flash-Lite** are shown at 0.8×, 1× and 1.25× their tipping point T_c (2.01 and 1.83):
  they settle into a majority below T_c and wander at T_c.
- **GPT-4o-mini** (T = 1, 2, 3) never forms a majority.
- **Gemma-4 31B** (T = 1, 2, 3) is shown from a random start and from an all-A start. The two starts end in different
  places (|m| ≈ 0.1 vs ≈ 0.4 at T = 1–2): history matters.

## twins_kymographs.png

*Identical twins, one with a tiny error.*
- Run A uses the model's table. Run B uses the same table, with agents leaning 0.1 nats more toward their contacts'
  side.
- Both runs share the same network, the same start and the same random draws.
- The displayed pair is the **median-divergence pair at T_c** (rank 25 of 50). It is used in every panel of that model.
- **Over 50 seed pairs, twins end in different states:**

| model | 0.8·T_c | T_c | 1.25·T_c |
|---|---|---|---|
| Qwen3.5-122B | 5/50 [CI 0.04–0.21] | **22/50 [0.31–0.58]** | 10/50 [0.11–0.33] |
| Gemini 3.1 Flash-Lite | 9/50 [0.10–0.31] | 12/50 [0.14–0.37] | 12/50 [0.14–0.37] |

- "End state" is the sign of the mean majority over the last 100 sweeps; |m| < 0.05 counts as undecided. CIs are
  Wilson 95%.
- **Honest reading:** for Qwen the tiny error changes the outcome far more often at the tipping point. For Gemini the
  rate is similar at all three T in this 400-agent view, although the aggregate amplification measure (|Δ|m|| per nat,
  `results/llm/AMPLIFICATION_CONFIRM_RESULTS.md`) peaks at Gemini's T_c too.

## twins_outcome_rate.png

The table above as a chart, with 95% Wilson CIs.

## Animations (≈ 12 s, 15 fps; GIF ≤ 2.7 MB, MP4 alongside)

Spectral-embedding layout (Laplacian eigenvectors 2–3, fixed), faint edges, sweep and |m| counter.

| file | content |
|---|---|
| `network_P1` | Qwen3.5-122B at T_c |
| `network_gemini` | Gemini 3.1 Flash-Lite at T_c |
| `network_selected` | GPT-4o-mini at T = 2 (no tipping point) |
| `network_gemma` | Gemma-4 31B at T = 2, random start (no tipping point) |
| `twins_P1`, `twins_gemini` | twin runs at T_c, side by side; the median-divergence pair |
