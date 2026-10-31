# Gemma-31B frozen, start-dependent regime: exploratory mechanism test — results

**Exploratory, labelled; it cannot change any confirmatory verdict.** Plan and pre-stated reading:
`llm/GEMMA_MECHANISM_PLAN.md`, committed before any run. Code: `llm/gemma_mechanism.py`. Numbers:
`results/llm/confirm/gemma_mechanism/summary.json`. Simulation only, from Gemma-31B's measured neutral K_avg. The T axis
is decision noise.

## Pre-stated reading: "both knockouts remove it: no single mechanism; each ingredient necessary"

Start-dependence at N = 1,600: the hot vs cold mean |m| gap, at 1× and 4× run length.

| kernel | T = 2.00 | T = 2.16 | T = 2.34 | persists (A3.3 rule)? |
|---|---|---|---|---|
| original | 0.27 → 0.18 | 0.26 → 0.13 | 0.24 → 0.07 | at T = 2.00 only |
| rival knockout | 0.000 → 0.000 | 0.001 → 0.000 | 0.000 → 0.000 | no |
| saturation knockout (additive) | 0.000 → 0.001 | 0.004 → 0.004 | 0.004 → 0.001 | no |

**Mean |m| at N = 1,600, 1× → 4×:**

| kernel | T | cold start | hot start |
|---|---|---|---|
| original | 2.00 | 0.40 → 0.34 | 0.12 → 0.16 |
| original | 2.34 | 0.38 → 0.36 | 0.14 → 0.29 |
| rival knockout | all three | 0.76–0.77 | 0.76–0.77 |
| additive | all three | 0.08–0.10 | 0.08–0.10 |

## What the knockouts actually do (post-hoc reading, labelled)

- **Neither knockout isolates a mechanism. Each changes the phase.**
  - Making rival positions uninformative turns the population into a strongly ordered one: |m| 0.77 from both starts.
    With rivals' repulsion gone, nothing opposes the allies' pull.
  - The additive projection of the same table gives ally weight +1.23, rival −0.90 (rivals repel) and own position
    +0.13 at T = 2. That model is disordered: |m| ≈ 0.09 from both starts.
  - Either way, start-dependence disappears because the system is no longer near the boundary where it orders slowly,
    not because a specific ingredient was removed.
- **What this does show:** the measured kernel's start-dependence needs both ingredients together.
  - Rival repulsion is needed: without it, the system simply orders.
  - The non-additive part of the table is needed: without it, the system does not order at all.
  - This fits a population sitting near an ordering boundary created by competing ally attraction and rival
    repulsion, where the higher-order (saturating) response decides on which side it falls.
- **The original kernel's start-dependence is slow relaxation, not a permanent freeze.**
  - Between 1× and 4× the hot starts gain order (0.12 → 0.16 at T = 2.0; 0.14 → 0.29 at T = 2.34) while the cold
    starts lose it.
  - By the A3.3 rule the gap persists only at T = 2.0. Under the A3.3 check's own seeds it persisted at all three T,
    so this is a borderline case near the decay threshold.

## Glass signatures (original kernel, W = 1,000 + 1,000, R = 50; descriptive)

| N | T | \|m\| | \|q\| identical starts | \|q\| independent starts | unsatisfied ties | floor (same graphs) | excess |
|---|---|---|---|---|---|---|---|
| 400 | 2.00 | 0.17 | 0.26 | 0.28 | 0.193 | 0.147 | 0.045 |
| 400 | 2.34 | 0.26 | 0.31 | 0.32 | 0.203 | 0.147 | 0.055 |
| 1,600 | 2.00 | 0.13 | 0.18 | 0.16 | 0.190 | 0.145 | 0.044 |
| 1,600 | 2.34 | 0.15 | 0.16 | 0.16 | 0.199 | 0.145 | 0.054 |

- **No glass signature in the overlap.** Two replicas from the *same* start, with different random draws, overlap no
  more than replicas from *independent* starts (|q| 0.16–0.32 in both cases). That is about the level their small
  magnetisation implies. The microscopic state does not stay frozen near its start.
- **Unsatisfied ties sit 0.04–0.06 above the frustration floor** of the same graphs (SD ≈ 0.004). This is roughly what
  thermal disorder at T ≈ 2 produces, and does not by itself indicate a glass.

## Bottom line

- **Mechanism: not a single one.** Under the pre-stated reading, both knockouts remove start-dependence, so no single
  ingredient is responsible.
- **Frustration alone:** not supported. The rival knockout removes start-dependence, but so does the saturation
  knockout.
- **Glass:** no evidence. The replica overlap shows no frozen microscopic state.
- **Best description:** slow, start-dependent relaxation near an ordering boundary, which depends jointly on rival
  repulsion and the non-additive shape of the response.
- **Inertia** is small in Gemma's neutral fits (γ 0.08, not significant). It is unlikely to be the driver, but it was
  not knocked out here.
