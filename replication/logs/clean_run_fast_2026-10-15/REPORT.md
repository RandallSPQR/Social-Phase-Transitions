# Replication report (fast mode)

Python 3.13.16 on Linux-6.18.44-fc-v114-x86_64-with-glibc2.41, 4 CPUs.

| claim | result | steps | files compared | files differing | units recomputed | headline numbers | minutes |
|---|---|---|---|---|---|---|---|
| C0 | **PASS** | base.tables, base.figures | 6 | 0 | 0 | 11/11 | 0.2 |
| C1 | **PASS** | s1.cache, s1.fits, s1.summary, s1.temperature, s1.amendments, s1.figures | 67 | 0 | 1 | 10/10 | 12.9 |
| C2 | **PASS** | s2.surrogates, s2.sims, s3.replay, s3.audit, k.test, k.avg, k.followups | 252 | 0 | 0 | 6/6 | 29.6 |
| C3 | **PASS** | k.test, k.avg, amp.stageB | 64 | 0 | 12 | 4/4 | 4.7 |
| C4 | **PASS** | s1.temperature, amp.stageB, conf.api, conf.locate, conf.step45 | 59 | 0 | 24 | 11/11 | 5.6 |
| C5 | **PASS** | conf.api, conf.locate, conf.step45, conf.gemma | 36 | 0 | 15 | 7/7 | 1.7 |
| S1 | **PASS** | r.tests, r.replicate | 21 | 0 | 0 | 4/4 | 6.0 |
| E1 | **not run** |  | 0 | 0 | 0 | 0/0 | 0.0 |
| X | **PASS** | substack | 18 | 0 | 0 | 0/0 | 1.1 |

Claims:

- **C0** Baseline: the simulator reproduces the Viana-Bray transitions (6 of 6 within 95% CIs) and the melting of frozen camps under non-reciprocity (paper §4.1)
- **C1** LLM response functions are not Ising-like: H1 9 of 12 (12 of 15, 15 of 18), H2 supported, H3 2 of 12, H4 not supported, H5 7 of 12 (paper §4.2)
- **C2** Trait summaries fail live (surrogate 0 of 4); order-averaged tables succeed per step (K_avg |dP| <= 0.004) (paper §4.3-4.4)
- **C3** Small per-step differences are amplified collectively (K_fixed vs K_live 3.4 SD from |dP| 0.002) (paper §4.4, §4.7.1)
- **C4** Near a tipping point amplification exceeds controls: Stage B refuted; corrected Rule A 3 of 3 shapes for both models with a transition; GPT-4o-mini below floor (paper §4.7)
- **C5** Whether a tipping point exists is model-dependent: two clear transitions, GPT-4o-mini none, Gemma unresolved (paper §4.8)
- **S1** Residual persistence excess does not replicate (Stouffer Z = -0.42) (paper §4.5-4.6)
- **E1** Exploratory probes on Gemma-4 activations (paper §4.2; no claim rests on them; full mode only)
- **X** Illustrations in figures/substack/ (not part of the analysis)

## Headline numbers (regenerated value vs the value printed in the paper)

| claim | quantity | paper | regenerated | ok |
|---|---|---|---|---|
| C0 | transitions within their 95% CI of theory (of 6) | 6 | 6.0 | yes |
| C0 | consensus rho=0: simulated T | 3.917 | 3.9168 | yes |
| C0 | consensus rho=0.2: simulated T | 2.156 | 2.1556 | yes |
| C0 | glass c=4: simulated T | 1.813 | 1.8131 | yes |
| C0 | glass c=6: simulated T | 2.233 | 2.2331 | yes |
| C0 | glass persistence at eps=0 (N = 4000) | 0.51 | 0.5122 | yes |
| C0 | glass persistence at eps=0.1 | 0.28 | 0.2831 | yes |
| C0 | glass persistence at eps=0.2 | 0.11 | 0.1086 | yes |
| C0 | glass persistence at eps=0.35 | 0.01 | 0.0144 | yes |
| C0 | consensus |m| at eps=0 | 0.62 | 0.6225 | yes |
| C0 | consensus |m| at eps=1 | 0.55 | 0.5482 | yes |
| C1 | H1 supported (of 12 confirmatory cells) | 9 | 9.0 | yes |
| C1 | H1 supported with Amendment 4 (of 15) | 12 | 12.0 | yes |
| C1 | H1 supported with Amendment 5 (of 18) | 15 | 15.0 | yes |
| C1 | H2 letter field (of 12) | 10 | 10.0 | yes |
| C1 | H2 content field (of 8 content cells) | 8 | 8.0 | yes |
| C1 | H3 pairwise supported (of 12) | 2 | 2.0 | yes |
| C1 | H3 cubic supported (of 12) | 0 | 0.0 | yes |
| C1 | H5 departs from summation (of 12) | 7 | 7.0 | yes |
| C1 | H1 gate-relaxed (of 21) | 18 | 18.0 | yes |
| C1 | H4 verdict | not supported | not supported | yes |
| C2 | surrogate: P1-P4 points passing both primary measures | 0 | 0.0 | yes |
| C2 | kernel test: K_avg points passing (of P1-P4) | 2 | 2.0 | yes |
| C2 | kernel test: A3 points passing (of P1-P4) | 1 | 1.0 | yes |
| C2 | K_avg vs K_live |dP| at P1 | 0.004 | 0.00376 | yes |
| C2 | K_avg vs K_live |dP| at P4 | 0.003 | 0.00302 | yes |
| C2 | K_avg vs K_live |dlogit| at P4 (nats) | 0.07 | 0.0688 | yes |
| C3 | K_fixed vs K_live |dP| at P3 | 0.002 | 0.00212 | yes |
| C3 | K_fixed vs K_live persistence difference at P3 (SD) | -3.4 | -3.365 | yes |
| C3 | same-N table: per nat at T_c, N = 1600 | 1.27 | 1.2725 | yes |
| C3 | same-N table: per nat at T = 1, N = 100 | 0.009 | 0.00905 | yes |
| C4 | Stage B: T_c of P1 | 1.93 | 1.9346 | yes |
| C4 | Stage B: H1 growth G, noise shape | 5.5 | 5.501 | yes |
| C4 | Stage B: pre-registered verdict (H2 for majority shift) | False | False | yes |
| C4 | confirmatory: Rule A passes for P1 (of 3 shapes) | 3 | 3.0 | yes |
| C4 | confirmatory: Rule A passes for Gemini (of 3 shapes) | 3 | 3.0 | yes |
| C4 | confirmatory: D, P1 stay-bias | 0.15 | 0.1523 | yes |
| C4 | confirmatory: D, P1 noise | 2.03 | 2.0319 | yes |
| C4 | confirmatory: D, Gemini majority | 1.03 | 1.0327 | yes |
| C4 | confirmatory: Rule B growth G, P1 noise | 11.7 | 11.712 | yes |
| C4 | GPT-4o-mini control: max |m| amplification, stay | 0.075 | 0.07452 | yes |
| C4 | GPT-4o-mini control: max |m| amplification, noise | 0.096 | 0.0963 | yes |
| C5 | T_c of P1 (confirmatory) | 2.01 | 2.0146 | yes |
| C5 | T_c of Gemini | 1.83 | 1.8319 | yes |
| C5 | GPT-4o-mini: transition located | None | None | yes |
| C5 | GPT-4o-mini: highest Binder cumulant at N = 1600 | 0.47 | 0.4655 | yes |
| C5 | Gemma-31B: transition located (pre-registered verdict) | None | None | yes |
| C5 | Gemma rival knockout |m| (N = 1600) | 0.77 | 0.7678 | yes |
| C5 | Gemma additive projection |m| (N = 1600) | 0.09 | 0.0891 | yes |
| S1 | replicate: Stouffer Z (actual graphs) | -0.42 | -0.4187 | yes |
| S1 | combined effect, original run | 0.0116 | 0.011627 | yes |
| S1 | combined effect, pooled | 0.0033 | 0.003348 | yes |
| S1 | heterogeneity between runs, Cochran Q | 7.6 | 7.622 | yes |

## Steps

| step | result | minutes | files | differing | spot units | peak RSS so far (MB) | exit |
|---|---|---|---|---|---|---|---|
| base.tables | PASS | 0.1 | 2 | 0 | 0 | 103 | 0 |
| base.figures | PASS | 0.1 | 4 | 0 | 0 | 158 | 0 |
| s1.cache | PASS | 3.3 | 0 | 0 | 0 | 4645 | 0 |
| s1.fits | PASS | 7.5 | 20 | 0 | 0 | 4645 | 0 |
| s1.summary | PASS | 0.2 | 17 | 0 | 0 | 4645 | 0 |
| s1.temperature | PASS | 1.4 | 9 | 0 | 0 | 4645 | 0 |
| s1.amendments | PASS | 0.4 | 17 | 0 | 1 | 4645 | 0 |
| s1.figures | PASS | 0.1 | 4 | 0 | 0 | 4645 | 0 |
| s2.surrogates | PASS | 0.9 | 33 | 0 | 0 | 4645 | 0 |
| s2.sims | PASS | 15.5 | 12 | 0 | 0 | 4645 | 0 |
| s3.replay | PASS | 1.6 | 144 | 0 | 0 | 4645 | 0 |
| s3.audit | PASS | 4.5 | 11 | 0 | 0 | 4645 | 0 |
| k.test | PASS | 0.9 | 39 | 0 | 0 | 4645 | 0 |
| k.avg | PASS | 1.3 | 10 | 0 | 0 | 4645 | 0 |
| k.followups | PASS | 5.0 | 3 | 0 | 0 | 4645 | 0 |
| r.tests | PASS | 3.9 | 11 | 0 | 0 | 4645 | 0 |
| r.replicate | PASS | 2.1 | 10 | 0 | 0 | 4645 | 0 |
| amp.stageB | PASS | 2.5 | 15 | 0 | 12 | 4645 | 0 |
| conf.api | PASS | 0.8 | 19 | 0 | 0 | 4645 | 0 |
| conf.locate | PASS | 0.5 | 8 | 0 | 6 | 4645 | 0 |
| conf.step45 | PASS | 0.4 | 8 | 0 | 6 | 4645 | 0 |
| conf.gemma | PASS | 0.0 | 1 | 0 | 3 | 4645 | 0 |
| substack | PASS | 1.1 | 18 | 0 | 0 | 4645 | 0 |

Total step time: 54.0 min.

Stage 1 cache check (replication/out/stage1/cache_check.json): all rows explained.
