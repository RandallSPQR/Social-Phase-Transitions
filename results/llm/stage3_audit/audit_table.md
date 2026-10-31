| check | result | evidence |
|---|---|---|
| A P1 | PASS | abs_m: live-seeds 0.401 in [0.24, 0.46] yes, reps inside 35/40; persistence: live-seeds 0.682 in [0.66, 0.74] yes, reps inside 36/40 |
| A P2 | FAIL | m: live-seeds 0.630 in [0.58, 0.66] yes, reps inside 29/40; persistence: live-seeds 0.637 in [0.59, 0.67] yes, reps inside 37/40 |
| A P3 | PASS | abs_m: live-seeds 0.213 in [0.20, 0.37] yes, reps inside 33/40; persistence: live-seeds 0.936 in [0.91, 0.95] yes, reps inside 34/40 |
| A P4 | PASS | m: live-seeds 0.882 in [0.87, 0.91] yes, reps inside 36/40; persistence: live-seeds 0.806 in [0.75, 0.84] yes, reps inside 40/40 |
| A P5 | PASS | abs_m: live-seeds 0.186 in [0.14, 0.24] yes, reps inside 33/40; persistence: live-seeds 0.167 in [0.11, 0.20] yes, reps inside 37/40 |
| A P2-nc | PASS | m: live-seeds 0.411 in [0.34, 0.43] yes, reps inside 39/40; persistence: live-seeds 0.507 in [0.47, 0.54] yes, reps inside 34/40 |
| A P4-nc | FAIL | m: live-seeds 0.511 in [0.52, 0.58] NO, reps inside 33/40; persistence: live-seeds 0.462 in [0.41, 0.48] yes, reps inside 38/40 |
| B1 | PASS | all chains 20 sweeps x 100 updates; agent draw χ² min p (Bonferroni) P1 1.00, P2 1.00, P3 0.03, P4 0.44, P5 1.00, P2-nc 1.00, P4-nc 0.85 |
| B2 | PASS | 140005 calls; stale/mismatched neighbour states 0 |
| B3 | PASS | mismatches {'mapping_logged_vs_prompt': 0, 'order_logged_vs_prompt': 0, 'gloss_inconsistent': 0, 'pA_mismatch': 0, 'letter_mismatch': 0, 'applied_position_mismatch': 0, 'exact_replay_chains_differing': 0}; chains replayed 70 |
| B4 | PASS | own-line mismatches 0 |
| B5 | PASS | P1 ρ=0.25 realised 0.220–0.280; P2 ρ=0.5 realised 0.485–0.550; P3 ρ=0.25 realised 0.220–0.280; P4 ρ=0.25 realised 0.210–0.255; P5 ρ=0.0 realised 0.000–0.000; P2-nc ρ=0.5 realised 0.485–0.550; P4-nc ρ=0.25 realised 0.210–0.255 |
| B6 | PASS | P1 map0 0.502 ord0 0.499; P2 map0 0.495 ord0 0.497; P3 map0 0.503 ord0 0.498; P4 map0 0.499 ord0 0.502; P5 map0 0.500 ord0 0.496; P2-nc map0 0.505 ord0 0.501; P4-nc map0 0.500 ord0 0.496 |
| B7 | PASS | P1 {'Alibaba': 20000}; P2 {'Parasail': 20004}; P3 {'CoreWeave': 20000}; P4 {'Io Net': 20000}; P5 {'OpenAI': 20000}; P2-nc {'Parasail': 20000}; P4-nc {'Io Net': 20001} |
| B8 | PASS | P1 0/20000 kept 0; P2 4/20000 kept 0; P3 0/20000 kept 0; P4 0/20000 kept 0; P5 0/20000 kept 0; P2-nc 0/20000 kept 0; P4-nc 1/20000 kept 0 |
| B9 | PASS | warm-up/measured 10/10 both; initial m z: mean -0.11, var 1.11 over 70 chains |
| B10 | PASS | max |Δ| vs measures_by_graph 1.1e-16, vs summary.csv 1.1e-16 |
| B11 | PASS | P1 qwen-122b/neutral_own/ρ0.25/fitted 40 runs; P2 qwen-9b/political/ρ0.5/fitted 40 runs; P3 llama-70b/political/ρ0.25/fitted 40 runs; P4 nemo-12b/workplace/ρ0.25/fitted 40 runs; P5 gpt6-luna/neutral_own/ρ0.0/fitted 40 runs; P2-nc qwen-9b/political/ρ0.5/fields_only 40 runs; P4-nc nemo-12b/workplace/ρ0.25/fields_only 40 runs |
| C P1 | PASS | n 20000; observed A 0.490 vs battery 0.493 (Δ -0.003); slope 0.91 [0.89, 0.94]; RMS logit live vs battery pA 0.83 |
| C P2 | PASS | n 20000; observed A 0.492 vs battery 0.492 (Δ -0.000); slope 0.97 [0.94, 0.99]; RMS logit live vs battery pA 0.40 |
| C P3 | FAIL | n 20000; observed A 0.507 vs battery 0.506 (Δ +0.001); slope 0.67 [0.64, 0.70]; RMS logit live vs battery pA 0.55 |
| C P4 | PASS | n 20000; observed A 0.502 vs battery 0.504 (Δ -0.002); slope 0.95 [0.92, 0.98]; RMS logit live vs battery pA 0.78 |
| C P5 | FAIL | n 20000; observed A 0.488 vs battery 0.484 (Δ +0.003); slope 0.39 [0.38, 0.40] |
| C P2-nc | PASS | no battery cell (no contacts): n/a |
| C P4-nc | PASS | no battery cell (no contacts): n/a |
