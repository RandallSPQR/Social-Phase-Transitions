| check | result | evidence |
|---|---|---|
| R1 | PASS | 0 cache misses over 140003 replayed calls |
| R2 | PASS | 70/70 chains reproduce log, pA and states exactly |
| R3 | PASS | 70/70 chains with exactly 2,000 updates, complete (sweep, step), 21 states |
| R4 | PASS | 23 resume points in 17 chains, all consecutive (non-consecutive 0); re-drawn keys 0; 730 replies in 5 chains reused from an earlier host's identical request |
| R5 | PASS | provider mismatches 0, pin violations 0 |
| R6 | PASS | calls used = 2,000 + resamples in every chain: True; P1 inv 0 res 0 kept 0; P2 inv 1 res 1 kept 0; P2-nc inv 2 res 2 kept 0; P3 inv 0 res 0 kept 0; P4 inv 0 res 0 kept 0; P4-nc inv 0 res 0 kept 0; P5 inv 0 res 0 kept 0 |
| R7 | PASS | configurations recomputed on the recorded graph match the log in every update (part of R2) |
