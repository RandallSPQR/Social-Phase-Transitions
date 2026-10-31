# Clean-run record: full mode, 2026-10-15 (partial: the host restarted mid-run)

## What happened

- **Started:** 04:54 UTC in a fresh container (image from `git archive` of the HEAD at launch; same setup as the fast
  record, `replication/logs/clean_run_fast_2026-10-15/`).
- **Lost:** the cloud host restarted between about 13:40 and 17:35 UTC, after the session went idle, and the run
  ended with it. The container and its working disk are gone. The step records were written to a mounted directory
  as each step finished, so they survived (CLAUDE.md: save results per unit).
- **Steps finished:** 8 of 25, all of them **PASS** (`records.json`). The `base.sim` result was recomputed after the
  fact with `--recompare` logic, using the comparator fix for pickled `.npy` arrays. In the run's own log, that step
  shows FAIL only because of that comparator bug.

## Results (measured)

| step | minutes | files compared | differing | note |
|---|---|---|---|---|
| base.sim | 153.7 | 23 | 0 | 22 validation arrays and the asymmetry sweep recomputed from scratch, all bit-identical |
| base.tables | 0.1 | 2 | 0 | |
| base.figures | 0.1 | 4 | 0 | |
| s1.cache | 3.8 | (row check) | 0 | every committed Stage 1 row tied to the cache |
| s1.fits | 8.3 | 20 | 0 | |
| s1.summary | 0.2 | 17 | 0 | |
| s1.temperature | 1.5 | 9 | 0 | |
| s1.amendments | 79.5 | 17 | 0 | all 14 bootstrap caches recomputed from scratch |
| s1.null | (interrupted) | — | — | 20 of 100 replicates finished, about 12.5–15 min each; all 20 equal the committed runs at printed precision (`steps/s1.null.log`) |

## Still unmeasured in full mode

Every step after `s1.null`: Stage 2–3, the kernel test, the residual tests, amplification Stage B, the confirmatory
steps and the probes. All except the probes passed in fast mode, which recomputes a sample of their units. The README
gives estimates for their full-mode times.
