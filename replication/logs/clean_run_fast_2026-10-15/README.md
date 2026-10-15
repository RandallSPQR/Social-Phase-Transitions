# Clean-run record: fast mode, 2026-10-15

**Result: every claim that ran passes; 53 of 53 headline numbers equal the values printed in the paper; exit status 0.**
The exploratory probes (E1) run only in full mode.

## Setup

- **Image:** `docker build` of a `git archive` of commit 38b926f. The commits after it change documentation only.
  The container had never seen a checkout: no `.git`, no cache, no simulation units, no outputs.
- **Data:** inside the container, the run downloaded and SHA-256-verified the 31 cache files and the 2 simulation unit
  bundles from the Hugging Face dataset (private at the time; read with a token passed as an environment variable,
  never written to disk or logs).
- **Hardware:** 4 vCPU (x86-64), 15 GB RAM, Linux; Python 3.13.16 from `replication/requirements.lock` (`env.json`).
- **Time:** 54.2 min of steps, plus about 1 min of download and verification. The largest single process used
  4.6 GB of memory.
- **Network:** behind a TLS-intercepting proxy (its CA was mounted read-only). No API endpoint was contacted
  (`SPT_OFFLINE=1`).

## Files

| file | content |
|---|---|
| `driver.log` | the complete console output (build retries, fetch, every step, the claim table) |
| `REPORT.md`, `report.json` | the generated report: claims, headline numbers, steps, differences |
| `records.json` | every compared file with its profile, value count, worst absolute difference and spot checks |
| `steps/*.log` | the full output of every command of every step |
| `stage1_cache_check.json` | the row-by-row Stage 1 cache check |
| `run_fast.sh`, `build_and_run.sh` | the exact commands used (scratch paths shortened to `<scratch>`) |
| `earlier_attempts/` | the four earlier attempts on the same machine, each stopped by a defect that was then fixed (below) |

## Earlier attempts and what each one fixed

| attempt | stopped at | cause | fix |
|---|---|---|---|
| 1 | fetching the unit bundles | the tar safety check rejected the bundles' directory entries | directories under the declared prefixes allowed; links, absolute and parent paths still refused |
| 2 | unpacking the unit bundles | the host machine's disk allowance was full (not the package) | space freed on the host; the README now states the free-disk requirement |
| 3 | baseline figures | the tight-bbox PNGs were 2–18 px (<1%) larger with the container's fonts; the check required identical pixel size | figure check tolerates ≤ 2% per dimension; regenerated files are kept for inspection |
| 4 | completed; 3 claims failed | the Stage 3 audit's B11 check called `git`, absent from the image; the probes' live API error texts cannot be replayed offline | B11 runs without git (provenance fields recorded as null, documented as not compared); only the error texts are skipped, the probe outcomes are compared |
