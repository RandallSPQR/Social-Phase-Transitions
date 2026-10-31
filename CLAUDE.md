# Rules for Claude sessions on this repository

## Remote and paid jobs: record output as it is produced, never only at the end

Failure on record (2026-10-09, `llm/EXECUTION_LOG.md` entry 27): a 3-hour A100 run of the Gemma-4 battery produced
nothing. The pod's script uploaded all outputs in one tarball at the very end and then removed the pod whether or not
that upload worked. The upload failed, the container disk went with the pod, and every output and the log were lost.
Nobody could see progress while it ran. The money was small; the waste was the point.

For any job that runs somewhere you cannot read directly (GPU pod, cloud VM, batch API, long background process):

1. **Stream a heartbeat to durable storage** you can read from the session (here: the private HF dataset,
   `runs/<run>/status.json` + the full log, every ~90 s): phase, progress, ETA, hardware use, what has been uploaded.
2. **Upload each unit of output as soon as it exists** (per model, per shard, per checkpoint) and **verify** it
   (re-list remote files, compare sizes). Never one big upload at the end.
3. **Never destroy the machine or its disk unless the final upload is verified.** If it cannot be verified, leave it
   up and let an external time cap end it.
4. **Print everything to stdout too**, so the provider's console shows it live.
5. **Make it visible to the owner while it runs** (a live dashboard or a regularly reported status), with spend
   against the cap and time since the last heartbeat, so a stall is obvious within minutes.
6. **Prove the whole chain end to end before paying for it**: a short local or tiny run that exercises the
   heartbeat, per-unit uploads, verification, the final upload and the teardown rule.
7. Before launch, write down what you will see if it fails at each stage, and make sure each failure leaves a
   record behind.

Implementation in this repo: `llm/pod_tracker.py`, `llm/pod_run.sh`, `llm/runpod_launch.py` (watchdog).

## Shared resources

- The owner's other RunPod pods (e.g. the SAE calibration runs) and the shared network volume are off-limits: never
  stop, modify, or write to them. Do not mount the shared volume for this project's jobs.
- Deleting anything on a shared resource needs the owner's explicit OK first.

## Free-space checks

`df` on a network volume is meaningless (it reported ~500 PB free on a 150 GB quota). Do not size downloads by `df`
on network storage.

## Background processes

Start every background job with `llm/bg.sh start NAME CMD...` (own process group, PID file in
`results/llm/logs/bg/`) and stop it with `llm/bg.sh stop NAME`. Never stop anything with `pkill -f` / `pgrep -f`
patterns: they match their own and other tools' command lines (this killed the session shell and a monitor on
2026-10-09/10). Long jobs must save results per unit so a container restart costs at most one unit
(`llm/run_amendment4.sh` is the pattern).

## Reproducible environments on remote hosts

Failure on record (2026-10-13, `llm/EXECUTION_LOG.md` entry 51): the Stage 3 replicate ran on different graphs than
pre-registered because the pod image had networkx 3.3 preinstalled while this session uses 3.7, and an unpinned
`pip install networkx` kept the old version; `random_regular_graph` builds different graphs from the same seed across
versions. Found only through an implausible secondary measure after the run.

- Pin every library that generates experimental structure (graphs, seeds, samplers) to the exact local version in
  any remote job, and assert a committed fingerprint of the generated structure (e.g. `stage3.graph` checks
  `results/llm/stage3/graph_hashes.json`) before the first paid call.
- The end-to-end test of a remote job must compare a few generated objects with the local ones, not just that the
  job runs.
