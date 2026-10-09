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
