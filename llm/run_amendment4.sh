#!/usr/bin/env bash
# Amendment 4 analysis chain, resumable: re-run this script after any interruption; every step skips work it already
# saved (comparison: per pair x arm cache; null check: per-run jsonl; probes: per-layer partial csv).
cd "$(dirname "$0")/.."
L=results/llm/amendment4; A=${ACTS:?set ACTS to the activations directory}
mkdir -p $L
echo "$(date -u +%FT%TZ) chain start" >> $L/chain.log
python3 -u llm/compare_local.py --B 1000 >> $L/compare_local.log 2>&1 && echo "$(date -u +%FT%TZ) COMPARE_DONE" >> $L/chain.log
python3 -u llm/compare_local_null.py --R 20 --B 100 >> $L/compare_null.log 2>&1 && echo "$(date -u +%FT%TZ) NULL_DONE" >> $L/chain.log
for k in gemma-31b-local gemma-26b-local; do
  python3 -u llm/probes.py --acts "$A" --key $k >> $L/probes_$k.log 2>&1 && echo "$(date -u +%FT%TZ) PROBES_$k" >> $L/chain.log
done
echo "$(date -u +%FT%TZ) CHAIN_END" >> $L/chain.log
