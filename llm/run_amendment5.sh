#!/usr/bin/env bash
# Amendment 5 analysis chain (rescue), resumable: each step skips work it already saved. Start with llm/bg.sh.
cd "$(dirname "$0")/.."
L=results/llm/amendment5; mkdir -p $L
echo "$(date -u +%FT%TZ) chain start" >> $L/chain.log
[ -f results/llm/fits/main/gemma-26b-local-bs1.json ] || python3 -u llm/analysis.py results/llm/stage1/main/gemma-26b-local-bs1.csv --B 1000 --out results/llm/fits/main > results/llm/fits/logs/fit_gemma-26b-local-bs1.txt 2>&1
echo "$(date -u +%FT%TZ) FITS_DONE" >> $L/chain.log
python3 -u llm/summarize.py --amendment5 > $L/summarize_amendment5.txt 2>&1 && echo "$(date -u +%FT%TZ) SUMMARY_DONE" >> $L/chain.log
python3 -u llm/compare_local.py --B 1000 --out $L --pairs "gemma-26b-local-bs1:gemma-26b:Parasail" "gemma-26b-local-bs1:gemma-26b-cw:CoreWeave" >> $L/compare_local.log 2>&1 && echo "$(date -u +%FT%TZ) COMPARE_DONE" >> $L/chain.log
echo "$(date -u +%FT%TZ) CHAIN_END" >> $L/chain.log
