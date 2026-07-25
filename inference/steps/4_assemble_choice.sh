#!/usr/bin/env bash
# Step 4 — choice assembly: cross-format x cross-model vote over the step-2/3 pools
# -> submission_clean_repro.csv (the base the grounding gens and the final assembler build on).
# No vLLM serve; assemble_clean uses bert_score on one card ($G1).
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
kg
echo "########## step4 assemble_clean -> $SUB/submission_clean_repro.csv ##########"
CUDA_VISIBLE_DEVICES=$G1 $LOC "$INF/assemble_clean.py" --dir $R --out $SUB/submission_clean_repro.csv
echo "########## step4 DONE ##########"
