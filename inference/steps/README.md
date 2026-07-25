# Step-by-step inference

The same pipeline as `../run_full_fast.sh`, cut into six scripts you can run one at a time —
useful for debugging, inspecting the candidate pools between phases, or re-running a single
phase after a change. `run_full_fast.sh` remains the canonical one-command runner; these
mirror it and read everything from `../config.sh`, so both stay in sync by construction.

Run in order (each step consumes the previous step's pools in `$R`):

```bash
bash steps/1_boxcue_prep.sh      # YOLO-boxed test mp4s (skipped if already rendered)
bash steps/2_votes.sh            # cosmos-base + cosmos_v2 + box-cue -> vote/TTA pools
bash steps/3_richpool.sh         # f1 + cosmos_v2 -> open-ended MBR pools
bash steps/4_assemble_choice.sh  # cross-format x cross-model vote -> submission_clean_repro.csv
bash steps/5_grounding.sh        # re-serve f1 -> locevent / full-event / summ-option / evidence pools
bash steps/6_assemble_final.sh   # judge + open-ended MBR + 1-Yes-1-No -> submission_repro.csv
```

Each step is resume-safe (the `gen_*.py` skip finished items) and frees the GPUs it used on
exit, so you can stop after any step and continue later. To re-run just the final assembly
after tweaking its logic (no re-serving the vision models), re-run only `6_assemble_final.sh`;
to re-do the choice vote from existing pools, only `4_assemble_choice.sh`.
