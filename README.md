# AI City Challenge 2026 — Track 3 (Traffic Anomaly Reasoning)

Code for our Track-3 entry: data prep, training, and inference to a submission CSV.
Public leaderboard mean **0.6696**.

The repo is organised by phase — `data_prep/`, `training/`, `inference/`. Run everything from
this folder; the training scripts add `training/` to the Python path for the shared
`trainlib` library, and `inference/` is self-contained.

## Task
10 question types per traffic clip: `bcq`, `bcq_openended`, `mcq`, `mcq_openended`, `open_qa`,
`causal_linkage`, `scene_description`, `temporal_description`, `video_summarization`, `temporal_localization`.

## Models (ensemble; one unified inference pipeline routes per task)
| Alias | Checkpoint / id | Trained by |
|---|---|---|
| **f1** | `qwen36_27b_f1_128_ck3600_merged` (Qwen3.6-27B, 128f) | `training/qwen_sft/chain_27b_f1_128.sh` |
| **cosmos_v2** | `cosmos_sft_merged` (Cosmos3-Super SFT, 32f) | `training/cosmos_v2/train_sft.sh` |
| **box-cue** | `qwen36_27b_bcqmcq_boxcue_merged` (Qwen3.6-27B, YOLO-boxed, 64f) | `training/qwen_sft/train_ds.sh --box-cue` |
| cosmos-base | `nvidia/Cosmos3-Super` | zero-shot (off-the-shelf) |
| judge | `Qwen/Qwen3-30B-A3B-Instruct-2507` | off-the-shelf (text cross-task consistency judge) |

## Layout
```
release/
├── README.md
├── data_prep/           data-prep phase
│   ├── prep/            library: tier classify, cross-task consistency, dataset builder
│   ├── extract_keyframes.py
│   └── config/data_prep.yaml   thresholds / paths
├── training/            training phase (3 fine-tuned models)
│   ├── trainlib/       shared lib: data.py (frame extract/sampling), model.py, inference.py, paths.py
│   ├── qwen_sft/       Qwen3.6-27B SFT: train_ds.sh + train_sft.py, box_cue.py (box-cue),
│   │                   chain_27b_f1_128.sh (f1 orchestrator), merge.py
│   ├── cosmos_v2/      train_sft.py/.sh, merge_lora.py, configs/sft.yaml   (Cosmos3-Super SFT)
│   └── data/           merged_sft.jsonl  (regenerate via data-prep; see below)
├── inference/           inference phase -> submission CSV
└── docs/                method notes
```

## Reproduce

### 0. Environment (external, not shipped)
- One Python 3.13 / CUDA 12.8 venv: `pip install -r requirements.txt` (pinned; see its header for the torch install).
- `HF_HOME` with base models cached: `Qwen/Qwen3.6-27B`, `nvidia/Cosmos3-Super`, `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- YOLO weights `yolo26x.pt` (for box-cue). Set paths in `inference/config.sh`.

### 1. Data prep  (raw challenge JSONs -> training corpus)
`data_prep/prep/` turns the 10 raw task JSONs into the tier-tagged training corpus
`training/data/merged_sft.jsonl` (S/A/B tier classification + cross-task consistency + dataset build;
the reasoning field is kept as flat text and wrapped as `<think>...</think>` at training time). Frame
extraction is done on-the-fly inside training (`training/trainlib/data.py`). Thresholds and paths live in
`data_prep/config/data_prep.yaml`.

### 2. Train the 3 fine-tuned models (each ~hours on H100; LoRA + merge)
```bash
# f1 (primary open-ended generator, 128 frames, 7 open-ended tasks)
bash training/qwen_sft/chain_27b_f1_128.sh       # -> checkpoints/qwen36_27b_f1_128 ; then training/qwen_sft/merge.py

# cosmos_v2 (Cosmos3-Super SFT, all tasks, tiers S+A, 32 frames)
bash training/cosmos_v2/train_sft.sh         # config training/cosmos_v2/configs/sft.yaml ; then merge_lora.py

# box-cue (Qwen3.6-27B on YOLO-boxed frames, bcq+mcq only, 64 frames; box_cue.py + YOLO box cache)
bash training/qwen_sft/chain_27b_boxcue_64.sh   # -> checkpoints/qwen36_27b_bcqmcq_boxcue ; then training/qwen_sft/merge.py
```

### 3. Inference -> submission
```bash
bash inference/run_full_fast.sh             # PAIR_REPAIR=1 default; 3 GPUs; ~4-5h
# -> dataset/official_test/submission_repro.csv
```
All model paths / frame counts / sampling N / GPUs in `inference/config.sh`.

Or run it phase by phase (same flow, one script per phase; see `inference/steps/README.md`):
```bash
bash inference/steps/1_boxcue_prep.sh       # YOLO-boxed test mp4s (skipped if already rendered)
bash inference/steps/2_votes.sh             # cosmos-base + cosmos_v2 + box-cue -> vote pools
bash inference/steps/3_richpool.sh          # f1 + cosmos_v2 -> open-ended MBR pools
bash inference/steps/4_assemble_choice.sh   # cross-format x cross-model vote -> submission_clean_repro.csv
bash inference/steps/5_grounding.sh         # re-serve f1 -> locevent / full-event / summ / evidence pools
bash inference/steps/6_assemble_final.sh    # judge + open-ended MBR + 1-Yes-1-No -> submission_repro.csv
```

## Expected result on rerun (no baked artifacts; `seed=0`)
| Component | Rerun |
|---|---|
| choice (bcq vote, mcq, mcq_oe, bcq_oe stance) | reproduces exactly |
| open-ended (5 desc + open_qa, MBR-BERTScore) | ±0.01 (medoid sampling) |
| bcq pair-repair (1-Yes-1-No) | bcq **0.9625** on the full test |
| **overall** | **~0.66–0.67** (submitted 0.6696) |

Small deviations are expected: the pipeline regenerates everything, including the text judge's verdicts, and
no ground-truth-derived file is shipped.

## Method (see `docs/`)
Cross-format cross-model vote (choice) · cross-task consistency check (text judge over own sibling answers) · grounding
(inject temporal-loc phrasing/evidence into open-ended regen) · MBR-BERTScore medoid (descriptions) ·
1-Yes-1-No pair-repair (bcq post-process, ON by default).

## Not shipped (must supply)
Merged checkpoints (`checkpoints/*_merged`), base models (via HF_HOME), test videos + `test.json`
(`dataset/official_test/`), YOLO weights. `training/data/merged_sft.jsonl` is regenerated by step 1.

## License
Code: **MIT** (see `LICENSE`). The models and datasets it uses carry their own licenses
(NVIDIA OpenMDW-1.1, Qwen Apache-2.0, YOLO AGPL-3.0, dataset terms) — see `NOTICE`. Obtain
models/data separately and comply with their licenses.
