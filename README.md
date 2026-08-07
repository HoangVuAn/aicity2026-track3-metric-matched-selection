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
└── inference/           inference phase -> submission CSV
```

## Reproduce

### 0. Environment (one env runs the whole pipeline; tested with Python 3.13 / CUDA 12.8)
```bash
conda create -n aicity python=3.13 -y
conda activate aicity
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Download the base models once (~240 GB total: 52 + 128 + 57 GB) and export the cache paths:
```bash
export HF_HOME=~/.cache/huggingface        # or wherever you keep the HF cache
huggingface-cli download Qwen/Qwen3.6-27B
huggingface-cli download nvidia/Cosmos3-Super
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507
export COSMOS_SNAP=$HF_HOME/hub/models--nvidia--Cosmos3-Super/snapshots/<hash>   # for Cosmos training/merge
```
Notes:
- YOLO weights `yolo26x.pt` (box-cue) are auto-downloaded by `ultralytics` on first use.
- Every path/binary is env-overridable; defaults live in `inference/config.sh`.
- Place the official challenge data under `dataset/` (see "Not shipped" below).

### 1. Data prep  (raw challenge JSONs -> training corpus)
```bash
python3 data_prep/build_corpus.py        # -> training/data/merged_sft.jsonl (44,413 records)
sha256sum training/data/merged_sft.jsonl # must match data_prep/expected_outputs.json
```
The build is byte-for-byte deterministic; `data_prep/expected_outputs.json` holds the reference
sha256 and record counts so you can verify your corpus is identical to the one we trained on.
(The corpus itself is not shipped: it is derived from the challenge annotations, which we may
not redistribute.)
Turns the 10 raw task JSONs (place the official data under `dataset/train/`) into the tier-tagged
training corpus (S/A/B tier classification + cross-task consistency + dataset build; the reasoning
field is kept as flat text and wrapped as `<think>...</think>` at training time). Frame extraction is
done on-the-fly inside training (`training/trainlib/data.py`). Thresholds and paths live in
`data_prep/config/data_prep.yaml`.

External Vad-R1 set (optional, used by `cosmos_v2`): download Vad-Reasoning-SFT from the official
release (https://github.com/wbfwonderful/Vad-R1) into `dataset/external/vad_r1/`; without it the
builder warns and produces the 41,844-record challenge-only corpus (`--skip-vad` to force).

### 2. Train the 3 fine-tuned models (each ~hours on H100; LoRA + merge)
```bash
# f1 (primary open-ended generator, 128 frames, 7 open-ended tasks)
bash training/qwen_sft/chain_27b_f1_128.sh       # -> checkpoints/qwen36_27b_f1_128 (3655 steps, save every 400)
python training/qwen_sft/merge.py checkpoints/qwen36_27b_f1_128/checkpoint-3600 \
       checkpoints/qwen36_27b_f1_128_ck3600_merged        # ck3600 = the deployed checkpoint

# cosmos_v2 (Cosmos3-Super SFT, all tasks, tiers S+A, 32 frames)
bash training/cosmos_v2/train_sft.sh --config training/cosmos_v2/configs/sft.yaml --deepspeed   # -> checkpoints/cosmos_sft
python training/cosmos_v2/merge_lora.py --adapter checkpoints/cosmos_sft/checkpoint-3800 \
       --out checkpoints/cosmos_sft_merged                # ck3800 = the deployed checkpoint

# box-cue (Qwen3.6-27B on YOLO-boxed frames, bcq+mcq only, 64 frames; box_cue.py + YOLO box cache)
YOLO_CACHE_DIR=training/data/yolo_boxes python3 inference/detector_pass.py   # build the train-split box cache once
bash training/qwen_sft/chain_27b_boxcue_64.sh   # -> checkpoints/qwen36_27b_bcqmcq_boxcue (600 steps)
python training/qwen_sft/merge.py checkpoints/qwen36_27b_bcqmcq_boxcue/checkpoint-600 \
       checkpoints/qwen36_27b_bcqmcq_boxcue_merged
```

### 2b. Or skip training: released checkpoints

The three merged checkpoints are published on Hugging Face, so inference can be verified
without retraining:

| Model | Hugging Face |
|---|---|
| f1 | [AnHoang200901/aicity2026-track3-f1-qwen3.6-27b](https://huggingface.co/AnHoang200901/aicity2026-track3-f1-qwen3.6-27b) |
| box-cue | [AnHoang200901/aicity2026-track3-boxcue-qwen3.6-27b](https://huggingface.co/AnHoang200901/aicity2026-track3-boxcue-qwen3.6-27b) |
| cosmos_v2 | [AnHoang200901/aicity2026-track3-cosmos3-super-sft](https://huggingface.co/AnHoang200901/aicity2026-track3-cosmos3-super-sft) |

Point the pipeline at them via env overrides (HF ids work anywhere a local path does):
```bash
export F1_MODEL=AnHoang200901/aicity2026-track3-f1-qwen3.6-27b
export BOXCUE_MODEL=AnHoang200901/aicity2026-track3-boxcue-qwen3.6-27b
export COSMOS_V2_MODEL=AnHoang200901/aicity2026-track3-cosmos3-super-sft
```

### 3. Inference -> submission
```bash
bash inference/run_full_fast.sh             # PAIR_REPAIR=1 default; 3 GPUs; ~4-5h
# -> dataset/official_test/submission_repro.csv
```
All model paths / frame counts / sampling N / GPUs in `inference/config.sh`.
Every stage is resume-safe (cached under `inference/full_run/`), so reruns skip finished work;
`data_prep/expected_outputs.json` also lists the expected item counts per step for mid-pipeline
sanity checks. Reference intermediate artifacts from our verified run (vote pools, candidate
pools, grounding generations, submissions — model outputs only, no challenge content) are on
Hugging Face: [AnHoang200901/aicity2026-track3-artifacts](https://huggingface.co/datasets/AnHoang200901/aicity2026-track3-artifacts) —
drop them into `inference/full_run/` to experiment with the selection/assembly stages without
re-running generation.

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

## Method (see the workshop paper)
Cross-format cross-model vote (choice) · cross-task consistency check (text judge over own sibling answers) · grounding
(inject temporal-loc phrasing/evidence into open-ended regen) · MBR-BERTScore medoid (descriptions) ·
1-Yes-1-No pair-repair (bcq post-process, ON by default).

## Not shipped (must supply)
Base models (via HF_HOME), test videos + `test.json` (`dataset/official_test/`).
Merged checkpoints are on Hugging Face (Sec. 2b) or retrainable via Sec. 2;
`training/data/merged_sft.jsonl` is regenerated by step 1; YOLO weights auto-download.

## License
Code: **MIT** (see `LICENSE`). The models and datasets it uses carry their own licenses
(NVIDIA OpenMDW-1.1, Qwen Apache-2.0, YOLO AGPL-3.0, dataset terms) — see `NOTICE`. Obtain
models/data separately and comply with their licenses.
