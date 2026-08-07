# Reproduction — AI City Challenge 2026 Track 3 (inference)

Runs the inference side of our entry from a test JSON and the videos to a `submission.csv`. It samples
candidates from the fine-tuned models, votes across models and answer formats for the closed tasks, and
selects open-ended answers by MBR-BERTScore with per-task grounding. Final full-test mean **0.6696**
(the competition's live leaderboard scored only a visible 50% subset, on which the same pipeline showed 0.6790).

> Training (how the checkpoints are produced) lives in `../training/` with data prep in
> `../data_prep/` (see the top-level README); this directory covers **inference** and treats
> the merged checkpoints as given.

## 1. Requirements
- 5 GPUs recommended (f1 on 2, cosmos_v2 on 2, judge on 1); fewer works if you shrink the GPU groups / run serially.
- Python env from the repo-root `requirements.txt` (`vllm`, `openai`, `bert_score`; roberta-large auto-downloaded).
- Checkpoints:
  - `checkpoints/qwen36_27b_f1_128_ck3600_merged` (f1 — Qwen3.6-27B, 128 frames)
  - `checkpoints/cosmos_sft_merged` (cosmos_v2 — NVIDIA Cosmos3-Super, 32 frames)
  - `Qwen/Qwen3-30B-A3B-Instruct-2507` (text judge, auto-downloaded)

## 2. Input schema
The test JSON (`TEST_JSON`) must contain `{"items": [ {item_index, video_id, task_type, question, ...}, ... ]}`,
with the 10 task types per video (`bcq, mcq, bcq_openended, mcq_openended, open_qa, causal_linkage,
scene_description, temporal_description, video_summarization, temporal_localization`).
Videos resolve as `$VIDEO_DIR/<video_id>`.

## 3. Run
`run_full_fast.sh` runs the full 5-model ensemble and the final assembly to one submission. All models, paths, GPU groups, ports, and sample counts live in `config.sh` and are
overridable by environment variable:
```bash
REPO_DIR="$PWD" \
TEST_JSON=path/to/test.json \
VIDEO_DIR=path/to/videos \
  bash inference/run_full_fast.sh
# -> $SUB/submission_repro.csv   (SUB defaults to dataset/official_test/ in config.sh)
```
Every stage is **resume-safe** (re-running skips completed items), so an interrupted run continues. The
box-cue path additionally needs `YOLO_WEIGHTS` (external ~118 MB) to render boxed frames on first run.

Prefer to run it phase by phase (to inspect the candidate pools, or re-run a single phase after a
change)? `steps/` has the same flow cut into six numbered scripts — see `steps/README.md`.

## 4. What each component does
| Task | Recipe |
|---|---|
| bcq | cross-model vote (f1×12 + cosmos×5–12) → consistency-judge override on uncertain votes → **1-Yes-1-No** structural repair |
| mcq / mcq_openended | cross-model vote |
| open_qa / causal_linkage | MBR over [evidence-grounded + richpool] |
| video_summarization | MBR over [fullevent+mcq-option grounded + richpool] |
| temporal_description | MBR over **grounded-only** (locevent) |
| scene_description | MBR over richpool |
| bcq_openended | MBR over richpool; No-answers regenerated in GT style, polarity aligned to bcq |

Candidate pools land in `$R/` (see `config.sh`): `*_votetta.json` (votes), `mbr_test*.json` (richpool), `gen_f1_oe.json`, and the grounding JSONs.

## 5. Fallbacks when the test set differs
The score leans on the dataset structure (all 10 tasks per video, plus human-authored choice answers). If a
private/held-out set differs, the pipeline falls back rather than crashing:

| Condition on the test set | Behavior |
|---|---|
| Full 10 tasks/video (public set, or a same-pipeline private set) | All levers active → reproduces ~0.6696 (full test; 0.6790 on the visible-50% subset) |
| A video missing a context task (no temporal_loc / mcq / …) | Grounding for that item **falls back to richpool MBR**; bcq falls back to the plain vote |
| A video without exactly 2 bcq | 1-Yes-1-No repair **skipped** for it |
| No judge served (`--judge-url` empty) | consistency judge skipped; the vote and structural repair still apply |
| Omit `--pair-repair`, or leave `--judge-url` empty | Turns off the label prior / the judge respectively (for ablation, or if a private set breaks the assumption) |

If the organizer's set provides **only one task per video**, cross-task grounding cannot apply; the pipeline
still produces valid answers from the base model generation (roughly the base-vote level).

## 6. Expected output
`submission.csv`: `item_index,prediction`, one row per test item, 0 empty.
On the full hidden test this reproduces **0.6696** (per-task: bcq 0.9625, mcq 0.9500, bcq_oe 0.6312,
mcq_oe 0.9520, open_qa 0.5215, causal 0.5253, scene 0.4445, temporal 0.4975, summ 0.5416) — matching
Table `tab:main` in the paper. The competition's live leaderboard scored only a visible 50% subset, on
which the same pipeline showed 0.6790 (bcq reaches 1.0000 there because the 1-Yes-1-No prior fully
determines the visible bcq pairs; on the full test bcq is 0.9625).

## 7. What `run_full_fast.sh` does
The full 5-model deployed pipeline in one pass (box-cue adds about $+0.005$ on MCQ): serve cosmos-base + cosmos\_v2 + box-cue + f1 (128 f) → votes / richpool / OE-stance / orphans →
`assemble_clean.py` (cross-format × cross-model choice ensemble) → grounding gens (locevent / evidence /
full-event) → `assemble_faithful.py` (final open-ended MBR + bcq cross-task consistency judge + one-Yes-one-No
repair + bcq\_oe polarity). Box-cue needs the boxed test mp4s, rendered on first run via `detector_pass.py`
(YOLO) + `box_cue.py` overlay. The run is **not** bit-identical across machines (temperature sampling); confirm
the score by submitting the output to the leaderboard.

## 8. Files
- `run_full_fast.sh` — top-level runner (5-model ensemble → assembly → submission)
- `config.sh` — all models / paths / GPU groups / ports / sample counts (env-overridable)
- `common.py` — cleaning, MBR-BERTScore, parsing, grounding-prompt builders, guards
- `gen_*.py` — candidate generators (votes, OE-stance, richpool, grounding: locevent / evidence / full-event)
- `assemble_clean.py` — cross-format × cross-model choice ensemble (bcq/mcq/mcq\_oe)
- `assemble_faithful.py` — final assembler (open-ended MBR + bcq consistency judge + 1-Yes-1-No repair + bcq\_oe polarity)
- `stage_boxcue.sh`, `detector_pass.py`, `render_boxcue.py`, `box_cue.py` — box-cue frame rendering (YOLO overlay)
- `paths.py` — canonical paths (env-honored: `TEST_JSON` / `VIDEO_DIR` / `TRAIN_VIDEO_DIR`)
