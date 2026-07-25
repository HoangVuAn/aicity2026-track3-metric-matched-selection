#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Central config for the AI City Track-3 reproduction pipeline.
# Every value is env-overridable: `VAR="${VAR:-default}"`. Sourced by the .sh
# orchestrators (run_full_fast.sh, stage_*.sh). Python scripts read TEST_JSON /
# VIDEO_DIR via the environment (see paths.py) — those are exported below.
#
# To re-run on another machine, override the paths that start with /mnt/data:
#   REPO_DIR, PY, VLLM, LOC, the *_MODEL vars, YOLO_WEIGHTS, HF_HOME.
# ─────────────────────────────────────────────────────────────────────────────

# ── repo + python environments / binaries ──
REPO_DIR="${REPO_DIR:-/mnt/data/anhv10/ai_city_challenge}"
export HF_HOME="${HF_HOME:-/mnt/data/anhv10/hf_cache}"
PY="${PY:-/mnt/data/anhv10/cosmos3_env/bin/python}"     # vllm-client env (openai, cv2)
VLLM="${VLLM:-/mnt/data/anhv10/cosmos3_env/bin/vllm}"   # vllm serve binary
LOC="${LOC:-/mnt/data/anhv10/locate_env/bin/python}"    # env with bert_score (assembly)

# ── data I/O (python reads TEST_JSON / VIDEO_DIR via env → paths.py) ──
export TEST_JSON="${TEST_JSON:-$REPO_DIR/dataset/official_test/test/test.json}"
export VIDEO_DIR="${VIDEO_DIR:-$REPO_DIR/dataset/official_test/videos}"
SUB="${SUB:-$REPO_DIR/dataset/official_test}"           # output submission dir
R="${R:-$REPO_DIR/inference/full_run}"                  # generated candidate pools

# ── models (HF hub id OR local checkpoint dir) ──
F1_MODEL="${F1_MODEL:-$REPO_DIR/checkpoints/qwen36_27b_f1_128_ck3600_merged}"   # core VLM, 128 frames
COSMOS_BASE_MODEL="${COSMOS_BASE_MODEL:-nvidia/Cosmos3-Super}"                  # zero-shot choice votes
COSMOS_V2_MODEL="${COSMOS_V2_MODEL:-$REPO_DIR/checkpoints/cosmos_sft_merged}"   # Cosmos SFT
BOXCUE_MODEL="${BOXCUE_MODEL:-$REPO_DIR/checkpoints/qwen36_27b_bcqmcq_boxcue_merged}"  # YOLO-box overlay
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"                  # text judge (cross-task consistency)
export YOLO_WEIGHTS="${YOLO_WEIGHTS:-/mnt/data/anhv10/VLM_video_understanding/api_backend/yolo26x.pt}"
COSMOS_OVERRIDES="${COSMOS_OVERRIDES:-{\"architectures\":[\"Cosmos3ReasonerForConditionalGeneration\"]}}"

# ── vLLM serving params ──
FRAMES_COSMOS="${FRAMES_COSMOS:-32}"; FRAMES_BOXCUE="${FRAMES_BOXCUE:-64}"; FRAMES_F1="${FRAMES_F1:-128}"
MML_COSMOS="${MML_COSMOS:-16384}"; MML_F1="${MML_F1:-32768}"; MML_JUDGE="${MML_JUDGE:-8192}"
GPU_UTIL="${GPU_UTIL:-0.9}"
PORT_CB="${PORT_CB:-9310}"; PORT_CV2="${PORT_CV2:-9311}"; PORT_BX="${PORT_BX:-9312}"
PORT_F1="${PORT_F1:-9313}"; PORT_JUDGE="${PORT_JUDGE:-9502}"

# ── sampling counts (votes / richpool / grounding ; worker threads) ──
# NV=12, NR=24: EXACT config of the shipped 0.6696 base. NOTE: NV drives bcq vote stability; at NV=5 the fresh
# base produced 17 1Y1N-violations (vs 13 at NV=12) -> more pair-repair flips -> ~0.925 bcq. Rerun at NV=12
# reproduces the shipped base -> pair-repair (same cross-task-consistency prompt + votepool margin) -> ~0.975.
NV="${NV:-12}"; NR="${NR:-24}"; NG="${NG:-16}"; W="${W:-12}"

# ── GPUs (override to free, allowed cards) ──
G1="${G1:-1}"; G2="${G2:-2}"; G3="${G3:-3}"; BOXCUE_GPU="${BOXCUE_GPU:-$G3}"

# ── box-cue artifact dirs (detector cache + rendered mp4s) ──
export YOLO_CACHE_DIR="${YOLO_CACHE_DIR:-$R/yolo_boxes}"
export BOXCUE_MP4_DIR="${BOXCUE_MP4_DIR:-$R/boxcue_mp4}"

# ── final assembly ──
PAIR_REPAIR="${PAIR_REPAIR:-1}"   # 1 → 1-Yes-1-No structural repair (SHIPPED default). Consistency+margin flip is
# self-contained (fresh judge, no baked artifacts): a clean rerun lands bcq ~0.92-0.975 (flip-direction has
# run variance on confidence-inverted pairs) + open-ended ±0.01 (MBR medoid sampling). Small deviation is expected.
                                  # 0 → consistency-only (~0.671), robust to a differently-built private set
