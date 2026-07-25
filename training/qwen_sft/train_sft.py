"""Qwen3.6-27B (model_type qwen3_5) LoRA SFT, run in cosmos3_env (transformers 5.10.1 supports
qwen3_5). Reuses the Cosmos data pipeline (cv2 frame extraction + tier-weighted records); only
the model/processor are Qwen3.6. Single process, device_map=auto over the visible GPUs.

  HF_HOME=/mnt/data/anhv10/hf_cache CUDA_VISIBLE_DEVICES=0,2 \
    cosmos3_env/bin/python training/qwen_sft/train_sft.py --n-frames 16 --max-steps 600 --cot
"""

from __future__ import annotations

import argparse
import re
import shutil
import json
import logging
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()

TRAIN_ROOT = Path(__file__).resolve().parents[1]      # training/ (holds the trainlib shared lib)
sys.path.insert(0, str(TRAIN_ROOT))
from trainlib.data import ALL_TASKS, TIER_DUP, _to_sec, to_sec_dur, extract_frames, _resize_native, load_sft_records, make_target  # noqa: E402
import box_cue as box_cue_mod  # noqa: E402  (B2-robust YOLO box-cue overlay + dropout)
from trainlib.inference import clip_duration  # noqa: E402
from trainlib.paths import rel_to_repo  # noqa: E402

MODEL = "Qwen/Qwen3.6-27B"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
logger = logging.getLogger(__name__)


def make_processor(model: str = MODEL):
    proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    proc.video_processor.do_sample_frames = False   # we pre-sample frames with cv2
    return proc


def load_model(model: str = MODEL, device_map: str | None = "auto", r: int = 128, alpha: int = 256):
    model = AutoModelForImageTextToText.from_pretrained(
        model, dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True,
        attn_implementation="sdpa")           # sdpa: fast + safe for the 32-frame vision context
    # device_map=None for DeepSpeed/FSDP (accelerate shards); "auto" for single-process naive MP.
    reentrant = device_map is None            # ZeRO-3 + grad ckpt needs reentrant=True
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    model.enable_input_require_grads()
    lora = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=LORA_TARGETS)   # text proj only -> vision tower stays frozen
    model = get_peft_model(model, lora)
    if device_map is None:                    # ZeRO-3 flattens params -> LoRA must be bf16 like base
        for p in model.parameters():
            if p.requires_grad and p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
    else:
        model.print_trainable_parameters()
    return model


def run_deepspeed(args) -> None:
    """Multi-GPU DeepSpeed ZeRO-3 (real data-parallel + model sharding). Launch with
    `accelerate launch --num_processes N`. ~N x throughput vs single-process device_map=auto."""
    from accelerate import Accelerator, DeepSpeedPlugin

    ds = DeepSpeedPlugin(zero_stage=3, gradient_accumulation_steps=args.grad_accum,
                         zero3_init_flag=True, offload_optimizer_device="none", offload_param_device="none")
    acc = Accelerator(mixed_precision="bf16", deepspeed_plugin=ds)
    dsc = acc.state.deepspeed_plugin.deepspeed_config
    dsc["train_micro_batch_size_per_gpu"] = 1
    dsc.setdefault("zero_optimization", {})["stage3_gather_16bit_weights_on_model_save"] = True
    logging.basicConfig(level=logging.INFO if acc.is_main_process else logging.ERROR,
                        format=f"%(asctime)s R{acc.process_index} %(levelname)s %(message)s", force=True)
    logger.setLevel(logging.INFO if acc.is_main_process else logging.ERROR)

    tasks = ALL_TASKS if args.tasks == "all" else set(args.tasks.split(","))
    records = load_data_jsonl(args.data, tasks) if args.data else \
        load_sft_records(tasks, set(args.tiers.split(",")), args.limit)
    shard = records[acc.process_index::acc.num_processes]
    wandb = None
    if acc.is_main_process:
        logger.info("DeepSpeed ZeRO-3 | %d GPU | %d records | %d/rank | n_frames=%d | loading Qwen3.6-27B...",
                    acc.num_processes, len(records), len(shard), args.n_frames)
        try:
            import wandb
            wandb.init(project="aicitytrack3", name=Path(args.output_dir).name,
                       config={k: v for k, v in vars(args).items()})
        except Exception as e:
            logger.warning("wandb off: %s", str(e)[:60]); wandb = None

    proc = make_processor(args.model)
    model = load_model(args.model, device_map=None); model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    model, opt = acc.prepare(model, opt)      # NOT the scheduler -> stepped on sync_gradients
    sched = get_cosine_schedule_with_warmup(opt, int(args.max_steps * 0.03), args.max_steps)
    acc.register_for_checkpointing(sched)     # clean-resume: scheduler state saved/loaded by save_state/load_state

    def next_batch(di):
        while True:
            rec = shard[di[0] % len(shard)]; di[0] += 1
            try:
                return build_batch(proc, rec, args.n_frames, args.sampling, args.cot, acc.device,
                                   args.adaptive_fps, args.raw_seconds, args.budget_fill, args.base_frames,
                                   args.augment, args.no_snap, args.snap_nearest, args.box_cue,
                                   args.window_sample), rec
            except Exception as e:
                logger.warning("skip %s: %s", rec.get("source"), str(e)[:80])

    out_dir = Path(args.output_dir)
    max_steps = 2 if args.smoke else args.max_steps
    di, step = [0], 0
    if args.resume_from:                      # clean-resume: restore model+opt(ZeRO-3)+sched+RNG + step/data-pos
        rdir = rel_to_repo(args.resume_from)
        acc.load_state(str(rdir))
        meta = json.loads((Path(rdir) / "resume_meta.json").read_text())
        step, di[0] = meta["step"], meta["di"]
        if acc.is_main_process:
            logger.info("RESUMED from %s at step %d/%d (di=%d)", args.resume_from, step, max_steps, di[0])
    while step < max_steps:
        (batch, plen), rec = next_batch(di)
        labels = batch["input_ids"].clone(); labels[0, :plen] = -100
        loss = model(**batch, labels=labels, use_cache=False).loss
        acc.backward(loss)
        opt.step(); opt.zero_grad()
        if acc.sync_gradients:
            sched.step(); step += 1
            if step % args.logging_steps == 0 and acc.is_main_process:
                logger.info("step %d/%d loss=%.4f lr=%.2e", step, max_steps, loss.item(), sched.get_last_lr()[0])
                if wandb:
                    wandb.log({"loss": loss.item(), "lr": sched.get_last_lr()[0],
                               "tile_scale": rec.get("tile_scale", "")}, step=step)
            if not args.smoke and step % args.save_steps == 0:
                acc.wait_for_everyone()
                acc.unwrap_model(model).save_pretrained(   # adapter (small) per milestone -> eval/inference
                    str(out_dir / f"checkpoint-{step}"), is_main_process=acc.is_main_process,
                    save_function=acc.save, state_dict=acc.get_state_dict(model))
                state_dir = out_dir / "state_latest"       # clean-resume: full state (model+opt+sched+RNG), rolling
                if acc.is_main_process:
                    shutil.rmtree(state_dir, ignore_errors=True)
                acc.wait_for_everyone()
                acc.save_state(str(state_dir))
                if acc.is_main_process:
                    (state_dir / "resume_meta.json").write_text(json.dumps({"step": step, "di": di[0]}))
                    logger.info("saved -> checkpoint-%d (+ state_latest: resume-ready)", step)
    if acc.is_main_process:
        logger.info("Qwen3.6 DeepSpeed SFT complete (%d steps, smoke=%s).", step, args.smoke)


def load_data_jsonl(path: str, tasks: set[str]) -> list[json.loads]:
    """Arbitrary jsonl corpus (e.g. temporal_mixed.jsonl): tier-dup like load_sft_records."""
    recs = []
    for line in open(rel_to_repo(path)):
        r = json.loads(line)
        if r["task"] in tasks and rel_to_repo(r["video_path"]).exists():
            recs.extend([r] * TIER_DUP.get(r.get("tier"), 1))
    import random
    random.Random(0).shuffle(recs)
    return recs


def adaptive_nframes(rec, fps: float, cap: int, floor: int = 8) -> int:
    """Qwen3-VL native sampling: nframes = duration x fps, clipped, even."""
    dur = clip_duration(rel_to_repo(rec["video_path"])) or 16.0
    n = max(floor, min(cap, int(dur * fps)))
    return n if n % 2 == 0 else n + 1


def extract_frames_window(rec, n: int, size: int = 640):
    """Tile-aware extraction: uniform frames inside [video_start, video_end] of the ORIGINAL file,
    with metadata shifted so the processor emits LOCAL 0-based timestamps — the tile is presented
    exactly like a pre-trimmed clip (matches the test distribution and the local GT answers)."""
    import cv2
    import numpy as np
    from transformers.video_utils import VideoMetadata

    cap = cv2.VideoCapture(str(rel_to_repo(rec["video_path"])))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ws, we = rec["video_start"], rec["video_end"]
    i0, i1 = max(0, int(ws * vfps)), min(total - 1, max(1, int(we * vfps) - 1))
    n = min(n, max(2, i1 - i0 + 1))
    n = n if n % 2 == 0 else n - 1
    frames, used = [], []
    for i in np.linspace(i0, i1, n).round().astype(int).tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(_resize_native(fr, size), cv2.COLOR_BGR2RGB))
            used.append(i - i0)                       # shift -> local indices -> local timestamps
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in window: {rec['video_path']}")
    if len(used) % 2:
        frames, used = frames[:-1], used[:-1]
    win_frames = max(i1 - i0 + 1, used[-1] + 1)
    # speed augment: frames decoded at REAL positions, but the model perceives a scaled clock
    # (displayed_time = real_time / speed) via a scaled fps -> matches the speed-scaled GT in answer.
    disp_fps = vfps * rec.get("speed", 1.0)
    meta = VideoMetadata(total_num_frames=win_frames, fps=disp_fps, duration=win_frames / disp_fps,
                         frames_indices=used, video_backend="custom")
    return np.stack(frames), meta


def raw_seconds_target(rec) -> str:
    """Temporal target as raw-seconds JSON (model-native), e.g. {"start": 56.4, "end": 63.2}.
    Duration-aware parse (challenge 3-part GTs mix HH:MM:SS and MM:SS:cc)."""
    j = json.loads(rec["answer"])
    dur = clip_duration(rel_to_repo(rec["video_path"])) or 0.0
    return json.dumps({"start": round(to_sec_dur(str(j["start"]), dur), 2),
                       "end": round(to_sec_dur(str(j["end"]), dur), 2)})


def _displayed_grid(meta) -> list:
    """The discrete timestamp grid the Qwen3-VL processor actually prints per frame
    (frame_index / fps, rounded to 0.1s) — the only positions the model can 'point at'."""
    fps = meta.fps or 30.0
    return sorted(set(round(i / fps, 1) for i in meta.frames_indices))


def snap_seconds_target(rec, meta, win: float) -> str:
    """Temporal target snapped OUTWARD to the displayed frame grid (UniTime find_closest, outward):
    start -> nearest grid point <= start, end -> nearest grid point >= end. Guarantees the target
    lands on a timestamp the model has literally seen, preserves coverage, and nudges end later
    (mild counter to the early-end label bias). Movement is bounded by one grid step (<=0.2s)."""
    j = json.loads(rec["answer"])
    s = to_sec_dur(str(j["start"]), win)
    e = to_sec_dur(str(j["end"]), win)
    grid = _displayed_grid(meta)
    below = [g for g in grid if g <= s + 1e-6]
    above = [g for g in grid if g >= e - 1e-6]
    s_snap = max(below) if below else grid[0]
    e_snap = min(above) if above else grid[-1]
    if e_snap <= s_snap:                                  # degenerate (event inside one grid cell)
        i = grid.index(s_snap)
        e_snap = grid[min(i + 1, len(grid) - 1)]
    return json.dumps({"start": round(s_snap, 2), "end": round(e_snap, 2)})


def snap_seconds_target_nearest(rec, meta, win: float) -> str:
    """Temporal target snapped to the NEAREST displayed grid point (symmetric — no outward widening).
    Keeps the target grid-emittable like snap_seconds_target but preserves the speed-balanced tight
    duration (outward-snap would re-widen short events; see error_analysis duration bias)."""
    j = json.loads(rec["answer"])
    s = to_sec_dur(str(j["start"]), win)
    e = to_sec_dur(str(j["end"]), win)
    grid = _displayed_grid(meta)
    s_snap = min(grid, key=lambda g: abs(g - s))
    e_snap = min(grid, key=lambda g: abs(g - e))
    if e_snap <= s_snap:                                  # degenerate (event inside one grid cell)
        i = grid.index(s_snap)
        e_snap = grid[min(i + 1, len(grid) - 1)]
    return json.dumps({"start": round(s_snap, 2), "end": round(e_snap, 2)})


def augment_clip(arr):
    """Appearance-only, clip-consistent augmentation (label-preserving; never touches the time axis).
    Simulates the heterogeneous test distribution (270p-1080p sources, varied CCTV lighting/compression)."""
    import random as _r
    import cv2
    import numpy as np
    n, h, w, _ = arr.shape
    out = arr
    if _r.random() < 0.5:                                 # resolution jitter: downscale -> upscale
        f = _r.uniform(0.4, 1.0)
        sw, sh = max(8, int(w * f)), max(8, int(h * f))
        out = np.stack([cv2.resize(cv2.resize(fr, (sw, sh)), (w, h), interpolation=cv2.INTER_LINEAR)
                        for fr in out])
    if _r.random() < 0.5:                                 # JPEG compression
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), _r.randint(25, 90)]
        out = np.stack([cv2.cvtColor(cv2.imdecode(cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), enc)[1],
                                                  cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for fr in out])
    if _r.random() < 0.5:                                 # photometric: brightness/contrast/saturation
        br, co, sa = _r.uniform(0.75, 1.25), _r.uniform(0.8, 1.2), _r.uniform(0.7, 1.3)
        f32 = out.astype(np.float32) * br
        mean = f32.mean(axis=(1, 2), keepdims=True)
        f32 = (f32 - mean) * co + mean
        gray = f32.mean(axis=3, keepdims=True)
        f32 = gray + (f32 - gray) * sa
        out = np.clip(f32, 0, 255).astype(np.uint8)
    if _r.random() < 0.3:                                 # mild blur OR gaussian noise
        if _r.random() < 0.5:
            k = _r.choice([3, 5, 7])
            out = np.stack([cv2.GaussianBlur(fr, (k, k), 0) for fr in out])
        else:
            noise = np.random.normal(0, _r.uniform(3, 12), out.shape)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


_QWIN = re.compile(r"(\d+):(\d+)\.(\d+)")


def parse_qwin(q):
    """Parse the two mm:ss.xx timestamps embedded in temporal_description / causal_linkage questions."""
    m = _QWIN.findall(q or "")
    if len(m) < 2:
        return None
    s = int(m[0][0]) * 60 + int(m[0][1]) + int(m[0][2]) / 100
    e = int(m[1][0]) * 60 + int(m[1][1]) + int(m[1][2]) / 100
    return (s, e) if e > s else None


def build_batch(proc, rec, n_frames, sampling, use_cot, device, adaptive_fps=0.0, raw_seconds=False,
                budget_fill=False, base_frames=0, augment=False, no_snap=False, snap_nearest=False,
                box_cue=False, window_sample=False):
    win = None
    if "video_start" in rec:                          # tile record (temporal_tiles.jsonl)
        win = rec["video_end"] - rec["video_start"]
        n = n_frames if budget_fill else (adaptive_nframes(rec, adaptive_fps, n_frames)
                                          if adaptive_fps > 0 else n_frames)
        eff_fps = n / max(win, 0.5)
        size = 448 if eff_fps <= 8 else 320           # UniTime adaptive frame scaling: flat budget
        arr, meta = extract_frames_window(rec, n, size)
    else:                                             # non-tile (9 other tasks): lighter frame count
        nf = base_frames if base_frames > 0 else n_frames
        qwin = parse_qwin(rec["question"]) if window_sample and \
            rec["task"] in ("temporal_description", "causal_linkage") else None
        if qwin:                                       # dense frames over the Q event window (+2s context each side)
            s, e = qwin
            wrec = dict(rec); wrec["video_start"] = max(0.0, s - 2.0); wrec["video_end"] = e + 2.0
            arr, meta = extract_frames_window(wrec, nf)
        else:
            n = adaptive_nframes(rec, adaptive_fps, nf) if adaptive_fps > 0 else nf
            arr, meta = extract_frames(rec, n, sampling)
    if augment:                                       # appearance-only (label-preserving)
        arr = augment_clip(arr)
    legend = ""
    if box_cue and win is None:                       # box-cue only on non-tile (9 QA/desc tasks)
        mode = box_cue_mod.sample_mode(random)
        arr, drew = box_cue_mod.draw_box_cue(arr, meta.frames_indices, rec["video_path"], mode, random)
        if drew:
            legend = box_cue_mod.LEGEND
    if rec["task"] in ("temporal_presence", "temporal_tightness"):
        target = rec["answer"]                        # verify: thinking-free Yes/No (interval in question)
    elif raw_seconds and rec["task"] == "temporal_localization":
        if win is None or no_snap:                    # A: honest displayed GT (raw, no grid)
            target = raw_seconds_target(rec)
        elif snap_nearest:                            # C: nearest grid (aligned, no widening)
            target = snap_seconds_target_nearest(rec, meta, win)
        else:                                         # B: outward grid snap (default, widens)
            target = snap_seconds_target(rec, meta, win)
    else:
        target = make_target(rec, use_cot)
    user = {"role": "user", "content": [{"type": "video", "video": arr},
                                        {"type": "text", "text": legend + rec["question"]}]}
    plen = proc.apply_chat_template([user], tokenize=True, add_generation_prompt=True,
                                    return_dict=True, return_tensors="pt",
                                    video_metadata=[meta])["input_ids"].shape[1]
    full = proc.apply_chat_template(
        [user, {"role": "assistant", "content": target}],
        tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt",
        video_metadata=[meta])
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in full.items()}
    return batch, plen


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--tiers", default="S,A")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--sampling", choices=["smart", "uniform"], default="smart")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--cot", action="store_true")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--model", default=MODEL, help="HF model id (e.g. Qwen/Qwen3.5-9B)")
    ap.add_argument("--output-dir", default="checkpoints/qwen36_sft")
    ap.add_argument("--deepspeed", action="store_true", help="multi-GPU DeepSpeed ZeRO-3 (accelerate launch)")
    ap.add_argument("--resume-from", default=None, help="clean-resume: dir with state_latest (model+opt+sched+step)")
    ap.add_argument("--data", default=None, help="jsonl corpus override (e.g. training/data/temporal_mixed.jsonl)")
    ap.add_argument("--adaptive-fps", type=float, default=0.0,
                    help="Qwen-native sampling: nframes = dur x fps (cap = --n-frames); 0 = fixed --n-frames")
    ap.add_argument("--raw-seconds", action="store_true",
                    help="temporal target as raw-seconds JSON (answer-only, no think)")
    ap.add_argument("--base-frames", type=int, default=0,
                    help="frame count for NON-tile records (9 other tasks); 0 = use --n-frames")
    ap.add_argument("--budget-fill", action="store_true",
                    help="tile records always fill --n-frames (UniTime flat budget; fps varies, "
                         "resolution drops to 320px above 8fps)")
    ap.add_argument("--box-cue", action="store_true",
                    help="B2-robust: draw cached YOLO actor-boxes on frames as a soft cue with "
                         "dropout(30%%)/correct(55%%)/corrupt(15%%) so a weak/missing box never hurts")
    ap.add_argument("--window-sample", action="store_true",
                    help="temporal_description/causal_linkage: sample n_frames DENSELY over the Q event "
                         "window [start-2s, end+2s] (parsed from the mm:ss.xx pair) instead of uniform-full-video")
    ap.add_argument("--augment", action="store_true",
                    help="appearance-only clip-consistent augmentation (resolution/JPEG/photometric/blur; "
                         "label-preserving, never touches the time axis)")
    ap.add_argument("--no-snap", action="store_true",
                    help="temporal tiles: use honest raw-seconds GT (no outward grid-snap widening) — "
                         "for speed-balanced data whose answer is already the displayed-clock GT")
    ap.add_argument("--snap-nearest", action="store_true",
                    help="temporal tiles: snap GT to the NEAREST displayed grid point (aligned, symmetric, "
                         "no outward widening) — preferred for speed-balanced tight-duration data")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.deepspeed:
        run_deepspeed(args)
        return

    tasks = ALL_TASKS if args.tasks == "all" else set(args.tasks.split(","))
    records = load_data_jsonl(args.data, tasks) if args.data else \
        load_sft_records(tasks, set(args.tiers.split(",")), args.limit)
    logger.info("records=%d | tiers=%s | n_frames=%d | loading model...",
                len(records), args.tiers, args.n_frames)

    proc = make_processor(args.model)
    model = load_model(args.model); model.train()
    device = model.get_input_embeddings().weight.device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(args.max_steps * 0.03), args.max_steps)
    out_dir = Path(args.output_dir)

    max_steps = 2 if args.smoke else args.max_steps
    step, di, accum = 0, 0, 0.0
    opt.zero_grad()
    while step < max_steps:
        rec = records[di % len(records)]; di += 1
        try:
            batch, plen = build_batch(proc, rec, args.n_frames, args.sampling, args.cot, device,
                                      args.adaptive_fps, args.raw_seconds, args.budget_fill, args.base_frames,
                                      args.augment, args.no_snap, args.snap_nearest)
            labels = batch["input_ids"].clone(); labels[0, :plen] = -100
            loss = model(**batch, labels=labels, use_cache=False).loss / args.grad_accum
            loss.backward(); accum += loss.item()
        except Exception as e:
            logger.warning("skip %s: %s", rec.get("source"), str(e)[:120]); continue
        if di % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % args.logging_steps == 0:
                logger.info("step %d/%d loss=%.4f lr=%.2e task=%s", step, max_steps, accum,
                            sched.get_last_lr()[0], rec["task"])
            accum = 0.0
            if not args.smoke and step % args.save_steps == 0:
                d = out_dir / f"checkpoint-{step}"; d.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(d)); logger.info("saved -> %s", d)
    if not args.smoke:
        d = out_dir / f"checkpoint-{step}"; d.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(d))
    logger.info("Qwen3.6 SFT complete (%d steps, smoke=%s).", step, args.smoke)


if __name__ == "__main__":
    main()
