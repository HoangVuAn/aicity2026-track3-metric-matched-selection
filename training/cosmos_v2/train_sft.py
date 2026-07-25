"""SFT for the Cosmos3-Super reasoner (33.5B), LoRA. Three launch paths: single-process
naive model-parallel (default), multi-GPU DeepSpeed ZeRO-3 (--deepspeed), multi-GPU FSDP
(--fsdp). Library (model load, frame extraction, batch building) lives in trainlib.

  # single process (device_map=auto)
  HF_HOME=... CUDA_VISIBLE_DEVICES=0,3,4,5,6,7 cosmos3_env/bin/python \
    training/cosmos_v2/train_sft.py --config training/cosmos_v2/configs/sft.yaml --deepspeed
  # multi-GPU (accelerate): see training/cosmos_v2/train_sft.sh
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from transformers import get_cosine_schedule_with_warmup

COSMOS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COSMOS))
from trainlib.config import apply_config_defaults
from trainlib.data import ALL_TASKS, build_batch, load_sft_records
from trainlib.model import load_sft_model, make_processor
from trainlib.paths import LOGS

logger = logging.getLogger(__name__)


def _select(tasks: str) -> set[str]:
    return ALL_TASKS if tasks == "all" else set(tasks.split(","))


def _next_batch(proc, shard, di, args, device):
    while True:  # always return a decodable batch -> every rank does 1 backward/iter (lockstep)
        rec = shard[di[0] % len(shard)]; di[0] += 1
        try:
            return build_batch(proc, rec, args.n_frames, args.sampling, args.cot, device,
                                augment=args.augment), rec
        except Exception as e:
            logger.warning("skip %s: %s", rec.get("source"), str(e)[:80])


def run_deepspeed(args) -> None:
    """DeepSpeed ZeRO-3: the 33.5B base is partitioned across GPUs, each rank trains a shard."""
    from accelerate import Accelerator, DeepSpeedPlugin

    ds = DeepSpeedPlugin(zero_stage=3, gradient_accumulation_steps=args.grad_accum,
                         zero3_init_flag=True,  # partition weights AT LOAD (avoids N x full model in CPU RAM)
                         offload_optimizer_device="none", offload_param_device="none")
    acc = Accelerator(mixed_precision="bf16", deepspeed_plugin=ds)
    dsc = acc.state.deepspeed_plugin.deepspeed_config
    dsc["train_micro_batch_size_per_gpu"] = 1   # custom loop (no DataLoader)
    dsc.setdefault("zero_optimization", {})["stage3_gather_16bit_weights_on_model_save"] = True
    run = None
    if args.wandb and acc.is_main_process:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    logging.basicConfig(level=logging.INFO if acc.is_main_process else logging.ERROR,
                        format=f"%(asctime)s R{acc.process_index} %(levelname)s %(message)s", force=True)
    logger.setLevel(logging.INFO)
    LOGS.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOGS / "cosmos_ds_run.log", mode="w")  # survives stdout loss
    fh.setFormatter(logging.Formatter(f"%(asctime)s R{acc.process_index} %(message)s"))
    logger.addHandler(fh)

    records = load_sft_records(_select(args.tasks), set(args.tiers.split(",")), args.limit)
    shard = records[acc.process_index::acc.num_processes]
    logger.info("DeepSpeed ZeRO-3 | %d GPU | %d records | %d/rank | n_frames=%d | loading model...",
                acc.num_processes, len(records), len(shard), args.n_frames)

    proc = make_processor()
    model = load_sft_model(device_map=None); model.train()  # ZeRO-3 partitions it
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    model, opt = acc.prepare(model, opt)   # NOT the scheduler -> we step it ourselves
    # self-managed cosine schedule, stepped once per OPTIMIZER update (sync_gradients), so LR
    # anneals over exactly max_steps (the accelerate-wrapped scheduler stepped out of sync).
    sched = get_cosine_schedule_with_warmup(opt, int(args.max_steps * 0.03), args.max_steps)
    logger.info("prepared OK, starting training loop")

    out_dir = Path(args.output_dir)
    max_steps = 2 if args.smoke else args.max_steps
    di, step = [0], 0
    while step < max_steps:
        (batch, plen), rec = _next_batch(proc, shard, di, args, acc.device)
        labels = batch["input_ids"].clone(); labels[0, :plen] = -100
        loss = model(**batch, labels=labels, use_cache=False).loss
        acc.backward(loss)            # DeepSpeed handles grad-accum + ZeRO reduce
        opt.step(); opt.zero_grad()
        if acc.sync_gradients:        # a real optimizer step happened
            sched.step(); step += 1   # step LR once per real update -> proper anneal
            if step % args.logging_steps == 0 and acc.is_main_process:
                logger.info("step %d/%d loss=%.4f lr=%.2e", step, max_steps, loss.item(), sched.get_last_lr()[0])
                if run is not None:
                    run.log({"loss": loss.item(), "lr": sched.get_last_lr()[0], "step": step})
            if not args.smoke and step % args.save_steps == 0:
                acc.wait_for_everyone()
                acc.unwrap_model(model).save_pretrained(
                    str(out_dir / f"checkpoint-{step}"), is_main_process=acc.is_main_process,
                    save_function=acc.save, state_dict=acc.get_state_dict(model))
                if acc.is_main_process:
                    logger.info("saved -> checkpoint-%d", step)
    if acc.is_main_process:
        logger.info("DeepSpeed SFT complete (%d steps, smoke=%s).", step, args.smoke)
        if run is not None:
            run.finish()


def run_fsdp(args) -> None:
    """FSDP FULL_SHARD: the 33.5B base is sharded across GPUs, ranks compute different samples."""
    import functools

    from accelerate import Accelerator, FullyShardedDataParallelPlugin
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (Qwen3VLTextDecoderLayer,
                                                                Qwen3VLVisionBlock)

    # explicit callable policy (string config didn't engage on accelerate 1.13 -> no shard -> OOM)
    policy = functools.partial(transformer_auto_wrap_policy,
                               transformer_layer_cls={Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock})
    fsdp = FullyShardedDataParallelPlugin(
        fsdp_version=1, sharding_strategy="FULL_SHARD", auto_wrap_policy=policy,
        use_orig_params=True,            # required: frozen base + trainable LoRA mix
        sync_module_states=True, state_dict_type="FULL_STATE_DICT")
    acc = Accelerator(mixed_precision="bf16", fsdp_plugin=fsdp, gradient_accumulation_steps=args.grad_accum)
    logging.basicConfig(level=logging.INFO if acc.is_main_process else logging.ERROR,
                        format=f"%(asctime)s R{acc.process_index} %(levelname)s %(message)s", force=True)
    logger.setLevel(logging.INFO if acc.is_main_process else logging.ERROR)

    records = load_sft_records(_select(args.tasks), set(args.tiers.split(",")), args.limit)
    shard = records[acc.process_index::acc.num_processes]
    if acc.is_main_process:
        logger.info("FSDP | %d GPU | %d records | %d/rank | n_frames=%d", acc.num_processes,
                    len(records), len(shard), args.n_frames)

    proc = make_processor()
    model = load_sft_model(device_map=None); model.train()  # FSDP shards it
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(args.max_steps * 0.03), args.max_steps)
    model, opt = acc.prepare(model, opt)

    out_dir = Path(args.output_dir)
    max_steps = 2 if args.smoke else args.max_steps
    di, step = [0], 0
    while step < max_steps:
        with acc.accumulate(model):
            (batch, plen), rec = _next_batch(proc, shard, di, args, acc.device)
            labels = batch["input_ids"].clone(); labels[0, :plen] = -100
            loss = model(**batch, labels=labels, use_cache=False).loss
            acc.backward(loss)
            if acc.sync_gradients:
                acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        if acc.sync_gradients:
            sched.step(); step += 1
            if step % args.logging_steps == 0 and acc.is_main_process:
                logger.info("step %d/%d loss=%.4f lr=%.2e", step, max_steps, loss.item(), sched.get_last_lr()[0])
            if not args.smoke and step % args.save_steps == 0:
                acc.wait_for_everyone()
                if acc.is_main_process:
                    acc.unwrap_model(model).save_pretrained(str(out_dir / f"checkpoint-{step}"))
                    logger.info("saved -> checkpoint-%d", step)
    acc.wait_for_everyone()
    if acc.is_main_process and not args.smoke:
        acc.unwrap_model(model).save_pretrained(str(out_dir / f"checkpoint-{step}"))
    if acc.is_main_process:
        logger.info("FSDP SFT complete (%d steps, smoke=%s).", step, args.smoke)


def run_single(args) -> None:
    """Single process, device_map=auto (naive model-parallel across visible GPUs)."""
    LOGS.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOGS / f"{args.run_name}.log", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    records = load_sft_records(_select(args.tasks), set(args.tiers.split(",")), args.limit)
    logger.info("records=%d | tiers=%s | wandb=%s", len(records), args.tiers, args.wandb)

    proc = make_processor()
    model = load_sft_model(); model.train()
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
            batch, plen = build_batch(proc, rec, args.n_frames, args.sampling, args.cot, device)
            labels = batch["input_ids"].clone(); labels[0, :plen] = -100
            loss = model(**batch, labels=labels, use_cache=False).loss / args.grad_accum
            loss.backward(); accum += loss.item()
        except Exception as e:
            logger.warning("skip %s: %s", rec.get("id"), str(e)[:120]); continue
        if di % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % args.logging_steps == 0:
                logger.info("step %d/%d loss=%.4f lr=%.2e task=%s", step, max_steps, accum,
                            sched.get_last_lr()[0], rec["task"])
                if run is not None:
                    run.log({"loss": accum, "lr": sched.get_last_lr()[0], "step": step})
            accum = 0.0
            if not args.smoke and step % args.save_steps == 0:
                d = out_dir / f"checkpoint-{step}"; d.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(d)); logger.info("saved -> %s", d)
    if not args.smoke:
        d = out_dir / f"checkpoint-{step}"; d.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(d))
    logger.info("SFT complete (%d steps, smoke=%s).", step, args.smoke)
    if run is not None:
        run.finish()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML of defaults (training/cosmos_v2/configs/sft.yaml)")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--tiers", default="S,A")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-frames", type=int, default=64)
    ap.add_argument("--sampling", choices=["smart", "uniform"], default="smart")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cot", action="store_true", help="train on <think> reasoning + answer")
    ap.add_argument("--augment", action="store_true", help="appearance-only clip augmentation (CCTV robustness)")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--output-dir", default="checkpoints/cosmos_sft")
    ap.add_argument("--smoke", action="store_true", help="2 steps, no save")
    ap.add_argument("--fsdp", action="store_true", help="multi-GPU FSDP (accelerate launch)")
    ap.add_argument("--deepspeed", action="store_true", help="multi-GPU DeepSpeed ZeRO-3 (accelerate launch)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="cosmos-aicity")
    ap.add_argument("--run-name", default="cosmos_sft")
    apply_config_defaults(ap)
    args = ap.parse_args()

    if args.deepspeed:
        run_deepspeed(args)
    elif args.fsdp:
        run_fsdp(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
