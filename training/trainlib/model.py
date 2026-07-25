"""Cosmos3-Super reasoner (33.5B) loading: processor, fresh-LoRA model (SFT),
adapter-loaded model (eval), and LoRA merge. Centralizes the Qwen3VLConfig
boilerplate that was duplicated across the SFT/eval/merge scripts."""

from __future__ import annotations

import json

import torch
import transformers
import transformers_cosmos3  # noqa: F401  registers the Cosmos3 architecture
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen3VLConfig
from transformers_cosmos3.model import Cosmos3ForConditionalGeneration

from trainlib.paths import SNAP

# the processor logs a kwargs warning on EVERY call (video_metadata path); over a long
# run that output flood can choke a backgrounded process -> silence transformers logging.
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_config() -> Qwen3VLConfig:
    cd = json.load(open(SNAP + "/config.json"))
    return Qwen3VLConfig(
        text_config=cd["text_config"], vision_config=cd["vision_config"],
        image_token_id=cd["image_token_id"], video_token_id=cd["video_token_id"],
        vision_start_token_id=cd["vision_start_token_id"],
        vision_end_token_id=cd["vision_end_token_id"],
        tie_word_embeddings=cd["tie_word_embeddings"])


def make_processor() -> AutoProcessor:
    proc = AutoProcessor.from_pretrained(SNAP, trust_remote_code=True)
    proc.video_processor.do_sample_frames = False  # we pre-sample frames ourselves
    return proc


def load_base(device_map: str | None = "auto"):
    return Cosmos3ForConditionalGeneration.from_pretrained(
        SNAP, config=build_config(), dtype=torch.bfloat16,
        device_map=device_map, trust_remote_code=True)


def _bf16_trainable(model) -> None:
    # FSDP/ZeRO flatten params per unit and require a uniform dtype, but PEFT creates LoRA
    # layers in fp32 while the base is bf16 -> cast trainable params to bf16.
    for p in model.parameters():
        if p.requires_grad and p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)


def load_sft_model(device_map: str | None = "auto", r: int = 64, alpha: int = 128):
    """Fresh LoRA on the base, for SFT. device_map=None for FSDP/DeepSpeed (accelerate
    shards), "auto" for single-process naive model-parallel."""
    model = load_base(device_map)
    # use_reentrant=True under ZeRO-3 (reentrant=False mismatches recomputed tensor metadata
    # when ZeRO-3 gathers/releases params -> CheckpointError); works with input require_grads.
    reentrant = device_map is None
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    model.enable_input_require_grads()
    lora = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=LORA_TARGETS)
    model = get_peft_model(model, lora)
    if device_map is None:
        _bf16_trainable(model)
    else:
        model.print_trainable_parameters()
    return model


def load_adapter(adapter: str, device_map: str | None = "auto", trainable: bool = True):
    """Load a saved LoRA adapter onto the base, for eval / merge (frozen)."""
    base = load_base(device_map)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.enable_input_require_grads()
        if device_map is None:
            _bf16_trainable(model)
    return model
