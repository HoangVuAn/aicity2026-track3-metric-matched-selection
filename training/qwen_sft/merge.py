"""Merge a Qwen3.6-27B LoRA adapter into the base and save a standalone checkpoint vLLM can serve.
  CUDA_VISIBLE_DEVICES=3 cosmos3_env/bin/python training/qwen_sft/merge.py <adapter> <out>"""
import sys
import warnings

warnings.filterwarnings("ignore")
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

ADAPTER, OUT = sys.argv[1], sys.argv[2]
BASE = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3.6-27B"
print("loading base + adapter (GPU)...", flush=True)
base = AutoModelForImageTextToText.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
print("merging...", flush=True)
merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
merged.save_pretrained(OUT, safe_serialization=True)
AutoProcessor.from_pretrained(BASE, trust_remote_code=True).save_pretrained(OUT)
print("merged ->", OUT, flush=True)
