"""Merge a trained LoRA adapter into the Cosmos3-Super reasoner and save a standalone
checkpoint vLLM can serve (fast test inference). Copies the tokenizer/processor/chat-template
files from the original snapshot so vLLM loads it like the base.

  cosmos3_env/bin/python training/cosmos_v2/merge_lora.py \
    --adapter checkpoints/cosmos_sft_ds_v2/checkpoint-3800 --out checkpoints/cosmos_sft_merged
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from peft import PeftModel

COSMOS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COSMOS))
from trainlib.model import load_base
from trainlib.paths import SNAP

KEEP = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "added_tokens.json", "special_tokens_map.json", "chat_template.json",
        "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("loading base + adapter...", flush=True)
    base = load_base(device_map="cpu")
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    print("merged weights saved", flush=True)
    for name in KEEP:
        src = Path(SNAP) / name
        if src.exists():
            shutil.copy2(src, out / name)
    print(f"tokenizer/processor copied -> {out}", flush=True)


if __name__ == "__main__":
    main()
