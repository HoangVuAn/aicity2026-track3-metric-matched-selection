"""From-scratch regeneration of the f1 base predictions (fixes Finding 3): greedy vLLM inference with the
merged f1 model over ALL test items (128 frames, temp 0), parse <answer>. This is the f1base / fallback +
temporal_localization source that assemble_clean.py consumes. Resume-safe (skips item_index already written)."""
from __future__ import annotations
import os, csv, sys, json, argparse

import openai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TEST_JSON, TEST_VIDEO_BASE  # noqa: E402
from inference import parse_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--out", default="inference/full_run/pred_f1_128_ck3600.csv")
    a = ap.parse_args()
    cli = openai.OpenAI(api_key="EMPTY", base_url=a.base_url, timeout=300, max_retries=4)
    model = cli.models.list().data[0].id
    items = json.load(open(TEST_JSON))["items"]

    done = {}
    if os.path.exists(a.out):
        done = {r["item_index"]: r["prediction"] for r in csv.DictReader(open(a.out))}
        print(f"resume: {len(done)} already done", flush=True)

    def url(vid):
        return "file://" + str((TEST_VIDEO_BASE / vid).resolve())

    def flush():
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["item_index", "prediction"])
            for k, v in done.items():
                w.writerow([k, v])

    print(f"{len(items)} test items | greedy 128f | {model}", flush=True)
    for i, it in enumerate(items):
        iid = str(it["item_index"])
        if iid in done:
            continue
        try:
            r = cli.chat.completions.create(
                model=model, max_tokens=a.max_new, temperature=0.0, n=1,
                messages=[{"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": url(it["video_id"])}},
                    {"type": "text", "text": it["question"]}]}])
            done[iid] = parse_answer(r.choices[0].message.content or "")
        except Exception as e:
            print(f"fail {iid}: {str(e)[:80]}", flush=True)
            done[iid] = ""
        if (i + 1) % 20 == 0:
            flush(); print(f"  {i + 1}/{len(items)}", flush=True)
    flush()
    empty = sum(1 for v in done.values() if not v.strip())
    print(f"DONE: wrote {len(done)} preds ({empty} empty) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
