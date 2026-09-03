"""Extra option-order permutations for mcq / mcq_oe (compliant rerun, award review).

The shipped vote script samples 4 cyclic permutations of the four options per model. With the
task-specialised checkpoints removed the pick pool shrinks, so we widen the *same* mechanism the
paper already uses — test-time augmentation over option order — with 8 further permutations that
are not cyclic rotations of each other. Output merges with the votetta pools by item_index.

  python inference/gen_extra_perms.py --base-url http://localhost:9311/v1 \
      --out inference/full_run_compliant/extra_perms_cosmos_v2.json
"""
import os
import re
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor

import openai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TEST_JSON, TEST_VIDEO_BASE  # noqa: E402

LET = ["A", "B", "C", "D"]
# 8 permutations that are NOT the cyclic rotations already covered by gen_cosmos_votetta.py
EXTRA_PERMS = [["D", "C", "B", "A"], ["C", "B", "A", "D"], ["B", "A", "D", "C"], ["A", "D", "C", "B"],
               ["A", "B", "D", "C"], ["B", "A", "C", "D"], ["C", "A", "D", "B"], ["D", "B", "C", "A"]]


def parse_answer(s):
    s = s or ""
    m = re.search(r"<answer>(.*?)</answer>", s, re.S)
    if m:
        return m.group(1).strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    return re.sub(r"^.*?</think>", "", s, flags=re.S).strip()


def extract_letter(s):
    m = re.search(r"\b([A-D])\b", parse_answer(s).upper())
    return m.group(1) if m else None


def parse_opts(q):
    m = re.search(r"(?m)^\s*A[.)]\s", q)
    if not m:
        return None
    parts = re.split(r"(?m)^\s*([A-D])[.)]\s+", q[m.start():]); o = {}
    for k in range(1, len(parts) - 1, 2):
        o[parts[k]] = parts[k + 1]
    if set(o) != set(LET):
        return None
    o["D"] = re.split(r"\n\s*\n|\bChoose the correct\b|\bAnswer with\b", o["D"], maxsplit=1)[0]
    return {k: " ".join(v.split()).strip().rstrip("?").strip() for k, v in o.items()}


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower())[:55]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cli = openai.OpenAI(api_key="EMPTY", base_url=a.base_url)
    model = cli.models.list().data[0].id
    items = [it for it in json.load(open(TEST_JSON))["items"]
             if it["task_type"] in ("mcq", "mcq_openended")]
    print(f"extra perms: {len(items)} items x {len(EXTRA_PERMS)} perms, model={model}", flush=True)

    def url(vid):
        return "file://" + os.path.join(TEST_VIDEO_BASE, vid)

    def one(it):
        o = parse_opts(it["question"])
        if not o:
            return str(it["item_index"]), {"task": it["task_type"], "picks": []}
        stem = re.split(r"(?m)^\s*A[.)]\s", it["question"])[0]
        picks = []
        for perm in EXTRA_PERMS:
            body = "\n".join(f"{LET[j]}. {o[perm[j]]}" for j in range(4))
            q2 = stem + body + "\nChoose the correct option. Answer with the letter."
            try:
                r = cli.chat.completions.create(
                    model=model, max_tokens=512, temperature=0.0, top_p=1.0, seed=0, n=1,
                    messages=[{"role": "user", "content": [
                        {"type": "video_url", "video_url": {"url": url(it["video_id"])}},
                        {"type": "text", "text": q2}]}])
                la = extract_letter(r.choices[0].message.content or "")
                if la in LET:
                    picks.append(norm(o[perm[LET.index(la)]]))
            except Exception as e:  # keep going; a missing pick just shrinks this item's pool
                print(f"  warn {it['item_index']}: {e}", flush=True)
        return str(it["item_index"]), {"task": it["task_type"], "picks": picks}

    out = {}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, (k, v) in enumerate(ex.map(one, items)):
            out[k] = v
            if (n + 1) % 20 == 0:
                json.dump(out, open(a.out, "w")); print(f"{n+1}/{len(items)}", flush=True)
    json.dump(out, open(a.out, "w"))
    print(f"GEN DONE -> {a.out} ({len(out)} items)", flush=True)


if __name__ == "__main__":
    main()
