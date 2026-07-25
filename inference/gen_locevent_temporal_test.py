"""locevent grounding on TEST for temporal_description: inject the SAME video's temporal_localization
QUESTION (event phrase, answer-format stripped) into the temporal_description gen prompt. This beat
entity-grounding on val (+0.0461 vs +0.0203, n=60 vs 22). temporal_localization Q is present in the test
set (task dropped from scoring, but its question text remains) -> legitimate cross-task grounding. cosmos3_env -> f1."""
import os
import re
import sys
import json
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import openai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TEST_JSON, TEST_VIDEO_BASE  # noqa: E402


def locphrase(q):
    return re.split(r"Answer with|Provide the result|Provide the", q)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--task", default="temporal_description",
                    help="target task to locevent-ground (default temporal_description; causal_linkage used only "
                         "for the selectivity ablation -- the deployed pipeline grounds causal with an evidence pack)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run/locevent_temporal_test.json"))
    a = ap.parse_args()
    cli = openai.OpenAI(api_key="EMPTY", base_url=a.base_url, timeout=300, max_retries=6)
    model = cli.models.list().data[0].id
    items = json.load(open(TEST_JSON))["items"]
    byv = defaultdict(dict)
    for it in items:
        byv[it["video_id"]][it["task_type"]] = it
    todo_items = []
    for it in items:
        if it["task_type"] != a.task:
            continue
        loc = byv[it["video_id"]].get("temporal_localization")
        if not loc:
            continue
        it = dict(it)
        it["_loc"] = locphrase(loc["question"])
        todo_items.append(it)
    print(f"locevent test gen: {len(todo_items)} {a.task} items with loc-Q, n={a.n}, model={model}", flush=True)

    def vurl(vid):
        return "file://" + str((TEST_VIDEO_BASE / vid).resolve())

    def gq(it):
        return (f"Note: this segment shows: {it['_loc']}. Describe what happened, consistent with this event.\n\n"
                + it["question"])

    def one(it):
        try:
            r = cli.chat.completions.create(model=model, max_tokens=512, temperature=0.7, top_p=0.95, n=a.n, seed=0,
                messages=[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": vurl(it["video_id"])}},
                    {"type": "text", "text": gq(it)}]}])
            samples = [c.message.content or "" for c in r.choices]
        except Exception as e:
            print("fail", it["item_index"], str(e)[:80], flush=True)
            samples = []
        return str(it["item_index"]), {"task": a.task, "samples": samples}

    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    todo = [it for it in todo_items if str(it["item_index"]) not in out]
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (iid, d) in enumerate(ex.map(one, todo)):
            out[iid] = d
            if (i + 1) % 20 == 0:
                json.dump(out, open(a.out, "w")); print(f"{i+1}/{len(todo)}", flush=True)
    json.dump(out, open(a.out, "w"))
    print(f"GEN DONE -> {a.out} ({len(out)} items)", flush=True)


if __name__ == "__main__":
    main()
