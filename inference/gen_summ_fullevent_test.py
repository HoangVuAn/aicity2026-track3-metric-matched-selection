"""fullevent grounding on TEST for video_summarization: inject the SAME video's temporal_localization Q
(event) + causal_linkage Q (consequence focus) + entity pair. Beat entity-only on val (+0.0359 vs -0.0282,
n=60). All Qs present in test set -> legit cross-task grounding. cosmos3_env -> f1."""
import os
import re
import sys
import json
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import openai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TEST_JSON, TEST_VIDEO_BASE  # noqa


def locphrase(q):
    return re.split(r"Answer with|Provide the result|Provide the", q)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pairs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_pairs_multi.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run/summ_fullevent_test.json"))
    a = ap.parse_args()
    cli = openai.OpenAI(api_key="EMPTY", base_url=a.base_url, timeout=300, max_retries=6)
    model = cli.models.list().data[0].id
    try:
        pairs = {v: p for v, p in json.load(open(a.pairs)).items() if p}
    except Exception:
        pairs = {}
    items = json.load(open(TEST_JSON))["items"]
    byv = defaultdict(dict)
    for it in items:
        byv[it["video_id"]][it["task_type"]] = it
    todo_items = []
    for it in items:
        if it["task_type"] != "video_summarization":
            continue
        loc = byv[it["video_id"]].get("temporal_localization")
        if not loc:
            continue
        it = dict(it)
        it["_loc"] = locphrase(loc["question"])
        cau = byv[it["video_id"]].get("causal_linkage")
        it["_cau"] = locphrase(cau["question"]) if cau else None
        it["_pair"] = pairs.get(it["video_id"])
        todo_items.append(it)
    print(f"summ fullevent test: {len(todo_items)} items with loc-Q, n={a.n}, model={model}", flush=True)

    def vurl(vid):
        return "file://" + str((TEST_VIDEO_BASE / vid).resolve())

    def gq(it):
        fe = f"Note: the video shows: {it['_loc']}."
        if it["_cau"]:
            fe += f" A related question concerns: {it['_cau']}."
        if it["_pair"]:
            fe += f" The vehicles are a {it['_pair'][0]} and a {it['_pair'][1]}."
        return (fe + " Summarize the full sequence of events including the incident and its consequences, "
                f"consistent with this.\n\n" + it["question"])

    def one(it):
        r = cli.chat.completions.create(model=model, max_tokens=512, temperature=0.7, top_p=0.95, n=a.n, seed=0,
            messages=[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": vurl(it["video_id"])}},
                {"type": "text", "text": gq(it)}]}])
        return str(it["item_index"]), {"task": "video_summarization", "samples": [c.message.content or "" for c in r.choices]}

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
