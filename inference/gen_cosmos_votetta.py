"""Cosmos (raw, no box) vote/TTA on TEST all 4 choice tasks: bcq SC-vote (N samples), mcq option-perm TTA,
bcq_oe N samples (for medoid+stance), mcq_oe option-perm TTA. Serve nvidia/Cosmos3-Super (port 9190,
num_frames=32). Saves per-item for the cross-format assembly."""
import os
import re
import sys
import json
import argparse

import openai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TEST_JSON, TEST_VIDEO_BASE  # noqa: E402
from inference import parse_answer, extract_yesno, extract_letter  # noqa: E402
LET = ["A", "B", "C", "D"]


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


def norm(s): return re.sub(r"[^a-z0-9 ]", "", str(s).lower())[:55]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:9190/v1")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run/cosmos_votetta.json"))
    a = ap.parse_args()
    cli = openai.OpenAI(api_key="EMPTY", base_url=a.base_url, timeout=240, max_retries=0)
    model = cli.models.list().data[0].id
    items = json.load(open(TEST_JSON))["items"]
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}

    def url(vid):
        return "file://" + str((TEST_VIDEO_BASE / vid).resolve())

    def ask(vid, q, n, temp):
        r = cli.chat.completions.create(model=model, max_tokens=512, temperature=temp, top_p=0.95, seed=0, n=n,
            messages=[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": url(vid)}},
                {"type": "text", "text": q}]}])
        return [parse_answer(c.message.content or "") for c in r.choices]

    def perm_picks(it, o):
        stem = re.split(r"(?m)^\s*A[.)]\s", it["question"])[0]
        picks = []
        for perm in [["A", "B", "C", "D"], ["B", "C", "D", "A"], ["C", "D", "A", "B"], ["D", "A", "B", "C"]]:
            body = "\n".join(f"{LET[j]}. {o[perm[j]]}" for j in range(4))
            q2 = stem + body + "\nChoose the correct option. Answer with the letter."
            la = extract_letter(ask(it["video_id"], q2, 1, 0.0)[0])
            if la in LET:
                picks.append(norm(o[perm[LET.index(la)]]))
        return picks

    tasks = {"bcq": [it for it in items if it["task_type"] == "bcq"],
             "mcq": [it for it in items if it["task_type"] == "mcq"],
             "bcq_openended": [it for it in items if it["task_type"] == "bcq_openended"],
             "mcq_openended": [it for it in items if it["task_type"] == "mcq_openended"]}
    print("cosmos gen | " + " ".join(f"{k}:{len(v)}" for k, v in tasks.items()) + f" | {model}", flush=True)

    def run(task, fn):
        lst = tasks[task]
        for i, it in enumerate(lst):
            iid = str(it["item_index"])
            if iid in out and out[iid].get("done"):
                continue
            try:
                out[iid] = fn(it); out[iid]["done"] = True
            except Exception as e:
                out[iid] = {"task": task, "err": str(e)[:60], "done": True}
            if (i + 1) % 20 == 0:
                json.dump(out, open(a.out, "w")); print(f"{task} {i+1}/{len(lst)}", flush=True)
        json.dump(out, open(a.out, "w"))

    run("bcq", lambda it: {"task": "bcq", "votes": [extract_yesno(s) for s in ask(it["video_id"], it["question"], a.n, 0.8) if extract_yesno(s)]})
    run("bcq_openended", lambda it: {"task": "bcq_openended", "samples": ask(it["video_id"], it["question"], a.n, 0.8)})
    run("mcq", lambda it: {"task": "mcq", "picks": perm_picks(it, parse_opts(it["question"])) if parse_opts(it["question"]) else []})
    run("mcq_openended", lambda it: {"task": "mcq_openended", "picks": perm_picks(it, parse_opts(it["question"])) if parse_opts(it["question"]) else []})
    print("DONE ->", a.out, flush=True)


if __name__ == "__main__":
    main()
