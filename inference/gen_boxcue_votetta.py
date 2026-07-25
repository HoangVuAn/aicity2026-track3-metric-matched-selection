"""Box-cue bcq (SC-vote, N samples) + mcq (option-permutation TTA) on TEST, using cached boxed mp4s.
Serve qwen36_27b_bcqmcq_boxcue_merged (port 9182). Saves per-item {bcq: [yes/no samples], mcq: [perm picks]}
for the clean vote/TTA assembly."""
import os
import re
import sys
import json
import random
import argparse

import openai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TEST_JSON  # noqa: E402
from inference import parse_answer, extract_yesno, extract_letter  # noqa: E402
import box_cue as bc  # noqa: E402

MP4 = os.getenv("BOXCUE_MP4_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run/boxcue_mp4"))
LET = ["A", "B", "C", "D"]


def parse_opts(q):
    m = re.search(r"(?m)^\s*A[.)]\s", q)
    if not m:
        return None
    parts = re.split(r"(?m)^\s*([A-D])[.)]\s+", q[m.start():])
    o = {}
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
    ap.add_argument("--base-url", default="http://localhost:9182/v1")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run/boxcue_votetta.json"))
    a = ap.parse_args()
    cli = openai.OpenAI(api_key="EMPTY", base_url=a.base_url, timeout=150, max_retries=0)
    model = cli.models.list().data[0].id
    items = json.load(open(TEST_JSON))["items"]
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}

    def mp4url(vid):
        return "file://" + os.path.abspath(f"{MP4}/{str(vid).replace('/', '__')}.mp4")

    def ask(vid, q, n, temp):
        r = cli.chat.completions.create(model=model, max_tokens=400, temperature=temp, top_p=0.95, seed=0, n=n,
            messages=[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": mp4url(vid)}},
                {"type": "text", "text": bc.LEGEND + q}]}])
        return [parse_answer(c.message.content or "") for c in r.choices]

    bcq = [it for it in items if it["task_type"] == "bcq"]
    mcq = [it for it in items if it["task_type"] == "mcq"]
    print(f"bcq {len(bcq)} (SC n={a.n}) | mcq {len(mcq)} (TTA 4-perm) | model {model}", flush=True)

    for i, it in enumerate(bcq):
        iid = str(it["item_index"])
        if iid in out:
            continue
        try:
            samples = ask(it["video_id"], it["question"], a.n, 0.8)
            out[iid] = {"task": "bcq", "votes": [extract_yesno(s) for s in samples if extract_yesno(s)]}
        except Exception as e:
            out[iid] = {"task": "bcq", "votes": [], "err": str(e)[:60]}
        if (i + 1) % 30 == 0:
            json.dump(out, open(a.out, "w")); print(f"bcq {i+1}/{len(bcq)}", flush=True)
    json.dump(out, open(a.out, "w"))

    for i, it in enumerate(mcq):
        iid = str(it["item_index"])
        if iid in out:
            continue
        o = parse_opts(it["question"])
        if not o:
            out[iid] = {"task": "mcq", "picks": []}
            continue
        stem = re.split(r"(?m)^\s*A[.)]\s", it["question"])[0]
        picks = []
        try:
            for perm in [["A", "B", "C", "D"], ["B", "C", "D", "A"], ["C", "D", "A", "B"], ["D", "A", "B", "C"]]:
                # build a shuffled-option question (re-lettered A..D)
                body = "\n".join(f"{LET[j]}. {o[perm[j]]}" for j in range(4))
                q2 = stem + body + "\nChoose the correct option. Answer with the letter."
                ans = ask(it["video_id"], q2, 1, 0.0)[0]
                la = extract_letter(ans)
                if la in LET:
                    picks.append(norm(o[perm[LET.index(la)]]))  # map new-letter -> original content
            out[iid] = {"task": "mcq", "picks": picks, "opts": {k: norm(v) for k, v in o.items()}}
        except Exception as e:
            out[iid] = {"task": "mcq", "picks": [], "err": str(e)[:60]}
        if (i + 1) % 20 == 0:
            json.dump(out, open(a.out, "w")); print(f"mcq {i+1}/{len(mcq)}", flush=True)
    json.dump(out, open(a.out, "w"))
    print("DONE ->", a.out, flush=True)


if __name__ == "__main__":
    main()
