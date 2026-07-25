"""TEST gen for the 3 val-winning evidence levers: causal<-mcq-option (+0.0483), open_qa<-evidence (+0.0295),
bcq_oe No-polarity regen (+0.0283; Yes items untouched). Evidence = temporal_loc Q phrase + OUR submitted mcq
option text (letter->text) + entity pair. bcq_oe polarity taken from OUR OWN bcq predictions (--sub, the
vote output), only items whose question EXACTLY matches a bcq question and whose bcq answer is No.
--sub must be our own prediction file (submission_clean_repro), never a GT-derived one. cosmos3_env -> f1 serve."""
import os
import re
import sys
import json
import csv
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import openai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TEST_JSON, TEST_VIDEO_BASE  # noqa

_OPT = re.compile(r"\b([A-E])\)\s*(.+?)(?=\n\s*[A-E]\)|\Z)", re.S)


def stripfmt(q):
    return re.split(r"Answer with|Provide the result|Provide the", q)[0].strip()


def opt_text(question, letter):
    letter = (letter or "").strip().upper()[:1]
    for lt, tx in _OPT.findall(question):
        if lt == letter:
            return re.sub(r"\s+", " ", tx).strip()[:220]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sub", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "../dataset/official_test/submission_clean_repro.csv"),
                    help="our own vote output (mcq pick + bcq); NEVER a GT-derived / pair-repaired file")
    ap.add_argument("--pairs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_pairs_multi.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run/evidence_test.json"))
    a = ap.parse_args()
    cli = openai.OpenAI(api_key="EMPTY", base_url=a.base_url, timeout=300, max_retries=6)
    model = cli.models.list().data[0].id
    items = {str(it["item_index"]): it for it in json.load(open(TEST_JSON))["items"]}
    sub = {r["item_index"]: r["prediction"] for r in csv.DictReader(open(a.sub))}
    try:
        pairs = {v: p for v, p in json.load(open(a.pairs)).items() if p}
    except Exception:
        pairs = {}
    byv = defaultdict(lambda: defaultdict(list))
    for i, it in items.items():
        byv[it["video_id"]][it["task_type"]].append(i)

    def evidence(v):
        d = byv[v]
        loc = stripfmt(items[d["temporal_localization"][0]]["question"]) if d.get("temporal_localization") else ""
        mco = ""
        if d.get("mcq"):
            mi = d["mcq"][0]
            mco = opt_text(items[mi]["question"], sub.get(mi, ""))
        return loc, mco, pairs.get(v)

    def ev_prefix(v):
        loc, mco, pr = evidence(v)
        parts = []
        if loc:
            parts.append(f"the video shows: {loc}")
        if mco:
            parts.append(f"what happened: {mco}")
        if pr:
            parts.append(f"the vehicles are a {pr[0]} and a {pr[1]}")
        return ("Context: " + "; ".join(parts) + ".") if parts else ""

    recs = []
    for v, d in byv.items():
        loc, mco, pr = evidence(v)
        for i in d.get("open_qa", []):
            if loc or mco:
                recs.append((i, "open_qa", f"{ev_prefix(v)} Answer the question consistent with this context.\n\n"
                             + items[i]["question"], v))
        for i in d.get("causal_linkage", []):
            if mco:
                recs.append((i, "causal_linkage", f"Context: the root cause is: {mco}. Keep your answer focused "
                             f"on the cause-and-effect relationship, consistent with this.\n\n"
                             + items[i]["question"], v))
        # bcq_oe: No-polarity evidence for EVERY paired bcq_oe (not just base-No). This lets the assemble step's
        # bcq_oe sync inherit the FINAL bcq polarity -- including items that pair-repair later flips Yes->No,
        # whose No-evidence would otherwise be missing (evidence is generated before pair-repair). Unused when
        # the final bcq is Yes (sync only consumes bcq_oe_no when should=="no"), so over-generating is harmless.
        bans = {items[j]["question"].split("\n")[0]: sub.get(j, "").strip().lower() for j in d.get("bcq", [])}
        for i in d.get("bcq_openended", []):
            q0 = items[i]["question"].split("\n")[0]
            if q0 in bans:
                recs.append((i, "bcq_oe_no", f"{ev_prefix(v)} The correct answer is No: the queried collision "
                             f"does NOT occur. Answer in the style: 'No. The actual collision involves ... , "
                             f"not ...' using the true vehicles from the context.\n\n" + items[i]["question"], v))
    from collections import Counter
    print(f"evidence test gen: {Counter(t for _, t, _, _ in recs)} | n={a.n} | {model}", flush=True)

    def vurl(vid):
        return "file://" + str((TEST_VIDEO_BASE / vid).resolve())

    def one(rec):
        i, t, q, v = rec
        r = cli.chat.completions.create(model=model, max_tokens=512, temperature=0.7, top_p=0.95, n=a.n, seed=0,
            messages=[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": vurl(v)}},
                {"type": "text", "text": q}]}])
        return i, {"task": t, "samples": [c.message.content or "" for c in r.choices]}

    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    todo = [r for r in recs if r[0] not in out]
    with ThreadPoolExecutor(a.workers) as ex:
        for k, (iid, d) in enumerate(ex.map(one, todo)):
            out[iid] = d
            if (k + 1) % 20 == 0:
                json.dump(out, open(a.out, "w")); print(f"{k+1}/{len(todo)}", flush=True)
    json.dump(out, open(a.out, "w"))
    print(f"GEN DONE -> {a.out} ({len(out)} items)", flush=True)


if __name__ == "__main__":
    main()
