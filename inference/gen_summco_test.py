"""summ fullevent+mco TEST gen: the deployed fullevent hint (loc + causal Q + entity) PLUS our submitted mcq
option text ('what happened'). val: fullevent 0.4958 -> +mco 0.5281 (+0.0324, n=20). cosmos3_env -> f1."""
import os
import re
import sys
import csv
import json
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import openai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TEST_JSON, TEST_VIDEO_BASE  # noqa

_OPT = re.compile(r"(?:^|\n)\s*([A-E])[\.\)]\s*(.+?)(?=\n\s*[A-E][\.\)]|\Z)", re.S)


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
                    help="our own vote output (mcq pick); NEVER a GT-derived / pair-repaired file")
    ap.add_argument("--pairs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_pairs_multi.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run/summco_test.json"))
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
    recs = []
    for v, d in byv.items():
        if not d.get("video_summarization"):
            continue
        loc = stripfmt(items[d["temporal_localization"][0]]["question"]) if d.get("temporal_localization") else ""
        cau = stripfmt(items[d["causal_linkage"][0]]["question"]) if d.get("causal_linkage") else ""
        mco = ""
        if d.get("mcq"):
            mi = d["mcq"][0]
            mco = opt_text(items[mi]["question"], sub.get(mi, ""))
        if not loc:
            continue
        pr = pairs.get(v)
        fe = f"Note: the video shows: {loc}."
        if cau:
            fe += f" A related question concerns: {cau}."
        if mco:
            fe += f" What happened: {mco}."
        if pr:
            fe += f" The vehicles are a {pr[0]} and a {pr[1]}."
        for i in d["video_summarization"]:
            recs.append((i, fe + " Summarize the full sequence of events including the incident and its "
                         "consequences, consistent with this.\n\n" + items[i]["question"], v))
    print(f"summco test gen: {len(recs)} items, n={a.n} | {model}", flush=True)

    def vurl(vid):
        return "file://" + str((TEST_VIDEO_BASE / vid).resolve())

    def one(rec):
        i, q, v = rec
        r = cli.chat.completions.create(model=model, max_tokens=512, temperature=0.7, top_p=0.95, n=a.n, seed=0,
            messages=[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": vurl(v)}},
                {"type": "text", "text": q}]}])
        return i, {"task": "video_summarization", "samples": [c.message.content or "" for c in r.choices]}

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
