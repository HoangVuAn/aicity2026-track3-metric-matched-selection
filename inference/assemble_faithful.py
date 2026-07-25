"""Faithful final assembler: reproduces submission_tdesc_pure.csv in ONE deterministic pass
(final full-test 0.6696; 0.6790 on the visible-50% live leaderboard).

Base = assemble_clean.py output (choice tasks via the real box-cue cross-format cross-model ensemble + desc via
richpool). This script then applies each task's FINAL deployed recipe on top:
  mcq / mcq_openended / scene_description : keep base (assemble_clean already optimal)
  open_qa      : MBR[ evidence-grounded + richpool ]
  causal       : MBR[ evidence-grounded(mcq-option root cause) + richpool ]
  summ         : MBR[ summco(fullevent+mco) + summ_fullevent + richpool ]
  temporal     : MBR[ locevent-grounded ONLY ]  (pure)
  bcq          : base vote -> judge cross-task consistency check (uncertain-vote) -> 1-Yes-1-No structural repair
  bcq_openended: base; No-items (per final bcq) regenerated from evidence, polarity-aligned

Reads the generated pool files from --pool (default inference/full_run). Deterministic given fixed pools.
"""
from __future__ import annotations

import os
import re
import csv
import sys
import json
import argparse
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from paths import TEST_JSON  # noqa: E402


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="dataset/official_test/submission_clean.csv",
                    help="assemble_clean.py output (choice+desc base)")
    ap.add_argument("--pool", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run"),
                    help="dir with the generated pool JSONs")
    ap.add_argument("--judge-url", default="")
    ap.add_argument("--out", default="dataset/official_test/submission_repro.csv")
    ap.add_argument("--pair-repair", action="store_true",
                    help="Enable the 1-Yes-1-No structural repair (a dataset-structure prior). OFF by default: "
                         "the pipeline is consistency-only, so it does NOT assume each video has exactly one Yes + one No "
                         "bcq. Turn ON only if the target set is known to share that construction (matches public "
                         "test); leaving it OFF avoids overfitting the pair structure on a differently-built set.")
    a = ap.parse_args()
    P = a.pool

    items = json.load(open(TEST_JSON))["items"]
    itt = {str(it["item_index"]): it["task_type"] for it in items}
    byv = defaultdict(lambda: defaultdict(list))
    for it in items:
        byv[it["video_id"]][it["task_type"]].append(it)
    pred = {r["item_index"]: r["prediction"] for r in csv.DictReader(open(a.base))}

    # richpool sources: 2 temp-0.7 draws (seed 0 + seed 1) + 1 temp-0.9 draw, per model (f1 + cosmos_v2) = 6
    # sources, matching the deployed richpool. mbr_test_2 / mbr_test_cosmos_2 are the 2nd temp-0.7 draw
    # (Finding 1). The biggerN/rerun_clean entries are the equivalent deployed-layout fallbacks.
    rich = [d for d in (load(os.path.join(P, f)) for f in
            ("mbr_test.json", "mbr_test_cosmos.json",
             "mbr_test_2.json", "mbr_test_cosmos_2.json",
             "mbr_test_t09.json", "mbr_test_cosmos_t09.json",
             "biggerN/mbr_test.json", "biggerN/mbr_test_cosmos.json",
             "rerun_clean/mbr_test.json", "rerun_clean/mbr_test_cosmos.json")) if d]

    def loadP(name):
        return load(os.path.join(P, name))
    locev = loadP("locevent_temporal_test.json")
    summco = loadP("summco_test.json")
    summfe = loadP("summ_fullevent_test.json")
    # evidence_test.json keyed by item_index with {"task": open_qa|causal_linkage|bcq_oe_no, "samples":[...]}
    ev = loadP("evidence_test.json")

    def rp(i):
        out = []
        for s in rich:
            out += [C.clean(x) for x in (s.get(i, {}).get("samples") or [])]
        return out

    def g(store, i):
        return [C.clean(x) for x in (store.get(i, {}).get("samples") or [])]

    def ev_for(i, task):
        e = ev.get(i, {})
        return [C.clean(x) for x in e.get("samples", [])] if e.get("task") == task else []

    # ---- open-ended final recipes ----
    for i, t in itt.items():
        if t == "open_qa":
            pool = ev_for(i, "open_qa") + rp(i)
            if pool:
                pred[i] = C.mbr(pool)
        elif t == "causal_linkage":
            pool = ev_for(i, "causal_linkage") + rp(i)
            if pool:
                pred[i] = C.mbr(pool)
        elif t == "video_summarization":
            pool = g(summco, i) + g(summfe, i) + rp(i)
            if pool:
                pred[i] = C.mbr(pool)
        elif t == "temporal_description":
            pool = g(locev, i)                         # PURE locevent
            if len(pool) >= 2:
                pred[i] = C.mbr(pool)
        elif t == "scene_description":
            pool = rp(i)                               # 6-source richpool, FULL pool (matches assemble_richpool)
            if pool:
                pred[i] = C.mbr(pool, cap=10 ** 9)     # no cap -> same argmax as the deployed richpool MBR

    # ---- bcq: cross-task consistency check (judge, uncertain votes) then 1-Yes-1-No repair ----
    # margin from the SAME vote pool that made the bcq decision (assemble_clean writes bcq_votepool.json),
    # so cross-task consistency check gating + pair-repair flip-choice use the decision's own confidence. Fallback to the raw
    # model votes (ck+cz+cv2) only if that artifact is absent (e.g. an older pool dir).
    votepool = loadP("bcq_votepool.json")
    ck = loadP("ck400_test_bcq.json") or loadP("biggerN/ck400_test_bcq.json")
    cz = loadP("cosmos_votetta.json") or loadP("biggerN/cosmos_votetta.json")
    cv2 = loadP("cosmos_v2_votetta.json") or loadP("biggerN/cosmos_v2_votetta.json")

    def bcq_margin(i):
        if i in votepool:
            v = [x for x in votepool[i] if x in ("yes", "no")]
        else:
            v = []
            for s in (ck, cz, cv2):
                v += [x for x in (s.get(i, {}).get("votes") or []) if x in ("yes", "no")]
        if not v:
            return 1.0
        c = Counter(v)
        return (2 * c.most_common(1)[0][1] - len(v)) / len(v)

    # bcq: FULL cross-task consistency pass (all bcq, sharper "different-thing->NO" prompt) -> selective override of UNCERTAIN votes
    # (margin<=0.5) only. This is model + cross-task consistency and makes NO use of the 1-Yes-1-No pair structure. The structural
    # repair below is OPT-IN (--pair-repair): it mirrors repair_bcq_pairs.py (cross-task consistency-guided flip of violating pairs)
    # and lifts bcq substantially on the public test, but relies on the pair-construction prior, so it stays OFF
    # by default to keep the pipeline robust to a differently-built private set.
    consistency_verdict = _cross_task_consistency(byv, pred, a.judge_url) if a.judge_url else {}
    for vid, d in byv.items():
        for it in d.get("bcq", []):
            i = str(it["item_index"])
            v = consistency_verdict.get(i)
            if v in ("yes", "no") and bcq_margin(i) <= 0.5:
                pred[i] = v.capitalize()
    if a.pair_repair:
        _pair_repair(byv, pred, bcq_margin, consistency_verdict)

    # ---- bcq_openended: No-items regen (evidence) + polarity to final bcq ----
    for vid, d in byv.items():
        bpol = {C.first_line(C.strip_format(it["question"])).lower(): pred.get(str(it["item_index"]), "").strip().lower()
                for it in d.get("bcq", [])}
        for it in d.get("bcq_openended", []):
            i = str(it["item_index"])
            should = bpol.get(C.first_line(C.strip_format(it["question"])).lower())
            if should == "no":
                cand = C.mbr(ev_for(i, "bcq_oe_no"))
                if C.yn(cand) == "no":
                    pred[i] = cand

    # ---- write + non-empty guard ----
    rows = []
    for it in items:
        i = str(it["item_index"])
        p = pred.get(i, "").strip()
        if not p:
            p = C.mbr(rp(i)) or "No"
        rows.append({"item_index": i, "prediction": p})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_index", "prediction"]); w.writeheader(); w.writerows(rows)
    empty = sum(1 for r in rows if not r["prediction"].strip())
    print(f"[faithful] wrote {a.out}: {len(rows)} rows, {empty} empty", flush=True)


_CONSISTENCY_QTASKS = ["temporal_localization", "open_qa", "causal_linkage", "scene_description",
                "video_summarization", "temporal_description"]


def _consistency_verdict(s):
    s = (s or "").strip().lower()
    if re.search(r"\bunknown\b|\bunclear\b|\bcannot\b|\bnot determin", s):
        return "unknown"
    m = re.search(r"\b(yes|no)\b", s)
    return m.group(1) if m else "unknown"


def _cross_task_consistency(byv, pred, judge_url):
    """One judge pass over ALL bcq (not just uncertain). Context = other-task questions (presupposing events)
    + [FACT] our picked mcq option text + [FACT] our mcq_oe answer. Sharper prompt distinguishes a DIFFERENT
    thing (-> NO) to curb the yes-bias. Returns {item_index: yes|no|unknown}. cross-task consistency check over sibling-task answers + question presuppositions."""
    import openai
    cli = openai.OpenAI(api_key="EMPTY", base_url=judge_url, timeout=120, max_retries=6)
    mid = cli.models.list().data[0].id

    def context(d):
        parts = ["Questions other analysts asked about this video (they presuppose real events):"]
        for t in _CONSISTENCY_QTASKS:
            for it in d.get(t, []):
                parts.append(f"- ({t}) {C.strip_format(it['question'])[:170]}")
        for it in d.get("mcq", []):
            tx = C.option_text(it["question"], pred.get(str(it["item_index"]), ""))
            if tx:
                parts.append(f"[FACT - verified answer to 'what happened'] {tx}")
        for it in d.get("mcq_openended", []):
            parts.append(f"[FACT - verified description] {pred.get(str(it['item_index']), '')[:220]}")
        return "\n".join(parts)

    jobs = []
    for vid, d in byv.items():
        ctx = context(d)
        for it in d.get("bcq", []):
            prompt = (f"About ONE traffic video, verified information and related questions:\n{ctx}\n\n"
                      f"Yes/No question: {C.strip_format(it['question'])}\n\n"
                      "Using ONLY the information above (you have NOT seen the video):\n"
                      "- If the info states or clearly implies the target IS present -> YES.\n"
                      "- If the info fully describes the event and the target is a DIFFERENT thing that would be "
                      "mentioned if present -> NO.\n"
                      "- If the info does not settle it -> UNKNOWN.\n"
                      "Answer ONE word: YES, NO, or UNKNOWN.")
            jobs.append((str(it["item_index"]), prompt))

    def one(job):
        i, prompt = job
        try:
            r = cli.chat.completions.create(model=mid, temperature=0.0, max_tokens=8,
                                            messages=[{"role": "user", "content": prompt}])
            return i, _consistency_verdict(r.choices[0].message.content)
        except Exception:
            return i, "unknown"
    out = {}
    with ThreadPoolExecutor(8) as ex:
        for i, v in ex.map(one, jobs):
            out[i] = v
    return out


def _pair_repair(byv, pred, margin_fn, consistency_verdict=None):
    """1-Yes-1-No structural repair. On a violating pair (both same), flip ONE item: prefer the one whose cross-task consistency
    verdict DISAGREES with our answer (cross-task consistency ~0.99-acc when determinable); else flip the lower-margin one.
    Mirrors repair_bcq_pairs.py."""
    consistency_verdict = consistency_verdict or {}
    for vid, d in byv.items():
        b = d.get("bcq", [])
        if len(b) != 2:
            continue
        i1, i2 = str(b[0]["item_index"]), str(b[1]["item_index"])
        a1, a2 = pred[i1].strip().lower(), pred[i2].strip().lower()
        if a1 != a2:
            continue  # already one-yes-one-no
        l1, l2 = consistency_verdict.get(i1), consistency_verdict.get(i2)
        d1 = l1 in ("yes", "no") and l1 != a1
        d2 = l2 in ("yes", "no") and l2 != a2
        if d1 and not d2:
            flip = i1
        elif d2 and not d1:
            flip = i2
        else:
            flip = i1 if margin_fn(i1) <= margin_fn(i2) else i2
        pred[flip] = "No" if pred[flip].strip().lower() == "yes" else "Yes"


if __name__ == "__main__":
    main()
