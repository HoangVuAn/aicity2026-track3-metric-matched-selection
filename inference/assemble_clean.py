"""CLEAN assembler (docs/vote_table_clean.md). Uniform 4-model, orphan = paired via synth-sibling with the
correct per-format model, shared 12/12/12/12 Yes/No stance pool, no f1 double-count. Run in locate_env.

Consumes (under --dir):
  cosmos_votetta.json     cosmos-base N=12 : bcq closed votes(12), mcq/mcq_oe perm-TTA picks, boe samples
  cosmos_v2_votetta.json  cosmos_v2  N=12 : boe samples(12, all 160), mcq_oe perm-TTA picks
  boxcue_votetta.json     box-cue    N=12 : bcq closed votes(12), mcq perm-TTA picks
  gen_f1_oe.json          f1              : boe_ samples(12), moe_ perm-TTA picks
  mbr_test.json           f1 desc N=24 ; mbr_test_cosmos.json  cosmos_v2 desc N=24
  orphan_votes_{cosmos,cosmos_v2,boxcue,f1}.json  synth-sibling second opinions
  dataset/official_test/pred_f1_128_ck3600.csv    fallback + temporal_localization
"""
import os, re, sys, csv, json, argparse
from collections import defaultdict, Counter
import bert_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TEST_JSON  # noqa: E402
LET = ["A", "B", "C", "D"]


def parse_answer(s):
    s = s or ""
    m = re.search(r"<answer>(.*?)</answer>", s, re.S)
    if m:
        return m.group(1).strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    return re.sub(r"^.*?</think>", "", s, flags=re.S).strip()


def extract_yesno(s):
    m = re.search(r"\b(yes|no)\b", (s or "").lower())
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


def stem(q):
    return re.sub(r"[^a-z0-9 ]", "", q.split("\n")[0].strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run"))
    ap.add_argument("--out", default="dataset/official_test/submission_clean.csv")
    ap.add_argument("--f1base", default="dataset/official_test/pred_f1_128_ck3600.csv",
                    help="f1 greedy base preds (fallback + temporal_localization). Point at a from-scratch "
                         "regen (inference/gen_pred_f1.py) for a fully self-contained reproduction.")
    a = ap.parse_args()
    D = a.dir

    def J(name, default=None):
        p = os.path.join(D, name)
        if os.path.exists(p):
            return json.load(open(p))
        return default if default is not None else {}

    items = json.load(open(TEST_JSON))["items"]
    cz = J("cosmos_votetta.json")
    cv2 = J("cosmos_v2_votetta.json")
    bx = J("boxcue_votetta.json")
    gf1 = J("gen_f1_oe.json")
    f1desc = J("mbr_test.json")
    cv2desc = J("mbr_test_cosmos.json")
    o_cz = J("orphan_votes_cosmos.json", {})
    o_cv2 = J("orphan_votes_cosmos_v2.json", {})
    o_bx = J("orphan_votes_boxcue.json", {})
    o_f1 = J("orphan_votes_f1.json", {})
    f1base = {r["item_index"]: r["prediction"]
              for r in csv.DictReader(open(a.f1base))}

    # cross-format pairing
    byv = defaultdict(lambda: defaultdict(list))
    for it in items:
        byv[it["video_id"]][it["task_type"]].append(it)
    mcq_pair, bcq_pair = {}, {}
    for v, d in byv.items():
        for x in d.get("mcq", []):
            ox = parse_opts(x["question"])
            for y in d.get("mcq_openended", []):
                oy = parse_opts(y["question"])
                if ox and oy and stem(x["question"]) == stem(y["question"]) and \
                        len(set(map(norm, ox.values())) & set(map(norm, oy.values()))) >= 3:
                    mcq_pair[str(x["item_index"])] = str(y["item_index"])
                    mcq_pair[str(y["item_index"])] = str(x["item_index"]); break
        for x in d.get("bcq", []):
            for y in d.get("bcq_openended", []):
                if stem(re.split(r"Answer with|Provide|Choose", x["question"])[0]) == \
                        stem(re.split(r"Answer with|Provide|Choose", y["question"])[0]):
                    bcq_pair[str(x["item_index"])] = str(y["item_index"])
                    bcq_pair[str(y["item_index"])] = str(x["item_index"]); break

    scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)

    def mbr(cands):
        cands = [c for c in cands if c]
        if len(cands) < 2:
            return cands[0] if cands else ""
        n = len(cands); pc, pr = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    pc.append(cands[i]); pr.append(cands[j])
        _, _, f1 = scorer.score(pc, pr); f1 = [float(x) for x in f1]
        k = 0; u = [0.0] * n
        for i in range(n):
            for j in range(n):
                if i != j:
                    u[i] += f1[k]; k += 1
        return cands[max(range(n), key=lambda i: u[i])]

    def boe_stances(iid):  # f1 + cosmos_v2 own free-text stances of a bcq_oe item
        v = [extract_yesno(parse_answer(s)) for s in (gf1.get("boe_" + iid) or [])]
        v += [extract_yesno(parse_answer(s)) for s in (cv2.get(iid, {}).get("samples") or [])]
        return [x for x in v if x]

    rows = []
    bcq_votepool = {}  # item_index -> exact yes/no vote list used for the decision (consumed by assemble_faithful margin)
    for it in items:
        i = str(it["item_index"]); t = it["task_type"]; p = f1base.get(i, "")

        if t == "bcq":
            # shared pool: cosmos-base closed(12) + box-cue closed(12) + f1 boe + cosmos_v2 boe (12 each)
            st = list(cz.get(i, {}).get("votes", [])) + list(bx.get(i, {}).get("votes", []))
            if i in bcq_pair:
                st += boe_stances(bcq_pair[i])
            else:  # orphan: f1 + cosmos_v2 synth bcq_oe stance
                st += [x for x in (o_f1.get(i, {}).get("votes") or []) if x]
                st += [x for x in (o_cv2.get(i, {}).get("votes") or []) if x]
            st = [x for x in st if x]
            bcq_votepool[i] = st
            p = "Yes" if (Counter(st).most_common(1)[0][0] if st else "no") == "yes" else "No"

        elif t == "mcq":
            o = parse_opts(it["question"])
            pool = list(cz.get(i, {}).get("picks", [])) + list(bx.get(i, {}).get("picks", []))
            if i in mcq_pair:
                pool += list(gf1.get("moe_" + mcq_pair[i]) or [])
                pool += list(cv2.get(mcq_pair[i], {}).get("picks") or [])
            else:  # orphan: f1 + cosmos_v2 synth mcq_oe picks
                pool += list(o_f1.get(i, {}).get("picks") or [])
                pool += list(o_cv2.get(i, {}).get("picks") or [])
            pool = [x for x in pool if o and x in set(map(norm, o.values()))]
            if pool and o:
                la = next((k for k, vv in o.items() if norm(vv) == Counter(pool).most_common(1)[0][0]), None)
                if la:
                    p = la

        elif t == "mcq_openended":
            ob = parse_opts(it["question"])
            pool = list(cz.get(i, {}).get("picks", [])) + list(gf1.get("moe_" + i) or []) \
                + list(cv2.get(i, {}).get("picks") or [])
            if i in mcq_pair:
                pool += list(bx.get(mcq_pair[i], {}).get("picks", []))
            else:  # orphan: box-cue synth closed mcq picks (cap 4 = perm-TTA-equivalent; NO f1 double-count)
                pool += list(o_bx.get(i, {}).get("picks") or [])[:4]
            pool = [x for x in pool if ob and x in set(map(norm, ob.values()))]
            if pool and ob:
                lb = next((k for k, vv in ob.items() if norm(vv) == Counter(pool).most_common(1)[0][0]), None)
                if lb:
                    p = f"{lb}. {ob[lb]}"

        elif t == "bcq_openended":
            f1s = [parse_answer(s) for s in (gf1.get("boe_" + i) or [])]
            cv2s = [parse_answer(s) for s in (cv2.get(i, {}).get("samples") or [])]
            allp = [x for x in (f1s + cv2s) if x]
            if allp:
                # shared stance pool: f1 boe + cosmos_v2 boe (own) + cosmos-base/box-cue closed (sibling/synth)
                votes = boe_stances(i)
                if i in bcq_pair:
                    votes += list(cz.get(bcq_pair[i], {}).get("votes", [])) + list(bx.get(bcq_pair[i], {}).get("votes", []))
                else:  # orphan: cosmos-base + box-cue synth closed bcq stance
                    votes += [x for x in (o_cz.get(i, {}).get("votes") or []) if x]
                    votes += [x for x in (o_bx.get(i, {}).get("votes") or []) if x]
                votes = [x for x in votes if x]
                vst = Counter(votes).most_common(1)[0][0] if votes else "no"
                cand = [x for x in allp if extract_yesno(x) == vst] or allp
                p = mbr(cand)

        elif t in ("open_qa", "scene_description", "video_summarization",
                   "causal_linkage", "temporal_description"):
            f1c = [parse_answer(s) for s in (f1desc.get(i, {}).get("samples") or [])]
            cv2c = [parse_answer(s) for s in (cv2desc.get(i, {}).get("samples") or [])]
            pool = [x for x in (f1c + cv2c) if x]
            if pool:
                p = mbr(pool)

        rows.append({"item_index": i, "prediction": p})

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_index", "prediction"]); w.writeheader(); w.writerows(rows)
    with open(os.path.join(D, "bcq_votepool.json"), "w") as f:
        json.dump(bcq_votepool, f)
    empty = sum(1 for r in rows if not r["prediction"].strip())
    print(f"WROTE {a.out}: {len(rows)} rows, {empty} empty")


if __name__ == "__main__":
    main()
