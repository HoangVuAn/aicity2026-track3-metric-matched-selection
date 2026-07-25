"""Format-TTA on the ORPHAN choice items (no real cross-pair): synthesize the sibling-format question
and generate an independent second-opinion vote. Run once per served model.
  --kind raw    : Cosmos-base / f1 (raw video_url), handles BOTH oe (explain) + plain forms.
  --kind boxcue : box-cue (boxed mp4 + LEGEND), PLAIN forms ONLY (box-cue can't do OE).
Output: orphan_votes_<tag>.json  {item_index: {task, form, votes|picks}}."""
import os
import re
import sys
import json
import argparse

import openai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TEST_JSON, TEST_VIDEO_BASE  # noqa: E402
from inference import parse_answer, extract_yesno, extract_letter  # noqa: E402

SC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run")
WORK = f"{SC}/orphan_work.json"
BOXMP4 = os.getenv("BOXCUE_MP4_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run/boxcue_mp4"))
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


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower())[:55]


def best_match(text, opts):
    """Map a chosen synth-option string to the best ORIGINAL option key by token-Jaccard.
    Synth siblings may reword options, so we can't rely on letter/position alignment."""
    tw = set(re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split())
    best, bs = None, -1.0
    for k, v in opts.items():
        vw = set(re.sub(r"[^a-z0-9 ]", " ", str(v).lower()).split())
        j = len(tw & vw) / max(1, len(tw | vw))
        if j > bs:
            bs, best = j, k
    return best if bs > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--kind", choices=["raw", "boxcue"], required=True)
    ap.add_argument("--tag", required=True)          # cosmos | f1 | boxcue
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"full_run/orphan_votes_{a.tag}.json")
    cli = openai.OpenAI(api_key="EMPTY", base_url=a.base_url, timeout=300, max_retries=6)
    model = cli.models.list().data[0].id
    work = json.load(open(WORK))
    items = {str(it["item_index"]): it for it in json.load(open(TEST_JSON))["items"]}
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}

    legend = ""
    if a.kind == "boxcue":
        import box_cue as bc
        legend = bc.LEGEND
        work = [w for w in work if w["form"] == "plain"]   # box-cue: plain only

    def vurl(w):
        if a.kind == "boxcue":
            return "file://" + os.path.abspath(f"{BOXMP4}/{w['video_id'].replace('/', '__')}.mp4")
        return "file://" + str((TEST_VIDEO_BASE / w["video_id"]).resolve())

    def ask(w, q, n, temp):
        r = cli.chat.completions.create(model=model, max_tokens=400, temperature=temp, top_p=0.95, seed=0, n=n,
            messages=[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": vurl(w)}},
                {"type": "text", "text": legend + q}]}])
        return [parse_answer(c.message.content or "") for c in r.choices]

    print(f"[{a.tag}] kind={a.kind} model={model} work={len(work)}", flush=True)
    for k, w in enumerate(work):
        iid = str(w["item_index"])
        if iid in out and out[iid].get("done"):
            continue
        t = w["task"]
        try:
            if t == "bcq":                # synth bcq_oe (explain) -> stance votes
                s = ask(w, w["synth_q"], a.n, 0.8)
                out[iid] = {"task": t, "form": w["form"], "votes": [extract_yesno(x) for x in s if extract_yesno(x)]}
            elif t == "bcq_openended":    # synth bcq (plain) -> stance votes
                s = ask(w, w["synth_q"], a.n, 0.8)
                out[iid] = {"task": t, "form": w["form"], "votes": [extract_yesno(x) for x in s if extract_yesno(x)]}
            elif t in ("mcq", "mcq_openended"):
                # perm-TTA-4 on the SYNTH's own options, then norm-map chosen content to the ORIGINAL option
                o_s = parse_opts(w["synth_q"])            # synth sibling's options (may be reworded)
                o_o = parse_opts(items[iid]["question"])  # original item's options (assembler matches these)
                picks = []
                if o_s and o_o:
                    s_stem = re.split(r"(?m)^\s*A[.)]\s", w["synth_q"])[0]
                    for perm in [["A", "B", "C", "D"], ["B", "C", "D", "A"], ["C", "D", "A", "B"], ["D", "A", "B", "C"]]:
                        body = "\n".join(f"{LET[j]}. {o_s[perm[j]]}" for j in range(4))
                        q2 = s_stem + body + "\nChoose the correct option. Answer with the letter."
                        la = extract_letter(ask(w, q2, 1, 0.0)[0])
                        if la in LET:
                            chosen = o_s[perm[LET.index(la)]]      # synth content picked under this rotation
                            mk = best_match(chosen, o_o)           # careful norm-map -> original option
                            if mk:
                                picks.append(norm(o_o[mk]))
                out[iid] = {"task": t, "form": w["form"], "picks": picks}
            out[iid]["done"] = True
        except Exception as e:
            out[iid] = {"task": t, "form": w["form"], "err": str(e)[:80], "done": True}
        if (k + 1) % 20 == 0:
            json.dump(out, open(out_path, "w")); print(f"[{a.tag}] {k+1}/{len(work)}", flush=True)
    json.dump(out, open(out_path, "w"))
    print(f"[{a.tag}] DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
