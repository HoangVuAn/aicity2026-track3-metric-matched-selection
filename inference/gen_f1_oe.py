import sys, json, re, argparse, os
from concurrent.futures import ThreadPoolExecutor
import openai
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TEST_JSON, TEST_VIDEO_BASE
from inference import parse_answer, extract_letter
SC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run")
LET=["A","B","C","D"]; PERMS=[[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]]
def parse_q(q):
    m=re.search(r"(?m)^\s*A[.)]\s",q)
    if not m: return None
    parts=re.split(r"(?m)^\s*([A-D])[.)]\s+",q[m.start():]); o={}
    for k in range(1,len(parts)-1,2): o[parts[k]]=parts[k+1]
    if set(o)!=set(LET): return None
    o["D"]=re.split(r"\n\s*\n|\bChoose the correct\b|\bAnswer with\b",o["D"],maxsplit=1)[0]
    return q[:m.start()].strip(), {k:" ".join(v.split()).strip().rstrip("?").strip() for k,v in o.items()}, \
           (re.search(r"(Choose the correct[^\n]*|Answer with[^\n]*)",q).group(1).strip() if re.search(r"Choose the correct|Answer with",q) else "Answer with the letter of the correct option.")
def norm(s): return re.sub(r"[^a-z0-9 ]","",str(s).lower())[:55]
def build(stem,opts,instr,perm): return stem+"\n"+"\n".join(f"{LET[i]}) {opts[LET[perm[i]]]}" for i in range(4))+"\n"+instr
CLI=None; MODEL=None
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--base-url"); ap.add_argument("--workers",type=int,default=16); ap.add_argument("--n",type=int,default=5); ap.add_argument("--out",default=None); a=ap.parse_args()
    global CLI,MODEL; CLI=openai.OpenAI(api_key="EMPTY",base_url=a.base_url,timeout=180,max_retries=0); MODEL=CLI.models.list().data[0].id
    items=json.load(open(TEST_JSON))["items"]
    mcqoe=[it for it in items if it["task_type"]=="mcq_openended"]
    bcqoe=[it for it in items if it["task_type"]=="bcq_openended"]
    pf=a.out or f"{SC}/gen_f1_oe.json"; out=json.load(open(pf)) if os.path.exists(pf) else {}
    # tasks: ('mcqoe', item) per-perm picks ; ('bcqoe', item) N samples
    jobs=[("mcqoe",it) for it in mcqoe if "moe_"+it["item_index"] not in out] + \
         [("bcqoe",it) for it in bcqoe if "boe_"+it["item_index"] not in out]
    print(f"mcq_oe {len(mcqoe)} + bcq_oe {len(bcqoe)} | jobs {len(jobs)}",flush=True)
    def one(job):
        kind,it=job; vp=(TEST_VIDEO_BASE/it["video_id"]).resolve().as_uri()
        if kind=="mcqoe":
            p=parse_q(it["question"])
            if not p: return "moe_"+it["item_index"],None
            stem,opts,instr=p; picks=[]
            for perm in PERMS:
                try:
                    r=CLI.chat.completions.create(model=MODEL,max_tokens=300,temperature=0,seed=0,
                        messages=[{"role":"user","content":[{"type":"video_url","video_url":{"url":vp}},{"type":"text","text":build(stem,opts,instr,perm)}]}])
                    L=extract_letter(parse_answer(r.choices[0].message.content or ""))
                except Exception: L=None
                if L in LET: picks.append(norm(opts[LET[perm[LET.index(L)]]]))
            return "moe_"+it["item_index"], picks
        else:  # bcqoe: N samples (raw text)
            try:
                r=CLI.chat.completions.create(model=MODEL,max_tokens=400,temperature=0.8,top_p=0.95,seed=0,n=a.n,
                    messages=[{"role":"user","content":[{"type":"video_url","video_url":{"url":vp}},{"type":"text","text":it["question"]}]}])
                outs=[c.message.content or "" for c in r.choices]
            except Exception: outs=[]
            return "boe_"+it["item_index"], outs
    with ThreadPoolExecutor(a.workers) as ex:
        for i,(k,v) in enumerate(ex.map(one,jobs)):
            out[k]=v
            if (i+1)%20==0: json.dump(out,open(pf,"w")); print(f"{i+1}/{len(jobs)}",flush=True)
    json.dump(out,open(pf,"w")); print("DONE",flush=True)
main()
