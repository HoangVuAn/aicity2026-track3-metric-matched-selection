"""MBR test GEN (cosmos3_env -> vLLM f1). Gen N diverse samples per TEST F1 desc item for MBR selection."""
import os, sys, json, argparse
from concurrent.futures import ThreadPoolExecutor
import openai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TEST_JSON, TEST_VIDEO_BASE  # noqa: E402

TASKS = ["open_qa", "causal_linkage", "temporal_description", "video_summarization", "scene_description"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run/mbr_test.json"))
    ap.add_argument("--tasks", default="")
    ap.add_argument("--temp", type=float, default=0.7)   # sampling temperature (diversity sweep)
    ap.add_argument("--seed", type=int, default=0)        # sampling seed; vary for a distinct temp-0.7 draw
    ap.add_argument("--shard", default="")   # comma-sep subset; empty = all 5
    a = ap.parse_args()
    cli = openai.OpenAI(api_key="EMPTY", base_url=a.base_url, timeout=300, max_retries=6)
    model = cli.models.list().data[0].id
    tset = set(a.tasks.split(",")) if a.tasks else set(TASKS)
    items = [it for it in json.load(open(TEST_JSON))["items"] if it["task_type"] in TASKS and it["task_type"] in tset]
    if a.shard:
        si, sn = map(int, a.shard.split("/"))
        items = [it for k, it in enumerate(items) if k % sn == si]
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    todo = [it for it in items if str(it["item_index"]) not in out]
    print(f"mbr test gen: {len(todo)} items, n={a.n}, model={model}", flush=True)

    def vurl(vid):
        return (TEST_VIDEO_BASE / vid).resolve().as_uri()

    def ask(vid, q, n, temp, top_p):
        r = cli.chat.completions.create(model=model, max_tokens=512, temperature=temp, top_p=top_p, n=n, seed=a.seed,
            messages=[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": vurl(vid)}},
                {"type": "text", "text": q}]}])
        return [c.message.content or "" for c in r.choices]

    def one(it):
        try:
            greedy = ask(it["video_id"], it["question"], 1, 0.0, 1.0)[0]
            samples = ask(it["video_id"], it["question"], a.n, a.temp, 0.95)
        except Exception as e:
            print("fail", it["item_index"], str(e)[:80], flush=True)
            greedy, samples = "", []
        return str(it["item_index"]), {"task": it["task_type"], "greedy": greedy, "samples": samples}

    with ThreadPoolExecutor(a.workers) as ex:
        for i, (iid, d) in enumerate(ex.map(one, todo)):
            out[iid] = d
            if (i + 1) % 20 == 0:
                json.dump(out, open(a.out, "w")); print(f"{i+1}/{len(todo)}", flush=True)
    json.dump(out, open(a.out, "w"))
    print("GEN DONE ->", a.out, flush=True)


if __name__ == "__main__":
    main()
