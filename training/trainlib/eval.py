"""Local validation scoring with the official-style metrics (BERTScore F1 for open-ended,
exact-match accuracy for bcq/mcq), shared by the transformers-checkpoint eval and the
vLLM prompt-tuning eval scripts."""

from __future__ import annotations

from collections import defaultdict

from trainlib.inference import extract_letter, extract_yesno

OPEN_ENDED = {"bcq_openended", "mcq_openended", "open_qa", "causal_linkage",
              "scene_description", "temporal_description", "video_summarization"}


def score_predictions(records: list[dict], preds: dict[str, str]) -> dict[str, float]:
    """records each have id/task/answer; preds maps id -> prediction text. Returns per-task score."""
    import bert_score
    scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
    by: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by[r["task"]].append(r)
    out: dict[str, float] = {}
    for task, rs in by.items():
        if task in OPEN_ENDED:
            _, _, f1 = scorer.score([preds.get(r["id"], "") or " " for r in rs],
                                    [str(r["answer"]) for r in rs])
            out[task] = float(f1.mean())
        elif task == "bcq":
            out[task] = sum(extract_yesno(preds.get(r["id"], "") or "") == extract_yesno(str(r["answer"]))
                            for r in rs) / len(rs)
        elif task == "mcq":
            out[task] = sum(extract_letter(preds.get(r["id"], "") or "") == extract_letter(str(r["answer"]))
                            for r in rs) / len(rs)
    return out


def print_scores(title: str, scores: dict[str, float]) -> None:
    print("\n" + "=" * 50)
    print(title)
    for t in sorted(scores):
        print(f"  {t:24} {scores[t]:.4f}")
    if scores:
        print(f"  {'MEAN':24} {sum(scores.values()) / len(scores):.4f}")
    print("=" * 50)
