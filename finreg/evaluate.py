"""Evaluate retrieval (and, with an API key, answers) for several chunk sizes.

    python -m finreg.evaluate --chunk-chars 512 1024                 # retrieval only, no API cost
    python -m finreg.evaluate --chunk-chars 512 1024 --answers       # + answers, citations, faithfulness

Retrieval metrics, per configuration, over evals/questions.jsonl:
  hit@k   share of questions where a chunk containing the gold paragraph is in the top k
  MRR@10  mean of 1/rank of the first such chunk (0 if not in the top 10)
  art@5   lenient hit@5: any chunk from the gold article (paragraph boundaries can differ)
  latency query embedding + vector search, p50/p95 seconds
Answer metrics (--answers, needs ANTHROPIC_API_KEY):
  citation accuracy  share of answers that cite a source containing the gold paragraph
  faithfulness       share of answers an LLM judge rates as fully supported by the cited sources
  not-found rate, latency (retrieval + generation) and USD per question from token counts
"""
import argparse
import json
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from finreg import answer as ans
from finreg import index
from finreg.chunk import chunk_units

KS = (1, 3, 5, 10)

JUDGE_SYSTEM = """\
You check whether an answer is supported by its sources. Reply with one word: SUPPORTED if \
every factual statement in the answer is stated in or directly follows from the sources, \
otherwise UNSUPPORTED."""


@dataclass
class QResult:
    qid: str
    rank: int | None          # rank of first chunk containing the gold paragraph (1-based), None if > 10
    article_rank: int | None
    retrieval_s: float


def gold_rank(hits: list[index.Hit], source: str) -> int | None:
    return next((h.rank for h in hits if source in h.unit_ids), None)


def article_rank(hits: list[index.Hit], source: str) -> int | None:
    doc_art = source.split("(")[0]  # "DORA Art. 19(4)" -> "DORA Art. 19"
    return next((h.rank for h in hits if f"{h.doc} {h.article}" == doc_art), None)


def retrieval_metrics(results: list[QResult]) -> dict:
    n = len(results)
    lat = sorted(r.retrieval_s for r in results)
    out = {f"hit@{k}": sum(r.rank is not None and r.rank <= k for r in results) / n for k in KS}
    out["mrr@10"] = sum(1 / r.rank for r in results if r.rank) / n
    out["art@5"] = sum(r.article_rank is not None and r.article_rank <= 5 for r in results) / n
    out["p50_s"] = statistics.median(lat)
    out["p95_s"] = lat[min(n - 1, int(round(0.95 * (n - 1))))]
    out["n"] = n
    return out


def judge(generate: ans.Generate, a: ans.Answer) -> tuple[bool, int, int]:
    cited = [a.sources[i - 1] for i in a.cited]
    srcs = "\n\n".join(f"[{i}] {h.citation}\n{h.text}" for i, h in zip(a.cited, cited, strict=True))
    text, tin, tout = generate(JUDGE_SYSTEM, f"Sources:\n{srcs}\n\nAnswer:\n{a.text}")
    return text.strip().upper().startswith("SUPPORTED"), tin, tout


def run_config(client, embedder, units, questions, chunk_chars, answers: ans.Generate | None,
               judge_gen: ans.Generate | None, judge_model: str = "claude-sonnet-5-5", k_answer: int = 5) -> dict:
    name = index.collection_name(chunk_chars)
    if not client.collection_exists(name):
        index.build(client, embedder, chunk_units(units, chunk_chars), name)
    per_q, answer_rows = [], []
    for q in questions:
        start = time.perf_counter()
        hits = index.search(client, embedder, name, q["question"], k=max(KS))
        elapsed = time.perf_counter() - start
        per_q.append(QResult(q["id"], gold_rank(hits, q["source"]), article_rank(hits, q["source"]), elapsed))
        if answers:
            t0 = time.perf_counter()
            a = ans.answer(q["question"], hits[:k_answer], answers)
            gen_s = time.perf_counter() - t0
            cites_gold = any(q["source"] in a.sources[i - 1].unit_ids for i in a.cited)
            supported, jin, jout = judge(judge_gen, a) if (judge_gen and a.cited) else (False, 0, 0)
            answer_rows.append({"qid": q["id"], "answer": a.text, "citations": a.citations,
                                "not_found": a.not_found, "cites_gold": cites_gold, "supported": supported,
                                "latency_s": elapsed + gen_s, "cost_usd": a.cost_usd,
                                "judge_cost_usd": ans.cost_usd(judge_model, jin, jout)})
    out = {"chunk_chars": chunk_chars, "retrieval": retrieval_metrics(per_q),
           "per_question": [r.__dict__ for r in per_q]}
    if answer_rows:
        n = len(answer_rows)
        out["answers"] = {
            "citation_accuracy": sum(r["cites_gold"] for r in answer_rows) / n,
            "faithfulness": sum(r["supported"] for r in answer_rows) / n,
            "not_found_rate": sum(r["not_found"] for r in answer_rows) / n,
            "p50_s": statistics.median(r["latency_s"] for r in answer_rows),
            "usd_per_question": sum(r["cost_usd"] for r in answer_rows) / n,
            "judge_usd_total": sum(r["judge_cost_usd"] for r in answer_rows), "rows": answer_rows}
    return out


def render(results: list[dict], embedder_name: str, n_questions: int) -> str:
    lines = [f"Retrieval, {n_questions} questions, embedder `{embedder_name}`, Qdrant cosine, top-10:", "",
             "| Chunk size (chars) | Chunks | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | article hit@5 | p50 / p95 latency |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m = r["retrieval"]
        lines.append(f"| {r['chunk_chars']} | {r.get('n_chunks', '–')} | {m['hit@1']:.0%} | {m['hit@3']:.0%} | "
                     f"{m['hit@5']:.0%} | {m['hit@10']:.0%} | {m['mrr@10']:.3f} | {m['art@5']:.0%} | "
                     f"{m['p50_s'] * 1000:.0f} / {m['p95_s'] * 1000:.0f} ms |")
    if any("answers" in r for r in results):
        lines += ["", f"Answers ({ans.MODEL}, top-5 sources):", "",
                  "| Chunk size | Citation accuracy | Faithfulness (LLM judge) | NOT_FOUND | p50 latency | $/question |",
                  "|---|---|---|---|---|---|"]
        for r in results:
            a = r.get("answers")
            if a:
                lines.append(f"| {r['chunk_chars']} | {a['citation_accuracy']:.0%} | {a['faithfulness']:.0%} | "
                             f"{a['not_found_rate']:.0%} | {a['p50_s']:.1f} s | ${a['usd_per_question']:.5f} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--chunk-chars", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--questions", default="evals/questions.jsonl")
    ap.add_argument("--units", default="data/processed/units.jsonl")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--answers", action="store_true", help="also generate and judge answers (API cost)")
    ap.add_argument("--judge-model", default="claude-sonnet-5-5")
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args(argv)

    from finreg.embed import FastEmbedder

    questions = [json.loads(line) for line in open(args.questions, encoding="utf-8")]
    units = index.load_units(args.units)
    client, emb = index.open_client(args.index_dir), FastEmbedder()
    gen = ans.claude_generate() if args.answers else None
    judge_gen = ans.claude_generate(model=args.judge_model) if args.answers else None
    results = []
    for n in args.chunk_chars:
        r = run_config(client, emb, units, questions, n, gen, judge_gen, args.judge_model)
        r["n_chunks"] = len(chunk_units(units, n))
        results.append(r)
    table = render(results, emb.name, len(questions))
    meta = {"timestamp": datetime.now(UTC).isoformat(timespec="seconds"), "embedder": emb.name,
            "questions": args.questions, "n_questions": len(questions), "answers": args.answers}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": meta, "results": results}, indent=1), encoding="utf-8")
    Path(args.out).with_suffix(".md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
