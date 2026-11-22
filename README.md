# eu-finreg-rag

Retrieval-augmented question answering over the **EU AI Act** and **DORA**, with
paragraph-level citations and an evaluation that measures retrieval and answers.

- **Corpus:** the two regulations, parsed into 799 citable units (`AI Act Art. 99(3)`, `DORA Art. 19(4)`, annexes).
- **Pipeline:** article-aware chunking → local `bge-small-en-v1.5` embeddings (ONNX, no API cost) → Qdrant (embedded) → Claude Haiku 5.5 answers that cite sources by number → FastAPI, Docker.
- **Eval:** 60 questions with reference answers and verbatim source quotes (draft, under author review); hit@k, MRR, latency; citation accuracy, faithfulness and $/question with an API key.
- **Compared:** chunk size 512 vs 1024 characters.

## Results

Retrieval, measured 2026-11-22 on CPU (Windows laptop) with
`python -m finreg.evaluate --chunk-chars 512 1024`. Output copied unchanged from
`results/eval.md`:

Retrieval, 60 questions, embedder `BAAI/bge-small-en-v1.5`, Qdrant cosine, top-10:

| Chunk size (chars) | Chunks | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | article hit@5 | p50 / p95 latency |
|---|---|---|---|---|---|---|---|---|
| 512 | 1514 | 57% | 82% | 87% | 92% | 0.695 | 93% | 18 / 26 ms |
| 1024 | 736 | 75% | 88% | 93% | 95% | 0.820 | 98% | 14 / 20 ms |

- 1024-character chunks rank the gold paragraph higher on 17 of 60 questions; 512 does on 4.
  The other 39 are ties. A two-sided sign test on the 21 non-ties gives p = 0.0072 (exact
  binomial: `python -c "from math import comb; print(2*sum(comb(21,i) for i in range(17,22))/2**21)"`). The hit@10
  gap (95% vs 92%) is 2 questions and says little on its own.
- Latency covers query embedding plus vector search only; it excludes generation.
- **Answer metrics (citation accuracy, faithfulness, answer latency, $/question): not measured yet.**
  They need an `ANTHROPIC_API_KEY`; the command is under "How to reproduce".

## How to reproduce

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python -m finreg.ingest                                  # fetch + parse -> data/processed/units.jsonl
python -m finreg.index --chunk-chars 512 1024            # embed + index -> data/index
python -m finreg.evaluate --chunk-chars 512 1024         # retrieval table (no API cost)
export ANTHROPIC_API_KEY=...                             # optional: answers, citations, faithfulness
python -m finreg.evaluate --chunk-chars 512 1024 --answers --out results/eval_answers.json
uvicorn finreg.api:app --port 8000                       # POST /search, POST /ask
pytest -q && ruff check .                                # tests need no model and no key
```

Docker (one container; the corpus is fetched and indexed at build time):

```bash
docker build -t eu-finreg-rag .
docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY eu-finreg-rag
curl -s localhost:8000/search -H 'content-type: application/json' -d '{"question":"How often is TLPT required?","k":3}'
```

## How it works

| Step | Module | Choice |
|---|---|---|
| Ingest | `finreg/ingest.py` | Official Journal XHTML; units = numbered paragraphs, single-paragraph articles, annexes; recitals excluded |
| Chunk | `finreg/chunk.py` | Pack paragraphs of one article up to N chars; never cross an article; split long units at sentence boundaries |
| Embed | `finreg/embed.py` | `BAAI/bge-small-en-v1.5` via fastembed (ONNX, CPU); query instruction prefix |
| Index | `finreg/index.py` | Qdrant embedded mode, cosine, one collection per chunk size |
| Answer | `finreg/answer.py` | Claude Haiku 5.5, numbered sources, every claim cited `[n]`, `NOT_FOUND` if unsupported |
| Serve | `finreg/api.py` | FastAPI `/search` (no LLM), `/ask`, `/health` |
| Evaluate | `finreg/evaluate.py` | hit@1/3/5/10, MRR@10, article-level hit@5, latency; citation accuracy, LLM-judged faithfulness, $/question |

Design notes and likely interview questions per module: [docs/walkthrough.md](docs/walkthrough.md).

## Evaluation set

`evals/questions.jsonl`: 60 questions (30 AI Act, 30 DORA). Each has a reference answer,
the gold paragraph (`source`) and a verbatim `quote` from it; a test checks every quote
against the parsed corpus. The questions are phrased the way a compliance analyst would
ask them, not copied from the text. **Status: draft, not yet reviewed by the author.**

## What failed / limitations

- **Answer quality is not measured yet:** citation accuracy, faithfulness, answer latency and
  $/question need an `ANTHROPIC_API_KEY`, and that run has not happened.
- **3 questions miss the top 10 at 1024 characters** (`results/eval.json`, `per_question`).
  q28 ("From what date do the AI Act's prohibitions apply?") misses in both configurations:
  the gold text, Art. 113, says "Chapters I and II shall apply from 2 February 2025" and never
  uses the word "prohibition", so a dense embedder has nothing to match. q47 (DORA Art. 24(6))
  and q51 (DORA Art. 28(1)) find the right article but not the paragraph. A reranker, BM25,
  or indexing the article heading and chapter titles with each chunk are the next things to test.
- The eval set is small (n=60: a 95% interval is roughly ±12 points) and single-annotator.
- One gold paragraph per question; answers spanning paragraphs are scored against one.
- No reranker, no hybrid (BM25) retrieval, no questions without an answer in the corpus.
- Recitals are excluded, so "why" questions about legislative intent are out of scope.
- Bank annual reports were planned but not included (PDF parsing and copyright terms).
- `docker build` has not been run in this environment.

## Sources and licences

| Document | CELEX | Source |
|---|---|---|
| Regulation (EU) 2024/1689 (Artificial Intelligence Act) | 32024R1689 | https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689 |
| Regulation (EU) 2022/2554 (DORA) | 32022R2554 | https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R2554 |

Fetched from the Publications Office of the EU (`http://publications.europa.eu/resource/celex/<CELEX>`).
© European Union, https://eur-lex.europa.eu. Reuse is authorised under Commission Decision
2011/833/EU, with acknowledgement of the source. Only the Official Journal publication is
authentic. The texts are not stored in this repository; `finreg.ingest` downloads them.
