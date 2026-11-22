# Code walkthrough

## `finreg/ingest.py`: corpus → citable units

**What it does.** Fetches the AI Act and DORA as XHTML from the Publications Office
(Cellar) and parses them with the standard-library XML parser into units. A unit is one
numbered paragraph ("AI Act Art. 99(3)"), a whole article when it has no numbered
paragraphs, or a whole annex. Recitals are skipped.

**Why.** Legal answers are only useful with a precise reference. If every unit carries
its citation from the start, every chunk, hit and answer can name the paragraph it came
from. The Official Journal XHTML marks articles (`id="art_99"`) and paragraphs
(`id="099.003"`), so no layout heuristics are needed.

**Alternative rejected.** Parsing the PDF version. PDF extraction loses the
article/paragraph structure and needs heuristics for headers and footers.

**Failure case.** EUR-Lex HTML pages return HTTP 202 with an empty body (a bot challenge)
to scripted clients. The ingester uses the Cellar CELEX resource with content
negotiation instead, which serves the same Official Journal text.

**Questions**
1. *Why leave out recitals?* They explain intent but create no obligations. In
   retrieval they compete with the articles that actually answer "must I …?" questions.
   This is a trade-off, listed under limitations.
2. *How would you add bank annual reports?* As PDF units with page-level citations
   ("ING AR 2024 p. 112"); the unit schema stays the same.
3. *Is the text authentic?* It is the Official Journal text, but only the OJ publication
   is legally binding, as the README says.

## `finreg/chunk.py`: chunking

**What it does.** Packs consecutive paragraphs of one article into chunks of at most N
characters. A chunk never crosses an article boundary, and an over-long unit is split at
sentence or clause boundaries. Each chunk lists the citation IDs it contains.

**Why.** Mixing two articles in one chunk produces citations that are half wrong. Sizes
are in characters so that both configurations stay under the 512-token input limit of
`bge-small`. A "1024-token" chunk would be silently truncated by the embedder, and the
comparison would measure truncation rather than chunk size.

**Alternative rejected.** Fixed-size sliding windows with overlap. They are simpler but
blur the paragraph boundaries that citations rely on.

**Failure case.** The AI Act's Article 3 (68 definitions in about 17,000 characters) is one
unit. It is split into sentence-bounded pieces that all carry the citation "AI Act Art. 3",
so a hit on a definition still cites the right article.

**Questions**
1. *Why would larger chunks help or hurt?* They carry more context per hit but dilute the
   embedding. The 512 vs 1024 comparison measures which effect dominates on this eval set.
2. *Why prepend the heading when embedding?* Article titles ("Penalties", "Record-keeping")
   add topic words that a paragraph's text often lacks.
3. *Is anything lost?* No; a test reassembles split pieces and checks the words match.

## `finreg/embed.py`, `finreg/index.py`: embeddings and Qdrant

**What it does.** Embeds chunks with `BAAI/bge-small-en-v1.5` through fastembed (an ONNX
runtime, no torch) and stores them in Qdrant in embedded mode (on-disk, no server), with
one collection per chunk size. Search returns ranked hits with their citations.

**Why.** Local embeddings cost nothing per query and run on CPU. fastembed avoids a
several-hundred-MB torch install, which matters for the Docker image and CI. Embedded
Qdrant gives the production client API without running a server.

**Alternative rejected.** sentence-transformers (needs torch). Also a hosted embedding API
(costs money per query and needs a key for every test run).

**Failure case.** No model in CI. Tests use `HashEmbedder`, a deterministic bag of hashed
words, so retrieval logic is tested without downloading anything.

**Questions**
1. *Why cosine?* bge embeddings are trained for cosine similarity.
2. *Why the query prefix?* bge was trained with an instruction prefix on queries;
   fastembed's `query_embed` adds it and `passage_embed` does not.
3. *How does it scale?* For about 1.5k chunks, exact search is fine. Qdrant's HNSW index
   takes over at larger sizes with the same API.

## `finreg/answer.py`, `finreg/api.py`: answers and HTTP

**What it does.** Sends the top-k chunks as numbered sources to Claude Haiku 5.5 through the
official `anthropic` SDK, with the instruction to cite every claim as [n] or reply
NOT_FOUND. Cited numbers are mapped back to legal citations. FastAPI exposes `/search`
(no LLM) and `/ask`.

**Why.** Numbered sources make citations machine-checkable, which is what the citation
accuracy metric measures. Keeping the LLM behind one function makes the API and the
evaluation testable with a fake.

**Alternative rejected.** Letting the model quote article numbers from memory. Those
cannot be checked against what was retrieved.

**Failure case.** No `ANTHROPIC_API_KEY`: `/ask` returns 503 and `/search` keeps working.
A refusal (`stop_reason == "refusal"`) raises instead of returning an empty answer. For
the Sonnet judge, the server-side fallback re-runs a refused request on a fallback model;
cost is then estimated at Sonnet prices.

**Questions**
1. *Why Haiku 5.5?* The task is extractive and short. At $0.10/$0.50 per 1M tokens, 60
   questions cost fractions of a cent; the eval measures whether it is good enough.
2. *Why no temperature?* Haiku 5.5 rejects non-default sampling values.
3. *How is faithfulness judged?* A stronger model (Sonnet 5.5) labels each answer
   SUPPORTED or UNSUPPORTED given only the cited sources. It is a model judgment, so a
   sample should be hand-checked.

## `finreg/evaluate.py` and `evals/questions.jsonl`

**What it does.** For each chunk size, it builds the index if missing and runs 60
questions. It records the rank of the first chunk that contains the gold paragraph, and
reports hit@k, MRR@10, a lenient article-level hit@5 and latency. With `--answers` it adds
citation accuracy, faithfulness, NOT_FOUND rate and $/question.

**Why.** Paragraph-level gold labels make hit@k exact. The article-level metric shows how
often retrieval finds the right article but a neighbouring paragraph. Every question
quotes its source verbatim, and a test checks each quote against the parsed corpus, so a
bad label fails CI.

**Alternative rejected.** LLM-generated questions with LLM-judged relevance. Labels would
be unverifiable, and the same model family would sit on both sides of the eval.

**Failure case.** A question whose answer spans two paragraphs is scored against one gold
paragraph; it shows up as an article hit but a paragraph miss.

**Questions**
1. *Who wrote the questions?* They were drafted with tooling from the legal text and are
   marked "draft, not yet reviewed by author".
2. *Why only 60?* It is enough to see large differences (a 95% interval at n=60 is about
   ±12 points) and small enough to review by hand.
3. *What would you add next?* A reranker (cross-encoder) as a third configuration, and
   questions with no answer in the corpus, to measure NOT_FOUND behaviour.
