Retrieval, 60 questions, embedder `BAAI/bge-small-en-v1.5`, Qdrant cosine, top-10:

| Chunk size (chars) | Chunks | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | article hit@5 | p50 / p95 latency |
|---|---|---|---|---|---|---|---|---|
| 512 | 1514 | 57% | 82% | 87% | 92% | 0.695 | 93% | 18 / 26 ms |
| 1024 | 736 | 75% | 88% | 93% | 95% | 0.820 | 98% | 14 / 20 ms |
