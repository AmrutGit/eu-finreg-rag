"""Build and search a Qdrant collection of chunks (embedded/local mode: no server).

    python -m finreg.index --chunk-chars 512      # -> data/index, collection chunks_512

One collection per chunk size, so configurations can be compared on the same store.
Payloads carry everything needed to cite a hit without another lookup.
"""
import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from finreg.chunk import Chunk, chunk_units
from finreg.embed import Embedder, FastEmbedder


@dataclass(frozen=True)
class Hit:
    rank: int
    score: float
    chunk_id: str
    citation: str
    unit_ids: tuple[str, ...]
    doc: str
    article: str
    heading: str
    text: str


def collection_name(chunk_chars: int) -> str:
    return f"chunks_{chunk_chars}"


def open_client(path: str | None):
    """path=None -> in-memory (tests); otherwise an on-disk embedded store."""
    from qdrant_client import QdrantClient

    return QdrantClient(location=":memory:") if path is None else QdrantClient(path=path)


def build(client, embedder: Embedder, chunks: list[Chunk], name: str) -> int:
    from qdrant_client import models

    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=models.VectorParams(size=embedder.dim,
                                                                      distance=models.Distance.COSINE))
    vectors = embedder.embed_passages([f"{c.heading}\n{c.text}" for c in chunks])  # heading adds topic words
    client.upsert(name, points=[
        models.PointStruct(id=i, vector=v, payload={**asdict(c), "unit_ids": list(c.unit_ids)})
        for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))])
    return len(chunks)


def search(client, embedder: Embedder, name: str, query: str, k: int = 5) -> list[Hit]:
    res = client.query_points(name, query=embedder.embed_query(query), limit=k, with_payload=True).points
    out = []
    for rank, p in enumerate(res, start=1):
        pl = p.payload
        out.append(Hit(rank, float(p.score), pl["id"], "; ".join(pl["unit_ids"]), tuple(pl["unit_ids"]),
                       pl["doc"], pl["article"], pl["heading"], pl["text"]))
    return out


def load_units(path: str = "data/processed/units.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--chunk-chars", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--units", default="data/processed/units.jsonl")
    ap.add_argument("--index-dir", default="data/index")
    args = ap.parse_args(argv)
    Path(args.index_dir).mkdir(parents=True, exist_ok=True)
    client, emb, units = open_client(args.index_dir), FastEmbedder(), load_units(args.units)
    for n in args.chunk_chars:
        start = time.perf_counter()
        count = build(client, emb, chunk_units(units, n), collection_name(n))
        print(f"{collection_name(n)}: {count} chunks embedded with {emb.name} in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
