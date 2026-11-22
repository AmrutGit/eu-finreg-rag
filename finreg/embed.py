"""Embedders. Production: a local ONNX model through fastembed (no API cost, no torch).
Tests: a deterministic hashing embedder, so CI downloads no model.

BAAI/bge-small-en-v1.5 (384-d, 512-token input limit) is the default: small enough for
CPU indexing of ~1.5k chunks in a few minutes, and an established English retrieval
baseline. Queries get bge's retrieval instruction prefix (fastembed's `query_embed`);
passages are embedded as-is, which is how the model was trained.
"""
import hashlib
import math
import re
from typing import Protocol

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    name: str
    dim: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | None = ".cache/fastembed"):
        from fastembed import TextEmbedding  # lazy: heavy import, not needed in tests

        self.name = model
        self._m = TextEmbedding(model_name=model, cache_dir=cache_dir)
        self.dim = len(self.embed_query("dimension probe"))

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._m.passage_embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._m.query_embed(text))).tolist()


class HashEmbedder:
    """Bag of hashed lowercase word tokens, L2-normalised. Lexical overlap only: good
    enough to make retrieval tests meaningful without a model."""

    def __init__(self, dim: int = 256):
        self.name, self.dim = f"hash-{dim}", dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
