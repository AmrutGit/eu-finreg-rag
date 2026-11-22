"""Units -> chunks of at most `max_chars` characters, never crossing an article boundary.

Consecutive paragraphs of the same article are packed together until the next one would
exceed `max_chars`. A single paragraph longer than `max_chars` (Annex III, long
definition lists) is split at sentence or clause boundaries. Each chunk keeps the
citation IDs of every unit it contains, so an answer that cites the chunk can name
"AI Act Art. 99(3)" rather than "chunk 412".

Sizes are in characters, so both configurations (512 and 1024) stay below the embedding
model's 512-token input limit (~1 token per 4 characters of English legal text). A
1024-*token* chunk would be silently truncated by the embedder.
"""
import re
from dataclasses import dataclass

_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(0-9])")


@dataclass(frozen=True)
class Chunk:
    id: str                  # "<first unit id>#<n>", unique within one chunking
    doc: str
    article: str
    heading: str
    unit_ids: tuple[str, ...]
    text: str

    @property
    def citation(self) -> str:
        """Human-readable source, e.g. 'AI Act Art. 99(1); AI Act Art. 99(2)'."""
        return "; ".join(self.unit_ids)


def _pieces(text: str, max_chars: int) -> list[str]:
    """Split one over-long text into pieces <= max_chars, preferring sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    out, cur = [], ""
    for sent in _SPLIT.split(text):
        while len(sent) > max_chars:  # no boundary at all: hard split on whitespace
            cut = sent.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                out.append(cur)
                cur = ""
            out.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if cur and len(cur) + 1 + len(sent) > max_chars:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_units(units: list[dict], max_chars: int) -> list[Chunk]:
    """`units` are dicts as written by finreg.ingest (id, doc, article, heading, text), in document order."""
    chunks: list[Chunk] = []

    def emit(group: list[dict], text: str):
        first = group[0]
        chunks.append(Chunk(f"{first['id']}#{len(chunks)}", first["doc"], first["article"], first["heading"],
                            tuple(u["id"] for u in group), text))

    group: list[dict] = []
    text = ""
    for u in units:
        same_article = group and (group[0]["doc"], group[0]["article"]) == (u["doc"], u["article"])
        if group and (not same_article or len(text) + 1 + len(u["text"]) > max_chars):
            emit(group, text)
            group, text = [], ""
        if len(u["text"]) > max_chars:
            for piece in _pieces(u["text"], max_chars):
                emit([u], piece)
            continue
        group.append(u)
        text = f"{text} {u['text']}".strip()
    if group:
        emit(group, text)
    return chunks
