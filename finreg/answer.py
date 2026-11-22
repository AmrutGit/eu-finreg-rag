"""Answer a question from retrieved chunks, citing them by number.

The model sees the top-k chunks as numbered sources [1]..[k] and must answer only from
them, citing each claim with [n]. If the sources do not contain the answer, it must reply
NOT_FOUND. The cited numbers are mapped back to legal citations (e.g. "DORA Art. 19(1)").

The LLM sits behind `Generate = (system, user) -> (text, in_tokens, out_tokens)`, so tests
and the API run with a fake, and the provider is one function.
"""
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from finreg.index import Hit

Generate = Callable[[str, str], tuple[str, int, int]]

MODEL = "claude-haiku-5-5"
# USD per 1M tokens (input, output), standard rates; Haiku 5.5 rate is for prompts <= 100k tokens.
# Source: https://platform.claude.com/docs/en/about-claude/pricing, read 2026-11-22.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}


def cost_usd(model: str, tin: int, tout: int) -> float:
    p_in, p_out = PRICES[model]
    return (tin * p_in + tout * p_out) / 1_000_000

SYSTEM = """\
You answer questions about EU financial-sector regulation (the EU AI Act and DORA) using ONLY \
the numbered sources provided. Rules:
- Every sentence that states a fact must end with the number(s) of the source(s) it comes from, e.g. [2] or [1][3].
- Do not use outside knowledge. Do not cite a source that does not support the sentence.
- If the sources do not contain the answer, reply exactly: NOT_FOUND
- Be concise: at most 5 sentences. Quote amounts, deadlines and thresholds exactly as written."""


@dataclass
class Answer:
    text: str
    cited: list[int]                         # source numbers cited, in order of first use
    citations: list[str]                     # legal citations of the cited sources
    not_found: bool
    in_tokens: int = 0
    out_tokens: int = 0
    sources: list[Hit] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return cost_usd(MODEL, self.in_tokens, self.out_tokens)


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [f"[{i}] {h.citation} ({h.heading})\n{h.text}" for i, h in enumerate(hits, start=1)]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def parse_citations(text: str, n_sources: int) -> list[int]:
    seen: list[int] = []
    for m in re.finditer(r"\[(\d+)\]", text):
        i = int(m.group(1))
        if 1 <= i <= n_sources and i not in seen:
            seen.append(i)
    return seen


def answer(question: str, hits: list[Hit], generate: Generate) -> Answer:
    text, tin, tout = generate(SYSTEM, build_prompt(question, hits))
    text = text.strip()
    not_found = text.upper().startswith("NOT_FOUND")
    cited = [] if not_found else parse_citations(text, len(hits))
    return Answer(text, cited, [hits[i - 1].citation for i in cited], not_found, tin, tout, hits)


def claude_generate(model: str = MODEL, effort: str = "medium") -> Generate:
    """Claude via the official SDK. Haiku 5.5 thinks by default and rejects non-default
    sampling parameters, so no temperature is sent; effort is set explicitly."""
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic()

    def generate(system: str, user: str) -> tuple[str, int, int]:
        kwargs = dict(model=model, max_tokens=4096, system=system, output_config={"effort": effort},
                      messages=[{"role": "user", "content": user}])
        if model == "claude-sonnet-5-5":
            # Server-side refusal fallback: a declined request is re-run on a fallback model
            # inside the same call. Not available for Haiku 5.5.
            resp = client.beta.messages.create(**kwargs, betas=["server-side-fallback-2026-07-01"],
                                               fallbacks="default")
        else:
            resp = client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {resp.stop_details}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens

    return generate
