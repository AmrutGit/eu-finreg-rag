import json
from pathlib import Path

import pytest

from finreg import answer as ans
from finreg import evaluate, index
from finreg.chunk import chunk_units
from finreg.embed import HashEmbedder

UNITS = [
    {"id": "DORA Art. 19(1)", "doc": "DORA", "article": "Art. 19", "heading": "Reporting of major ICT-related incidents",
     "text": "Financial entities shall report major ICT-related incidents to the relevant competent authority."},
    {"id": "DORA Art. 26(1)", "doc": "DORA", "article": "Art. 26", "heading": "Advanced testing based on TLPT",
     "text": "Financial entities shall carry out at least every 3 years advanced testing by means of TLPT."},
    {"id": "AI Act Art. 99(3)", "doc": "AI Act", "article": "Art. 99", "heading": "Penalties",
     "text": "Prohibited AI practices are subject to administrative fines of up to EUR 35 000 000 or 7 % of turnover."},
]


@pytest.fixture
def store():
    client, emb = index.open_client(None), HashEmbedder()
    index.build(client, emb, chunk_units(UNITS, 512), "chunks_512")
    return client, emb


def test_search_returns_cited_hits(store):
    hits = index.search(*store, "chunks_512", "How often is TLPT advanced testing required?", k=2)
    assert hits[0].unit_ids == ("DORA Art. 26(1)",) and hits[0].rank == 1
    assert hits[0].citation == "DORA Art. 26(1)" and hits[0].heading.startswith("Advanced testing")


def test_rebuild_replaces_collection(store):
    client, emb = store
    index.build(client, emb, chunk_units(UNITS[:1], 512), "chunks_512")
    assert len(index.search(client, emb, "chunks_512", "anything", k=10)) == 1


def _fake(reply, tin=100, tout=20):
    calls = []

    def gen(system, user):
        calls.append((system, user))
        return reply, tin, tout
    gen.calls = calls
    return gen


def test_answer_maps_citations_and_cost(store):
    hits = index.search(*store, "chunks_512", "fines for prohibited AI practices", k=3)
    gen = _fake("Fines reach EUR 35 000 000 [1]. See also [1][7].")
    a = ans.answer("max fine?", hits, gen)
    assert a.cited == [1] and a.citations == [hits[0].citation]  # [7] is out of range: ignored
    assert "[1] " + hits[0].citation in gen.calls[0][1] and "Question: max fine?" in gen.calls[0][1]
    assert a.cost_usd == pytest.approx((100 * 0.10 + 20 * 0.50) / 1e6)


def test_not_found_has_no_citations(store):
    hits = index.search(*store, "chunks_512", "weather", k=2)
    a = ans.answer("weather?", hits, _fake("NOT_FOUND"))
    assert a.not_found and a.cited == [] and a.citations == []


def test_ranks_and_metrics():
    h = [index.Hit(i, 1.0, f"c{i}", u, (u,), u.split(" Art.")[0], "Art. " + u.split("Art. ")[1].split("(")[0], "", "")
         for i, u in enumerate(["DORA Art. 19(2)", "DORA Art. 19(1)"], start=1)]
    assert evaluate.gold_rank(h, "DORA Art. 19(1)") == 2
    assert evaluate.article_rank(h, "DORA Art. 19(1)") == 1   # same article, different paragraph
    assert evaluate.gold_rank(h, "DORA Art. 5(1)") is None
    m = evaluate.retrieval_metrics([evaluate.QResult("a", 1, 1, 0.01), evaluate.QResult("b", 4, 2, 0.03),
                                    evaluate.QResult("c", None, None, 0.02)])
    assert m["hit@1"] == pytest.approx(1 / 3) and m["hit@5"] == pytest.approx(2 / 3)
    assert m["mrr@10"] == pytest.approx((1 + 0.25) / 3) and m["p50_s"] == 0.02


def test_run_config_with_fake_llm_and_judge(store):
    client, emb = store
    qs = [{"id": "q1", "question": "How often must TLPT be done?", "source": "DORA Art. 26(1)"}]
    r = evaluate.run_config(client, emb, UNITS, qs, 512, _fake("Every 3 years [1]."), _fake("SUPPORTED", 50, 1))
    assert r["retrieval"]["hit@1"] == 1.0
    a = r["answers"]
    assert a["citation_accuracy"] == 1.0 and a["faithfulness"] == 1.0 and a["not_found_rate"] == 0.0
    assert "| 512 |" in evaluate.render([r], "hash", 1)


QUESTIONS = Path(__file__).parents[1] / "evals" / "questions.jsonl"
UNITS_FILE = Path(__file__).parents[1] / "data" / "processed" / "units.jsonl"


def test_question_set_shape():
    qs = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]
    assert len(qs) == 60 and len({q["id"] for q in qs}) == 60
    assert sum(q["doc"] == "AI Act" for q in qs) == 30 and sum(q["doc"] == "DORA" for q in qs) == 30
    assert all(q["question"] and q["reference_answer"] and q["quote"] for q in qs)
    assert all(q["status"] == "draft, not yet reviewed by author" for q in qs)


@pytest.mark.skipif(not UNITS_FILE.exists(), reason="run `python -m finreg.ingest` first")
def test_every_quote_is_verbatim_in_its_source():
    units = {json.loads(line)["id"]: json.loads(line)["text"] for line in UNITS_FILE.read_text(encoding="utf-8").splitlines()}
    for q in (json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()):
        assert q["source"] in units, q["id"]
        assert q["quote"] in units[q["source"]], q["id"]
