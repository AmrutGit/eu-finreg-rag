from finreg.chunk import _pieces, chunk_units


def _u(uid, article, text, doc="DORA"):
    return {"id": uid, "doc": doc, "article": article, "heading": "h", "text": text}


def test_packs_paragraphs_of_one_article_up_to_limit():
    units = [_u("DORA Art. 5(1)", "Art. 5", "a" * 40), _u("DORA Art. 5(2)", "Art. 5", "b" * 40),
             _u("DORA Art. 5(3)", "Art. 5", "c" * 40)]
    chunks = chunk_units(units, max_chars=90)
    assert [c.unit_ids for c in chunks] == [("DORA Art. 5(1)", "DORA Art. 5(2)"), ("DORA Art. 5(3)",)]
    assert all(len(c.text) <= 90 for c in chunks)


def test_never_crosses_article_boundary():
    units = [_u("DORA Art. 5(1)", "Art. 5", "short"), _u("DORA Art. 6(1)", "Art. 6", "short")]
    chunks = chunk_units(units, max_chars=1000)
    assert [c.article for c in chunks] == ["Art. 5", "Art. 6"]


def test_long_paragraph_split_at_sentences_and_keeps_citation():
    text = "First sentence here. Second one follows; third clause: (a) item. " * 5
    chunks = chunk_units([_u("AI Act Annex III", "Annex III", text.strip(), doc="AI Act")], max_chars=60)
    assert len(chunks) > 1 and all(len(c.text) <= 60 for c in chunks)
    assert {c.unit_ids for c in chunks} == {("AI Act Annex III",)}
    assert " ".join(c.text for c in chunks).split() == text.split()  # nothing lost or reordered


def test_hard_split_when_no_boundary():
    assert all(len(p) <= 10 for p in _pieces("x" * 35, 10))
    assert "".join(_pieces("x" * 35, 10)) == "x" * 35


def test_chunk_ids_unique_and_citation():
    units = [_u(f"DORA Art. 5({i})", "Art. 5", "t" * 30) for i in range(1, 6)]
    chunks = chunk_units(units, max_chars=70)
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks[0].citation == "DORA Art. 5(1); DORA Art. 5(2)"
