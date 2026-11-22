from pathlib import Path

from finreg.ingest import parse
from finreg.sources import Source

SRC = Source("AI Act", "32024R1689", "test")
FIXTURE = Path(__file__).parent / "fixtures" / "mini_oj.xhtml"


def test_parse_units_and_citation_ids():
    units = {u.id: u for u in parse(FIXTURE.read_bytes(), SRC)}
    assert set(units) == {"AI Act Art. 1", "AI Act Art. 99(1)", "AI Act Art. 99(3)", "AI Act Annex III"}
    assert units["AI Act Art. 99(3)"].heading == "Penalties"
    assert units["AI Act Art. 99(3)"].text == "3. Fines of up to EUR 35 000 000 or 7 % of turnover. (a) whichever is higher;"
    assert units["AI Act Art. 1"].text == "This Regulation lays down uniform requirements."  # whitespace collapsed
    assert units["AI Act Annex III"].article == "Annex III"


def test_recitals_are_excluded():
    assert not any("recital" in u.text for u in parse(FIXTURE.read_bytes(), SRC))


def test_source_urls():
    assert SRC.fetch_url == "http://publications.europa.eu/resource/celex/32024R1689"
    assert SRC.eurlex_url.endswith("CELEX:32024R1689")
