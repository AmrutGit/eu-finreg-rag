import pytest
from fastapi.testclient import TestClient

from finreg import index
from finreg.api import app
from finreg.chunk import chunk_units
from finreg.embed import HashEmbedder

UNITS = [{"id": "DORA Art. 64", "doc": "DORA", "article": "Art. 64", "heading": "Entry into force and application",
          "text": "It shall apply from 17 January 2025."}]


@pytest.fixture
def client():
    st = app.state
    st.client, st.embedder, st.collection = index.open_client(None), HashEmbedder(), "chunks_512"
    index.build(st.client, st.embedder, chunk_units(UNITS, 512), "chunks_512")
    yield TestClient(app)
    for attr in ("client", "embedder", "collection", "generate"):
        if hasattr(st, attr):
            delattr(st, attr)


def test_search(client):
    r = client.post("/search", json={"question": "When does DORA apply?", "k": 1})
    assert r.status_code == 200 and r.json()["hits"][0]["citation"] == "DORA Art. 64"


def test_ask_with_fake_llm(client):
    app.state.generate = lambda s, u: ("DORA applies from 17 January 2025 [1].", 80, 12)
    body = client.post("/ask", json={"question": "When does DORA apply?"}).json()
    assert body["citations"] == ["DORA Art. 64"] and body["not_found"] is False and body["cost_usd"] > 0


def test_ask_without_key_is_503(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert client.post("/ask", json={"question": "When does DORA apply?"}).status_code == 503


def test_validation(client):
    assert client.post("/search", json={"question": "", "k": 1}).status_code == 422
    assert client.post("/search", json={"question": "valid question", "k": 99}).status_code == 422
