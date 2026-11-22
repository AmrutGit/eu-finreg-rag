"""The corpus: official EU texts, fetched from the Publications Office of the EU (Cellar).

EUR-Lex's HTML pages sit behind a bot challenge, so the documents are fetched through the
Publications Office's content-negotiated CELEX resource, which serves the same Official
Journal text as XHTML. Reuse: EU legal texts on EUR-Lex/Cellar may be reused with
acknowledgement under Commission Decision 2011/833/EU; only the authentic Official
Journal version is legally binding.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    key: str      # short name used in citations
    celex: str
    title: str

    @property
    def fetch_url(self) -> str:
        return f"http://publications.europa.eu/resource/celex/{self.celex}"

    @property
    def eurlex_url(self) -> str:
        return f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{self.celex}"


SOURCES = [
    Source("AI Act", "32024R1689",
           "Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)"),
    Source("DORA", "32022R2554",
           "Regulation (EU) 2022/2554 on digital operational resilience for the financial sector (DORA)"),
]
