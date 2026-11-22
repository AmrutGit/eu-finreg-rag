"""Fetch the corpus and parse it into citable units.

    python -m finreg.ingest            # -> data/raw/<celex>.html, data/processed/units.jsonl

A unit is one numbered paragraph of an article (citation "AI Act Art. 99(3)"), a whole
article when it has no numbered paragraphs ("DORA Art. 1"), or a whole annex
("AI Act Annex III"). Recitals are left out: they explain the law but do not state
obligations, and they would compete with the articles in retrieval.
"""
import argparse
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from finreg.sources import SOURCES, Source

X = "{http://www.w3.org/1999/xhtml}"
_PARA_ID = re.compile(r"^\d{3}\.(\d{3})$")


@dataclass(frozen=True)
class Unit:
    id: str          # citation, e.g. "AI Act Art. 99(3)"
    doc: str         # "AI Act" | "DORA"
    article: str     # "Art. 99" | "Annex III"
    heading: str     # article title, e.g. "Penalties"
    text: str


def _text(el) -> str:
    return " ".join(" ".join(el.itertext()).split())


def parse(xhtml: str | bytes, src: Source) -> list[Unit]:
    root = ET.fromstring(xhtml)
    units: list[Unit] = []
    for div in root.iter(X + "div"):
        div_id = div.get("id") or ""
        if re.fullmatch(r"art_\d+", div_id):
            num = div_id.split("_")[1]
            title_div = next((c for c in div if (c.get("id") or "").endswith(".tit_1")), None)
            heading = _text(title_div) if title_div is not None else ""
            article = f"Art. {num}"
            paras = [c for c in div if _PARA_ID.match(c.get("id") or "")]
            if paras:
                for p in paras:
                    n = int(_PARA_ID.match(p.get("id")).group(1))
                    units.append(Unit(f"{src.key} {article}({n})", src.key, article, heading, _text(p)))
            else:  # single-paragraph article: everything after the number and title
                body = [c for c in div if c is not title_div and c.get("class") != "oj-ti-art"]
                text = " ".join(_text(c) for c in body)
                if text:
                    units.append(Unit(f"{src.key} {article}", src.key, article, heading, text))
        elif re.fullmatch(r"anx_[IVXLC]+", div_id):
            numeral = div_id.split("_")[1]
            full = _text(div)
            heading = re.sub(rf"^ANNEX {numeral}\s*", "", full)[:120]
            units.append(Unit(f"{src.key} Annex {numeral}", src.key, f"Annex {numeral}", heading, full))
    return units


def fetch(src: Source, raw_dir: Path) -> Path:
    path = raw_dir / f"{src.celex}.html"
    if not path.exists():
        req = urllib.request.Request(src.fetch_url, headers={
            "Accept": "application/xhtml+xml;q=0.9,text/html;q=0.8", "Accept-Language": "eng"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/processed/units.jsonl")
    args = ap.parse_args(argv)
    raw, out = Path(args.raw_dir), Path(args.out)
    raw.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for src in SOURCES:
            units = parse(fetch(src, raw).read_bytes(), src)
            for u in units:
                f.write(json.dumps(asdict(u), ensure_ascii=False) + "\n")
            arts = {u.article for u in units}
            print(f"{src.key}: {len(units)} units from {len(arts)} articles/annexes ({src.fetch_url})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
