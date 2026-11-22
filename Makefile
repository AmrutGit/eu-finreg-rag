.PHONY: install ingest index eval eval-answers api test lint docker
install: ; pip install -r requirements.txt -r requirements-dev.txt
ingest:  ; python -m finreg.ingest                       # EUR-Lex texts -> data/processed/units.jsonl
index:   ; python -m finreg.index --chunk-chars 512 1024   # local embeddings -> Qdrant (data/index)
eval:    ; python -m finreg.evaluate --chunk-chars 512 1024 # retrieval metrics, no API cost
eval-answers: ; python -m finreg.evaluate --chunk-chars 512 1024 --answers --out results/eval_answers.json
api:     ; uvicorn finreg.api:app --port 8000
test:    ; pytest -q
lint:    ; ruff check .
docker:  ; docker build -t eu-finreg-rag . && docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY eu-finreg-rag
