# Single container: FastAPI + Qdrant in embedded mode + local ONNX embeddings.
# The corpus is fetched and indexed at build time, so the image serves /search offline.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY finreg/ finreg/
RUN python -m finreg.ingest && python -m finreg.index --chunk-chars 512
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
# ANTHROPIC_API_KEY is passed at run time (docker run -e ANTHROPIC_API_KEY); /search works without it.
CMD ["uvicorn", "finreg.api:app", "--host", "0.0.0.0", "--port", "8000"]
