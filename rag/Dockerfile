FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
COPY rag/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rag/ ./rag/
COPY ml/explainability/ ./ml/explainability/
COPY ml/__init__.py ./ml/__init__.py
COPY feature_engineering/feature_schema.py ./feature_engineering/feature_schema.py
COPY feature_engineering/__init__.py ./feature_engineering/__init__.py
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8002
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health', timeout=4)"
CMD ["sh", "-c", "uvicorn rag.app:app --host 0.0.0.0 --port ${RAG_PORT:-8002}"]
