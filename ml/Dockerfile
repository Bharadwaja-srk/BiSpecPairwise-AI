FROM python:3.11-slim
WORKDIR /app
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt
COPY feature_engineering/ feature_engineering/
COPY ml/ ml/
COPY models/ models/
ENV MODEL_PATH=models/xgb_pairwise_v1.pkl ML_PORT=8001 PYTHONPATH=/app
EXPOSE 8001
CMD ["sh", "-c", "uvicorn ml.service.app:app --host 0.0.0.0 --port ${ML_PORT:-8001}"]
