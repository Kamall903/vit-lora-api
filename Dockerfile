FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir \
    torch==2.11.0 \
    torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu
    COPY requirements-docker.txt .
RUN pip install --no-cache-dir \
    -r requirements-docker.txt
COPY app/ ./app/
COPY models/vit_quantized_int8.pt ./models/
COPY models/model_metadata.json ./models/
COPY models/model_assets/ ./models/model_assets/
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]