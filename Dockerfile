FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installer PyTorch CPU uniquement (beaucoup plus léger — 200MB au lieu de 3.6GB)
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Installer le reste des dépendances
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.0 \
    timm==1.0.3 \
    albumentations==1.4.7 \
    opencv-python-headless==4.9.0.80 \
    Pillow==10.3.0 \
    python-multipart==0.0.9 \
    faiss-cpu==1.8.0 \
    anthropic==0.28.0 \
    mlflow==2.13.0 \
    pandas==2.2.2 \
    evidently==0.4.33 \
    prometheus-client==0.20.0

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
