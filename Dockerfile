FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ECOCLASIFICA_MODEL_PATH=/app/models/ecoclasifica-resnet50 \
    ECOCLASIFICA_FRONTEND_DIST=/app/frontend/dist

# Dependencias de sistema (OpenCV/Pillow/torchvision)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias Python (capa cacheable)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código fuente
COPY src/ ./src/

# Interfaz web (build de producción de React/Vite)
COPY frontend/dist/ ./frontend/dist/

# Archivos del modelo ResNet-50 afinado
COPY models/ecoclasifica-resnet50/ ./models/ecoclasifica-resnet50/

EXPOSE 8000

CMD ["uvicorn", "src.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
