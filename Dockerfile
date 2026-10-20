FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

ARG DINOV2_REF=main
RUN git clone https://github.com/facebookresearch/dinov2 /opt/dinov2 \
    && git -C /opt/dinov2 checkout ${DINOV2_REF}

ARG WITH_PRETRAINED=1
ENV TORCH_HOME=/opt/torch
RUN if [ "$WITH_PRETRAINED" = "1" ]; then \
      mkdir -p $TORCH_HOME/hub/checkpoints && \
      curl -fL -o $TORCH_HOME/hub/checkpoints/dinov2_vitb14_pretrain.pth \
        https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth; \
    fi

COPY src ./src
COPY schema ./schema
COPY annotations ./annotations

ENV PYTHONPATH=/app/src \
    LOADEST_DINOV2_REPO=/opt/dinov2 \
    LOADEST_SKIP_PRETRAINED=1 \
    LOADEST_MODELS_DIR=/models \
    LOADEST_DATA_DIR=/var/loadest \
    LOADEST_TRAIN_DIR=/data/train

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fs http://localhost:8000/api/v1/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
