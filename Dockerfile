# ---------------------------------------------------------------------------
# AI Servo-Motor Predictive Maintenance Prototype — runtime image
# ---------------------------------------------------------------------------
# A single image is shared by the FastAPI and Streamlit services; the command
# is overridden in docker-compose.yml.
#
# Build:   docker build -t pmm-app:latest .
# Run API: docker run --rm -p 8000:8000 -v $(pwd)/data:/app/data \
#                                       -v $(pwd)/outputs:/app/outputs \
#                                       pmm-app:latest
# Or just: docker compose up
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS runtime

# libgomp1 is required at runtime by LightGBM / XGBoost wheels on linux/amd64.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python dependencies first so the layer is cached across code edits.
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Application code.  Note: data/ and outputs/ are mounted as volumes at run
# time so the container does NOT bake in the dataset or trained models.
COPY config.yaml ./
COPY src/ ./src/
COPY app/ ./app/

# Document both ports; docker-compose picks the right command per service.
EXPOSE 8000 8501

# Default command = FastAPI.  docker-compose overrides this for the UI.
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
