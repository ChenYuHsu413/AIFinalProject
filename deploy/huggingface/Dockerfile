# ---------------------------------------------------------------------------
# Hugging Face Spaces (Docker SDK) — FastAPI backend on port 7860
#
# Use this as the *root* Dockerfile of a Hugging Face **Docker Space**. The free
# CPU-basic tier (2 vCPU / 16 GB RAM) comfortably loads the sklearn servo models.
#
# Setup: see docs/DEPLOYMENT.md §9 (免費部署替代方案).
#   - app_port 7860 is declared in deploy/huggingface/README.md frontmatter.
#   - The frontend (Vercel) calls this Space's URL directly; backend CORS is
#     already allow_origins=["*"], so no change is needed.
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# libgomp1 is required at runtime by LightGBM / XGBoost wheels on linux/amd64.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# requirements-dev.txt pulls in requirements.txt via its leading -r and adds the
# FastAPI backend deps (same as the main Dockerfile).
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements-dev.txt

# Application code + committed runtime data the backend reads at request time:
#   - outputs/models, outputs/metrics  (models + eval JSONs)
#   - data/processed/servo_*           (Servo features / demo / sample predictions —
#                                        needed by /servo/simulate, /servo/samples, /servo/fleet)
#   - data/knowledge/*.md              (knowledge base for /knowledge/*)
# (The root .dockerignore keeps these in context; IMS/XJTU raw datasets are NOT
#  bundled, so /ims/* and /xjtu/* degrade to available:false — the rest works.)
COPY config.yaml ./
COPY src/ ./src/
COPY app/ ./app/
COPY outputs/ ./outputs/
COPY data/ ./data/

EXPOSE 7860
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
