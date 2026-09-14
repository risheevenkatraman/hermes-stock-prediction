FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    HOME=/home/hermes

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 hermes \
    && mkdir /data && chown hermes:hermes /data
COPY requirements.txt ./
# CPU wheels avoid installing the CUDA runtime on the x86 EC2 instance.
RUN python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt
COPY backend/ ./backend/
COPY index.html styles.css app.js config.js ./
USER hermes
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8)"
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "8", "--no-proxy-headers"]
