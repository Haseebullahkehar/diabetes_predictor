FROM python:3.12-slim

# Keep Python lean and predictable in containers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project (model artifacts are built at image-build time below).
COPY . .

# Train the model during the build so the image ships ready-to-serve.
RUN python train.py

EXPOSE 8000

# Container healthcheck against the API (uses $PORT if the platform injects one).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health').status==200 else 1)"

# Bind to the platform-provided $PORT (Render/Railway inject it); default 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
