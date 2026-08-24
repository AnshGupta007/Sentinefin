FROM python:3.11-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# CPU-only torch keeps the image small; swap for the CUDA wheel on GPU hosts.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e . && pip install -e .

CMD ["sentinefin", "pipeline", "--smoke"]
