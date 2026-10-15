# syntax=docker/dockerfile:1
#
# Multi-stage build: business logic in src/ is compiled with Cython into
# native extension modules; only the compiled .so files, app.py entrypoint,
# and __init__.py files ship in the runtime image. No .py source for the
# RAG/agent/tree_builder logic reaches the final image.
#
# Runs the same on Windows, Intel/Apple Silicon Mac, and Linux hosts —
# Docker Desktop on Windows/Mac always executes the linux/<arch> variant
# matching the host CPU, never a native Windows/macOS binary. `docker build`
# targets the host's own architecture; no registry push involved.

FROM python:3.12-bookworm AS builder
WORKDIR /src

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv export --frozen --no-dev --no-group build --no-hashes -o requirements.txt \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements.txt cython setuptools

COPY setup.py ./
COPY scripts/build_cython.sh scripts/build_cython.sh
COPY src ./src
RUN bash scripts/build_cython.sh

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app

# opencv-python (pulled in transitively by docling's rapidocr) needs these at
# import time even in headless use — python:3.12-slim strips them out.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /src/src ./src
RUN chown -R appuser:appuser /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501
USER appuser

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
