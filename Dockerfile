# Build stage: the full Python image has a compiler for llama.cpp, so no apt is needed.
FROM python:3.12 AS build
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_OPENMP=OFF" \
    PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[retrieval,answer,serve]"

# Runtime stage
FROM python:3.12-slim
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    FUNDSTELLE_DATA=/app/data \
    FUNDSTELLE_RETRIEVER=dense:intfloat/multilingual-e5-small \
    FUNDSTELLE_LLM=llama:qwen3-4b \
    HF_HOME=/app/hf
WORKDIR /app
# statutes are fetched at build time; models download on first start into $HF_HOME
RUN fundstelle --data /app/data download
EXPOSE 8000
CMD ["fundstelle", "--data", "/app/data", "serve", "--host", "0.0.0.0", "--port", "8000"]
