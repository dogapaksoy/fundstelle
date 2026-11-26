FROM python:3.12-slim

# llama-cpp-python compiles llama.cpp on install
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[retrieval,answer,serve]"

# statutes are fetched at build time; models download on first start into the HF cache
RUN fundstelle --data /app/data download
ENV FUNDSTELLE_DATA=/app/data \
    FUNDSTELLE_RETRIEVER=bm25+dense:intfloat/multilingual-e5-small \
    FUNDSTELLE_LLM=llama:qwen3-4b \
    HF_HOME=/app/hf

EXPOSE 8000
CMD ["fundstelle", "--data", "/app/data", "serve", "--host", "0.0.0.0", "--port", "8000"]
