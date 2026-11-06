FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project definition and source
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

# Install dependencies into system environment
RUN uv pip install --system -e .

# Expose Web UI & REST API port
EXPOSE 8484

# Default environment for persistent sqlite database
ENV SYNAPSE_DB_PATH=/data/memories.db

VOLUME ["/data"]

CMD ["synapse", "ui", "--host", "0.0.0.0", "--port", "8484"]
