ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim-bullseye AS builder

# Copy uv from GitHub container registry
COPY --from=ghcr.io/astral-sh/uv:0.6.9 /uv /bin/uv

# Set uv environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install dependencies but not the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy the application and install it with dependencies
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Then, use a final image without uv
FROM python:${PYTHON_VERSION}-slim-bullseye AS final

# Install just, jq
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends curl ca-certificates jq && \
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual environment to /opt/venv
COPY --from=builder /app/.venv /opt/venv

# Copy the application from the builder (without the virtual environment)
COPY --from=builder /app /app
# Remove the .venv directory in /app since we've moved it to /opt/venv
RUN rm -rf /app/.venv

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Create a non-root user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/bin/bash" \
    --no-create-home \
    --uid "${UID}" \
    appuser && \
    mkdir -p /app/logs /app/results && \
    chmod -R 777 /app/logs /app/results && \
    chown -R appuser:appuser /app/logs /app/results

# Create entrypoint before switching to appuser
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'python -m convfinqa --help' >> /app/entrypoint.sh && \
    echo 'exec "$@"' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Switch to non-root user
USER appuser

# Create volumes for logs and cache
VOLUME ["/app/logs", "/app/.cache"]

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/bin/bash"]
