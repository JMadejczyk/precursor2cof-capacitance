# Use the stable full Debian base image
FROM ghcr.io/astral-sh/uv:debian

# Set the working directory
WORKDIR /app

# Set the PYTHONPATH so Python can find your modules and scripts
ENV PYTHONPATH=/app

## Force scientific libraries to run single-threaded. This prevents race
## conditions and reduces memory pressure, which are common causes of segfaults.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Install dependencies first for layer caching
COPY pyproject.toml .
# COPY uv.lock . # uncomment if you track uv.lock
RUN uv sync --all-extras --dev

# Copy configuration and source directories
COPY .pylintrc .
COPY modules/ ./modules/
COPY scripts/ ./scripts/

# Run pylint on both modules and scripts
RUN uv run pylint modules scripts --rcfile=.pylintrc