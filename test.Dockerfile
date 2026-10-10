# Use the stable full Debian base image
FROM ghcr.io/astral-sh/uv:debian

# Set the working directory
WORKDIR /app

# Set the PYTHONPATH so Python can find your modules
ENV PYTHONPATH=/app

## Force scientific libraries to run single-threaded. This prevents race
## conditions and reduces memory pressure, which are common causes of segfaults.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

COPY pyproject.toml .
RUN uv sync --all-extras --dev

COPY precursor2cof-capacitance .

RUN uv run pylint modules --rcfile=.pylintrc
