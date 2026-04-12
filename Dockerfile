# # This first line enables the BuildKit features like cache mounts.
ARG PYTHON_VERSION=3.11
#ARG BASE_IMAGE=nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG LOG_LEVEL=info

# ---- Base Stage ----
# Use the official NVIDIA CUDA development image as a base
FROM ${BASE_IMAGE} AS base
ARG PYTHON_VERSION
ARG LOG_LEVEL
# # Expects PYTHON_VERSION to be set in a compose file
RUN echo "The Python version is set to: python${PYTHON_VERSION}"

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set environment variables for default execution
ENV DEVICE=cuda
ENV COMPUTE_TYPE=float16

# Set the working directory
WORKDIR /app

# Prevent tzdata from asking for input during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies: Python, pip
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a standard symlink for the python executable for robustness
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Create a non-root user and group
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser


# volume used for model cache
VOLUME /home/appuser/.cache
# Create the directory and then change its ownership in one step
RUN mkdir -p /home/appuser/.cache && chown -R appuser:appuser /home/appuser/.cache

# ---- Builder Stage ----
# # The result is cached and reused by dev and prod, speeding up builds.
FROM base AS builder
ARG PYTHON_VERSION
ARG LOG_LEVEL

# Create a virtual environment at a predictable location
RUN python -m venv /opt/venv

# Activate the virtual environment for subsequent RUN commands
ENV PATH="/opt/venv/bin:$PATH"

# Copy the requirements file
COPY --chown=appuser:appuser requirements.txt .

# # Use a cache mount to speed up pip installs across builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ---- Production Stage ----
FROM base AS prod
ARG PYTHON_VERSION
ARG LOG_LEVEL

# Copy the entire venv folder to avoid Python patch version naming errors
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Ensure the venv is active in the final image
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application source code after installing dependencies to improve caching
COPY --chown=appuser:appuser src/ ./src

# Switch to the non-root user
USER appuser

ENV LOG_LEVEL="$LOG_LEVEL" \
    UVICORN_PORT="80" \
    UVICORN_HOST="0.0.0.0"

# Set the command to run the FastAPI application with Uvicorn
CMD ["python", "-m", "uvicorn", "src.app:app"]

# ---- Development Stage ----
# Sets up a container for interactive development with tools and mounted source code.
FROM base AS dev
ARG PYTHON_VERSION

# Install system dependencies for dev only
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire venv and update PATH for dev stage
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Switch to the non-root user
USER appuser

ENV LOG_LEVEL="$LOG_LEVEL" \
    UVICORN_PORT="80" \
    UVICORN_HOST="0.0.0.0"

# Run the container in reload mode for live coding
CMD ["python", "-m", "uvicorn", "src.app:app", "--reload"]