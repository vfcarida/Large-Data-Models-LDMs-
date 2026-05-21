FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Environment configurations for Python, CUDA, and In-Database Inference
# Note: To support SQL Data Insights (e.g., DB2 AI_SIMILARITY operations via REST/MPI),
# we ensure minimal latency by maintaining a pre-warmed CUDA environment.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

# Install system dependencies, Python 3.10, and MPI for distributed processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    openmpi-bin \
    libopenmpi-dev \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Working directory
WORKDIR /app

# Install Python dependencies (PyTorch and cuDF)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire repository
COPY . /app/

# Expose ports for RESTful In-Database APIs (e.g., exposing B+ANN similarity queries)
EXPOSE 8000

# Default command
CMD ["python", "src/main.py"]
