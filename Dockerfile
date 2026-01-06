# Dockerfile
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    wget \
    curl \
    build-essential \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the project
RUN pip3 install -e .

# Create data directories
RUN mkdir -p data/{audio,transcripts,models,vector_db}

# Set permissions
RUN chmod +x scripts/*.py

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["python3", "scripts/query_system.py", "--interactive"]
