#!/bin/bash

# Docker setup script for Video RAG project

echo "🐳 Setting up Docker environment for Video RAG..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{audio,transcripts,models,vector_db}
mkdir -p videos
mkdir -p logs

# Check for NVIDIA GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 NVIDIA GPU detected. Using GPU-enabled setup..."
    COMPOSE_FILE="docker-compose.yml"
    
    # Install nvidia-container-runtime if not present
    if ! docker info | grep -q nvidia; then
        echo "⚠️  NVIDIA container runtime not detected."
        echo "Please install nvidia-container-toolkit:"
        echo "  sudo pacman -S nvidia-container-toolkit"
        echo "  sudo systemctl restart docker"
        echo ""
        echo "Or use CPU-only mode: ./docker_setup.sh --cpu"
        exit 1
    fi
else
    echo "💻 No NVIDIA GPU detected. Using CPU-only setup..."
    COMPOSE_FILE="docker-compose.cpu.yml"
fi

# Check for --cpu flag
if [[ "$1" == "--cpu" ]]; then
    echo "💻 Forced CPU-only mode..."
    COMPOSE_FILE="docker-compose.cpu.yml"
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker-compose -f $COMPOSE_FILE build

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker setup complete!"
echo ""
echo "📋 Usage:"
echo "  # Process a video:"
echo "  docker-compose -f $COMPOSE_FILE run --rm video-rag python3 scripts/process_video.py /app/videos/your_video.mp4"
echo ""
echo "  # Interactive query:"
echo "  docker-compose -f $COMPOSE_FILE run --rm video-rag python3 scripts/query_system.py --interactive"
echo ""
echo "  # Single question:"
echo "  docker-compose -f $COMPOSE_FILE run --rm video-rag python3 scripts/query_system.py -q \"What is discussed?\""
echo ""
echo "📁 Place your videos in the 'videos/' directory"
echo "📁 Download your GGUF model to 'data/models/'"
echo "⚙️  Edit 'config/config.yaml' to configure the system"
