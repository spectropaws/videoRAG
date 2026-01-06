.PHONY: help build run-gpu run-cpu process query clean

help: ## Show this help message
	@echo "Video RAG Docker Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $1, $2}'

setup: ## Set up Docker environment
	@chmod +x scripts/docker_setup.sh
	@./scripts/docker_setup.sh

setup-cpu: ## Set up Docker environment (CPU only)
	@chmod +x scripts/docker_setup.sh
	@./scripts/docker_setup.sh --cpu

build: ## Build Docker image
	@docker-compose build

run-gpu: ## Run interactive session with GPU
	@docker-compose run --rm video-rag bash

run-cpu: ## Run interactive session with CPU only
	@docker-compose -f docker-compose.cpu.yml run --rm video-rag bash

process: ## Process a video file (usage: make process VIDEO=path/to/video.mp4)
	@docker-compose run --rm video-rag python3 scripts/process_video.py /app/videos/$(VIDEO)

query: ## Start interactive query session
	@docker-compose run --rm video-rag python3 scripts/query_system.py --interactive

query-cpu: ## Start interactive query session (CPU only)
	@docker-compose -f docker-compose.cpu.yml run --rm video-rag python3 scripts/query_system.py --interactive

shell: ## Open shell in container
	@docker-compose run --rm video-rag bash

clean: ## Clean up Docker resources
	@docker-compose down
	@docker system prune -f
