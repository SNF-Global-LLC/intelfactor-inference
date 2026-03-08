# IntelFactor Inference Engine - Build & Deploy
# Usage: make <target>

.PHONY: help build build-edge build-hybrid push test clean deploy-edge deploy-hybrid

# Configuration
REGISTRY ?= ghcr.io/tonesgainz
IMAGE_NAME ?= intelfactor-inference
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
PLATFORM ?= linux/arm64,linux/amd64

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

help: ## Show this help
	@echo "IntelFactor Inference Engine"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# ── Build ────────────────────────────────────────────────────────────────────

build: build-edge build-sync ## Build all Docker images

build-edge: ## Build edge station image
	@echo "$(GREEN)Building edge station image...$(NC)"
	docker build \
		-f deploy/edge-only/Dockerfile \
		-t $(REGISTRY)/$(IMAGE_NAME):$(VERSION) \
		-t $(REGISTRY)/$(IMAGE_NAME):latest \
		.
	@echo "$(GREEN)Built: $(REGISTRY)/$(IMAGE_NAME):$(VERSION)$(NC)"

build-sync: ## Build cloud sync agent image
	@echo "$(GREEN)Building sync agent image...$(NC)"
	docker build \
		-f deploy/hybrid/Dockerfile.sync \
		-t $(REGISTRY)/$(IMAGE_NAME)-sync:$(VERSION) \
		-t $(REGISTRY)/$(IMAGE_NAME)-sync:latest \
		.
	@echo "$(GREEN)Built: $(REGISTRY)/$(IMAGE_NAME)-sync:$(VERSION)$(NC)"

build-jetson: ## Build for Jetson (ARM64 with L4T base)
	@echo "$(GREEN)Building for Jetson...$(NC)"
	docker build \
		-f deploy/edge-only/Dockerfile \
		--platform linux/arm64 \
		-t $(REGISTRY)/$(IMAGE_NAME):$(VERSION)-jetson \
		.

# ── Push ─────────────────────────────────────────────────────────────────────

push: ## Push images to registry
	@echo "$(GREEN)Pushing images to $(REGISTRY)...$(NC)"
	docker push $(REGISTRY)/$(IMAGE_NAME):$(VERSION)
	docker push $(REGISTRY)/$(IMAGE_NAME):latest
	docker push $(REGISTRY)/$(IMAGE_NAME)-sync:$(VERSION)
	docker push $(REGISTRY)/$(IMAGE_NAME)-sync:latest

login: ## Login to GitHub Container Registry
	@echo "$(YELLOW)Login to ghcr.io...$(NC)"
	echo $$GITHUB_TOKEN | docker login ghcr.io -u $$GITHUB_USER --password-stdin

# ── Test ─────────────────────────────────────────────────────────────────────

test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	python3 -m pytest tests/ -v

test-storage: ## Run storage tests only
	python3 -m pytest tests/test_storage.py -v

test-api: ## Run API tests only
	python3 -m pytest tests/test_api_v2.py -v

test-sensors: ## Run sensor + maintenance IQ tests only
	python3 -m pytest tests/test_sensor_service.py tests/test_maintenance_iq.py -v

lint: ## Run linter
	python3 -m ruff check packages/ tests/

# ── TensorRT Engine ──────────────────────────────────────────────────────────

build-trt: ## Build TRT engine from .pt or .onnx on this device (usage: make build-trt MODEL=yolov8n.pt PRECISION=fp16)
	@if [ -z "$(MODEL)" ]; then \
		echo "$(RED)ERROR: MODEL required$(NC)"; \
		echo "Usage: make build-trt MODEL=yolov8n.pt PRECISION=fp16"; \
		echo "       make build-trt MODEL=yolov8n.onnx PRECISION=fp16"; \
		exit 1; \
	fi
	@echo "$(GREEN)Building TRT engine: $(MODEL) [$(or $(PRECISION),fp16)]$(NC)"
	@echo "$(YELLOW)⚠  Engine is device-specific — build ON the target Jetson$(NC)"
	./scripts/build_trt_engine.sh "$(MODEL)" "$(or $(PRECISION),fp16)"

build-trt-int8: ## Build INT8 TRT engine with calibration (usage: make build-trt-int8 MODEL=yolov8n.pt CALIB_DIR=./calibration_images/)
	@if [ -z "$(MODEL)" ]; then \
		echo "$(RED)ERROR: MODEL required$(NC)"; \
		echo "Usage: make build-trt-int8 MODEL=yolov8n.pt CALIB_DIR=./calibration_images/"; \
		exit 1; \
	fi
	@if [ -z "$(CALIB_DIR)" ]; then \
		echo "$(RED)ERROR: CALIB_DIR required for INT8$(NC)"; \
		echo "Usage: make build-trt-int8 MODEL=yolov8n.pt CALIB_DIR=./calibration_images/"; \
		echo "       Minimum 100 calibration images required."; \
		exit 1; \
	fi
	@echo "$(GREEN)Building INT8 TRT engine: $(MODEL)$(NC)"
	@echo "$(YELLOW)⚠  Engine is device-specific — build ON the target Jetson$(NC)"
	./scripts/build_trt_engine.sh "$(MODEL)" int8 --calib "$(CALIB_DIR)"

verify-trt: ## Verify a TRT engine loads and runs on this device (usage: make verify-trt ENGINE=/opt/intelfactor/models/vision/yolov8n_fp16.engine)
	@if [ -z "$(ENGINE)" ]; then \
		echo "$(RED)ERROR: ENGINE path required$(NC)"; \
		echo "Usage: make verify-trt ENGINE=/opt/intelfactor/models/vision/yolov8n_fp16.engine"; \
		exit 1; \
	fi
	@echo "$(GREEN)Verifying engine: $(ENGINE)$(NC)"
	python3 scripts/verify_trt_engine.py "$(ENGINE)"

# ── Deploy ───────────────────────────────────────────────────────────────────

deploy-edge: ## Deploy edge-only mode (local)
	@echo "$(GREEN)Deploying edge-only mode...$(NC)"
	cd deploy/edge-only && docker compose up -d
	@echo "$(GREEN)Dashboard: http://localhost:8080$(NC)"

deploy-hybrid: ## Deploy hybrid mode (with cloud sync)
	@echo "$(GREEN)Deploying hybrid mode...$(NC)"
	cd deploy/hybrid && docker compose up -d
	@echo "$(GREEN)Dashboard: http://localhost:8080$(NC)"
	@echo "$(GREEN)Syncing to: $(CLOUD_API_URL)$(NC)"

deploy-hub: ## Deploy site hub (Postgres + Grafana)
	@echo "$(GREEN)Deploying site hub...$(NC)"
	cd deploy/hub && docker compose up -d
	@echo "$(GREEN)Grafana: http://localhost:3000$(NC)"

stop: ## Stop all containers
	-cd deploy/edge-only && docker compose down
	-cd deploy/hybrid && docker compose down
	-cd deploy/hub && docker compose down

logs: ## Show station logs
	docker logs -f intelfactor-station

# ── Development ──────────────────────────────────────────────────────────────

dev: ## Run in development mode (no Docker)
	@echo "$(GREEN)Starting development server...$(NC)"
	STORAGE_MODE=local \
	SQLITE_DB_PATH=./data/dev.db \
	EVIDENCE_DIR=./data/evidence \
	python3 -m packages.inference.cli serve --api-only

doctor: ## Run system health checks
	python3 scripts/doctor.py --full

clean: ## Clean build artifacts
	rm -rf .pytest_cache __pycache__ .ruff_cache
	rm -rf data/dev.db data/evidence
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ── Release ──────────────────────────────────────────────────────────────────

tag: ## Create a git tag (usage: make tag VERSION=v1.0.0)
	@if [ -z "$(VERSION)" ]; then echo "VERSION required"; exit 1; fi
	git tag -a $(VERSION) -m "Release $(VERSION)"
	git push origin $(VERSION)

release: build push tag ## Build, push, and tag a release
	@echo "$(GREEN)Released $(VERSION)$(NC)"
