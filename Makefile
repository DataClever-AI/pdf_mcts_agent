IMAGE      ?= mvp-v2
TAG        ?= dev
IMAGE_REF  := $(IMAGE):$(TAG)
PORT       ?= 8501
CONTAINER  := mvp-v2-local

# ECR — set AWS_REGION / AWS_ACCOUNT_ID before using ecr-*/deploy targets, e.g.:
#   make deploy AWS_REGION=us-east-1 AWS_ACCOUNT_ID=123456789012
# `deploy` = full local pipeline, then push :latest (what QA pulls) + a git-sha tag (rollback/audit).
# Repo tag mutability is MUTABLE, so re-pushing :latest each time overwrites it as expected.
AWS_REGION     ?=
AWS_ACCOUNT_ID ?=
ECR_HOST       := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
ECR_REPO       := $(ECR_HOST)/$(IMAGE)

GIT_SHA := $(shell git rev-parse --short HEAD)

.PHONY: all test verify-cython docker-build smoketest run stop clean help ecr-login ecr-push deploy

all: test verify-cython docker-build smoketest ## Run full local pipeline: test, cython check, build image, smoketest

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /: /'

test: ## Run pytest against plain .py sources
	uv run pytest tests/ -q

verify-cython: ## Compile src/ with Cython in a scratch copy, run tests against .so, then discard — catches Cython breakage before the slow docker build
	rm -rf /tmp/mvp_v2_cython_check
	cp -R src /tmp/mvp_v2_cython_check
	uv run --group build bash scripts/build_cython.sh
	uv run pytest tests/ -q
	rm -rf src
	mv /tmp/mvp_v2_cython_check src

docker-build: ## Build local image (host platform)
	docker build -t $(IMAGE_REF) .

smoketest: ## Boot the built image, hit the health endpoint, then stop it
	-docker rm -f $(CONTAINER) >/dev/null 2>&1
	docker run -d --rm --name $(CONTAINER) -p $(PORT):8501 \
		-e LLM_API_KEY=smoketest -e LLM_BASE_URL=http://localhost:9999/v1 -e LLM_MODEL=smoketest \
		$(IMAGE_REF)
	@sleep 6
	@code=$$(curl -sS -o /dev/null -w "%{http_code}" --max-time 5 http://localhost:$(PORT)/_stcore/health); \
	docker logs $(CONTAINER) 2>&1 | tail -20; \
	docker stop $(CONTAINER) >/dev/null; \
	if [ "$$code" != "200" ]; then echo "SMOKETEST FAILED: HTTP $$code"; exit 1; fi; \
	echo "SMOKETEST OK: HTTP $$code"

run: ## Run the image locally in the foreground — visit http://localhost:8501 (needs a real .env; see .env.example)
	@mkdir -p data logs
	-docker rm -f $(CONTAINER) >/dev/null 2>&1
	docker run --rm -it --name $(CONTAINER) -p $(PORT):8501 --env-file .env \
		-v "$$(pwd)/data:/app/data" -v "$$(pwd)/logs:/app/logs" \
		$(IMAGE_REF)

stop: ## Stop and remove the locally running container
	-docker rm -f $(CONTAINER) >/dev/null 2>&1

clean: ## Remove local image and any leftover scratch copies
	-docker rmi $(IMAGE_REF) >/dev/null 2>&1
	rm -rf /tmp/mvp_v2_cython_check build

ecr-login: ## Authenticate docker to the private ECR repo (token valid ~12h). One-time repo/IAM setup: scripts/aws_ecr_setup.sh
	@test -n "$(AWS_REGION)" || (echo "set AWS_REGION=..."; exit 1)
	@test -n "$(AWS_ACCOUNT_ID)" || (echo "set AWS_ACCOUNT_ID=..."; exit 1)
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_HOST)

ecr-push: ecr-login ## Build multi-arch (amd64+arm64) and push a specific TAG to private ECR
	docker buildx create --use --name mvp-v2-builder 2>/dev/null || docker buildx use mvp-v2-builder
	docker buildx build --platform linux/amd64,linux/arm64 -t $(ECR_REPO):$(TAG) --push .

deploy: all ecr-login ## Local checks (test+cython+smoketest), then build+push :latest (QA always pulls this) + a git-sha tag (rollback/audit)
	docker buildx create --use --name mvp-v2-builder 2>/dev/null || docker buildx use mvp-v2-builder
	docker buildx build --platform linux/amd64,linux/arm64 \
		-t $(ECR_REPO):latest -t $(ECR_REPO):$(GIT_SHA) --push .
	@echo "Pushed $(ECR_REPO):latest and $(ECR_REPO):$(GIT_SHA) — QA runs: docker pull $(ECR_REPO):latest"
