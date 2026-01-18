.PHONY: help install install-dev test test-quick test-slow lint format clean run-experiments

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install project dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev,viz]"

test:  ## Run all tests
	pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

test-quick:  ## Run quick tests only (exclude slow tests)
	pytest tests/ -v -m "not slow" --cov=. --cov-report=term-missing

test-slow:  ## Run slow tests only
	pytest tests/ -v -m "slow"

test-smoke:  ## Run smoke tests for CI
	pytest tests/ -v -m "smoke"

lint:  ## Run linters (flake8, mypy)
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics
	mypy . --ignore-missing-imports

format:  ## Format code with black and isort
	black .
	isort .

format-check:  ## Check code formatting without making changes
	black --check --diff .
	isort --check-only --diff .

clean:  ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

run-demo:  ## Run the demo implementation
	python fl_personal_finance_implementation.py

run-fedavg:  ## Run FedAvg baseline experiment
	@echo "Running FedAvg baseline..."
	# python experiments/run_fedavg.py

run-fedprox:  ## Run FedProx experiment
	@echo "Running FedProx experiment..."
	# python experiments/run_fedprox.py

run-scaffold:  ## Run SCAFFOLD experiment
	@echo "Running SCAFFOLD experiment..."
	# python experiments/run_scaffold.py

run-hybrid:  ## Run hybrid FedProx-SCAFFOLD experiment
	@echo "Running hybrid experiment..."
	# python experiments/run_hybrid.py

run-all-experiments:  ## Run all experiments sequentially
	@echo "Running all FL experiments..."
	make run-fedavg
	make run-fedprox
	make run-scaffold
	make run-hybrid

docker-build:  ## Build Docker image
	docker build -t fl-personal-finance:latest .

docker-run:  ## Run Docker container
	docker run --rm -it fl-personal-finance:latest

ci-local:  ## Run CI checks locally
	@echo "Running local CI checks..."
	make format-check
	make lint
	make test-quick
	make run-demo

setup-hooks:  ## Set up pre-commit hooks
	@echo "Setting up git pre-commit hooks..."
	@echo '#!/bin/bash' > .git/hooks/pre-commit
	@echo 'make format-check' >> .git/hooks/pre-commit
	@echo 'make lint' >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Pre-commit hooks installed!"

benchmark:  ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	# python benchmarks/run_benchmarks.py
