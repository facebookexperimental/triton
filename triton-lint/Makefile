.PHONY: install test lint format clean example

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=triton_lint --cov-report=html --cov-report=term

lint:
	ruff check triton_lint/

format:
	ruff format triton_lint/

type-check:
	mypy triton_lint/

example:
	triton-lint examples/bad_kernel.py

example-json:
	triton-lint --format=json examples/bad_kernel.py

list-rules:
	triton-lint --list-rules

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +

help:
	@echo "Triton-Lint Development Commands:"
	@echo "  make install      - Install in development mode"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make lint         - Check code style"
	@echo "  make format       - Format code"
	@echo "  make type-check   - Run type checker"
	@echo "  make example      - Run on example file"
	@echo "  make list-rules   - List all rules"
	@echo "  make clean        - Clean build artifacts"
