.PHONY: install test test-cov lint format benchmark clean

install:
	pip install -e ".[dev]" --break-system-packages

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=mlengine --cov-report=term-missing --cov-report=html

lint:
	flake8 mlengine/ tests/ --max-line-length=100 --ignore=E203,W503
	python -m isort --check-only mlengine/ tests/

format:
	python -m isort mlengine/ tests/
	python -m black mlengine/ tests/ --line-length=100

benchmark:
	python benchmarks/benchmark_vs_sklearn.py

clean:
	find . -type d -name __pycache__ | xargs rm -rf
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
