# Makefile for RiffSpace

.PHONY: help install install-dev test lint format clean ui demo docs

help:
	@echo "RiffSpace - Development Commands"
	@echo ""
	@echo "  make install        Install core dependencies"
	@echo "  make install-dev    Install with dev tools"
	@echo "  make install-all    Install everything (vectordb, ui, audio)"
	@echo "  make test           Run tests"
	@echo "  make lint           Run linter"
	@echo "  make format         Format code with black"
	@echo "  make ui             Run Streamlit UI demo"
	@echo "  make demo           Run CLI demos"
	@echo "  make clean          Clean temporary files"
	@echo "  make docs           Open documentation"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

install-all:
	pip install -r requirements.txt
	pip install -e ".[dev,vectordb,ui,audio]"

test:
	pytest tests/ -v

lint:
	flake8 src/ tests/ --max-line-length=100

format:
	black src/ tests/ examples/ *.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf build/ dist/

ui:
	streamlit run app.py

demo:
	@echo "Running demos..."
	@echo ""
	@echo "=== Riff Demo ==="
	python examples/demo.py
	@echo ""
	@echo "=== Vector DB Demo ==="
	python examples/vectordb_demo.py
	@echo ""
	@echo "=== Song Demo ==="
	python examples/song_demo.py

docs:
	@echo "Opening documentation..."
	@echo "README: README.md"
	@echo "Getting Started: GETTING_STARTED.md"
	@echo "Vector DB Guide: VECTOR_DB_GUIDE.md"
	@echo "Architecture: ARCHITECTURE.md"
