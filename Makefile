.PHONY: help mock test lint clean install install-data install-model install-ui

# ── Default ────────────────────────────────────────────
help: ## Hiển thị các lệnh available
	@echo "Fake News Detection — Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Install ────────────────────────────────────────────
install: ## Cài tất cả dependencies
	uv sync

# ── Development ────────────────────────────────────────
mock: ## Chạy mock server trên port 8000
	uv run uvicorn api.mock:app --port 8000 --reload

ui: ## Chạy Streamlit app
	uv run streamlit run ui/app.py

# ── Testing ────────────────────────────────────────────
test: ## Chạy tất cả tests
	uv run pytest tests/ -v --tb=short

test-cov: ## Chạy tests với coverage report
	uv run pytest tests/ -v --cov=. --cov-report=html --cov-report=term

test-schemas: ## Test API contract
	uv run pytest tests/test_schemas.py -v

test-mock: ## Test mock server
	uv run pytest tests/test_mock_server.py -v

# ── Linting ────────────────────────────────────────────
lint: ## Chạy linter (ruff)
	uv run ruff check .

format: ## Auto-format code
	uv run ruff format .

# ── Clean ──────────────────────────────────────────────
clean: ## Xóa cache và build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
