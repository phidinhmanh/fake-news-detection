.PHONY: help mock test lint clean install install-data install-model install-ui

# ── Default ────────────────────────────────────────────
help: ## Hiển thị các lệnh available
	@echo "Fake News Detection — Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Install ────────────────────────────────────────────
install: ## Cài tất cả dependencies
	pip install -r requirements.txt

install-data: ## Cài dependencies cho Người A (Data)
	pip install -r requirements-data.txt

install-model: ## Cài dependencies cho Người B (Model)
	pip install -r requirements-model.txt

install-ui: ## Cài dependencies cho Người C (UI)
	pip install -r requirements-ui.txt

# ── Development ────────────────────────────────────────
mock: ## Chạy mock server trên port 8000
	uvicorn mock_server:app --port 8000 --reload

ui: ## Chạy Streamlit app
	streamlit run ui/app.py

# ── Testing ────────────────────────────────────────────
test: ## Chạy tất cả tests
	pytest tests/ -v --tb=short

test-cov: ## Chạy tests với coverage report
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

test-schemas: ## Test API contract
	pytest tests/test_schemas.py -v

test-mock: ## Test mock server
	pytest tests/test_mock_server.py -v

# ── Linting ────────────────────────────────────────────
lint: ## Chạy linter (ruff)
	ruff check .

format: ## Auto-format code
	ruff format .

# ── Clean ──────────────────────────────────────────────
clean: ## Xóa cache và build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
