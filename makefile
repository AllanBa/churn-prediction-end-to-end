.PHONY: run test lint format

# Run the FastAPI server locally
run:
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Run automated tests using Pytest
test:
	pytest tests/ -v

# Run formatting and linting using Ruff
lint:
	ruff check .
	ruff format .