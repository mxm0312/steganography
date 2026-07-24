.PHONY: install test lint format ui

install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

ui:
	uv run streamlit run stego/ui/app.py
