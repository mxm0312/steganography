.PHONY: install test lint format ui bench

install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check stego tests benchmarks
	uv run ruff format --check stego tests benchmarks

format:
	uv run ruff check --fix stego tests benchmarks
	uv run ruff format stego tests benchmarks

ui:
	uv run streamlit run stego/ui/app.py

# Бенчмарк: make bench / make bench METHODS="lsb lfvsn" SCENARIOS="small medium" DEVICE=cuda REPEATS=5
bench:
	uv run python -m benchmarks.bench \
		$(if $(METHODS),--methods $(METHODS),) \
		$(if $(SCENARIOS),--scenarios $(SCENARIOS),) \
		$(if $(DEVICE),--device $(DEVICE),) \
		$(if $(REPEATS),--repeats $(REPEATS),) \
		$(if $(BITS),--bits-per-channel $(BITS),) \
		$(if $(ARCH),--architecture $(ARCH),)
