.PHONY: install setup test lint format clean demo

install:
	pip install -r requirements.txt

setup: install
	python scripts/setup_models.py

test:
	pytest tests/ -v

lint:
	flake8 app/ webui/ cli/ tests/
	mypy app/ webui/ cli/

format:
	black app/ webui/ cli/ tests/
	isort app/ webui/ cli/ tests/

clean:
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

demo:
	python webui/app.py --demo

benchmark:
	python scripts/benchmark.py

cli-demo:
	python -m cli.echo_cli assets/sample_audio/demo.wav --report

dev-install:
	pip install -e ".[dev]"

docker-build:
	docker build -t echo:latest .

docker-run:
	docker run -p 7860:7860 echo:latest