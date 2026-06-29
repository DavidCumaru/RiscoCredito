.PHONY: data train api app test lint clean docker-up docker-down

data:
	python -m src.data.generator

train:
	python -m src.models.trainer

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

app:
	streamlit run src/app/streamlit_app.py --server.port 8501

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

clean:
	rm -rf artifacts/ mlruns/ chroma_db/ *.db __pycache__ .pytest_cache

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down
