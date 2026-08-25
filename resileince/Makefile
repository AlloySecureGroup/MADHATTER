.PHONY: test smoke-cpu up-cpu up-gpu

test:
	python -m pytest -q

smoke-cpu:
	FORCE_CPU=1 python scripts/smoke_model.py --model HuggingFaceTB/SmolLM2-360M-Instruct

up-cpu:
	docker compose -f docker-compose.cpu.yml up --build

up-gpu:
	docker compose up --build
