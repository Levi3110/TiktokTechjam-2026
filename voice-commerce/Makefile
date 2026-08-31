.PHONY: backend frontend test docker

backend:
	cd backend && uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	cd backend && pytest -q

docker:
	docker compose up --build

