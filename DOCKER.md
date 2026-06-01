# Docker Deployment

## Prerequisites
- Docker + Docker Compose installed
- 4GB+ RAM (for Ollama + Qwen model)

## Steps

### 1. Build and start all services
```bash
docker compose up --build
```

### 2. Pull the LLM model (first run only)
```bash
docker compose exec ollama ollama pull qwen2.5:3b
```

### 3. Access
| Service | URL |
|---|---|
| Frontend | http://localhost:8501 |
| API | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |
| Ollama | http://localhost:11434 |

---

## Individual Containers

**Backend only:**
```bash
docker build -t ahoum .
docker run -p 8000:8000 ahoum
```

**Frontend only:**
```bash
docker run -p 8501:8501 ahoum streamlit run src/frontend/app.py --server.address 0.0.0.0
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `BACKEND_URL` | `http://localhost:8000` | Backend URL for frontend |

Set these in a `.env` file at the project root or pass via `docker run -e`.

---

## Data Persistence

The `./data` directory is mounted into the container — facet indexes and processed CSVs persist across restarts.

## Stopping
```bash
docker compose down          # stop containers
docker compose down -v       # stop + delete Ollama model volume
```
