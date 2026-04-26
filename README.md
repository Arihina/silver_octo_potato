# RAG Service

FastAPI + Qdrant + Ollama (gemma2:2b). Гибридный поиск: vector + BM25 через RRF. Исправление опечаток. История диалогов. Веб-интерфейс.

## Быстрый старт (Docker)

```bash
docker compose up -d
```

Это поднимет три контейнера: Qdrant, Ollama и RAG API. Первый запуск дольше — собирается образ и скачивается embedding-модель (~120 МБ).

После запуска нужно скачать LLM-модель в контейнер Ollama:

```bash
docker exec rag_ollama ollama pull gemma2:2b
```

Готово. Открывай http://localhost:8000

### GPU (опционально)

Для GPU-ускорения Ollama раскомментируй блок `deploy` в `docker-compose.yml` у сервиса `ollama`. Нужен NVIDIA Container Toolkit:

```bash
# Установка nvidia-container-toolkit (Ubuntu/Debian)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Данные

Все данные хранятся в Docker volumes: `qdrant_data`, `ollama_data`, `rag_data`. Они переживают пересоздание контейнеров. Чтобы начать с чистого листа:

```bash
docker compose down -v
```

## Запуск без Docker (вручную)

### 1. Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma2:2b
ollama serve
```

### 2. Qdrant

Можно поднять только Qdrant в Docker:

```bash
docker run -d --name qdrant -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 3. Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Интерфейс

http://localhost:8000 — веб-интерфейс с чатами и базой знаний.
http://localhost:8000/docs — OpenAPI / Swagger UI.

## API

### Документы и файлы

| Метод  | Путь             | Назначение                           |
|--------|------------------|--------------------------------------|
| POST   | /upload          | загрузка .txt (multipart, поле file) |
| GET    | /files           | список загруженных файлов            |
| DELETE | /files/{file_id} | удалить файл и его чанки             |
| DELETE | /clear           | очистить все документы и BM25        |
| GET    | /health          | статус, счётчики                     |

### Чаты

| Метод  | Путь                      | Назначение              |
|--------|---------------------------|-------------------------|
| POST   | /chats                    | создать чат             |
| GET    | /chats                    | список чатов            |
| GET    | /chats/{chat_id}          | информация о чате       |
| PATCH  | /chats/{chat_id}          | переименовать           |
| DELETE | /chats/{chat_id}          | удалить чат с историей  |
| GET    | /chats/{chat_id}/messages | история сообщений       |

### Вопрос-ответ

| Метод | Путь | Назначение                                |
|-------|------|-------------------------------------------|
| POST  | /ask | вопрос с привязкой к чату и учётом истории|
