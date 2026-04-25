from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .bm25_store import bm25_store
from .chat_store import chat_store
from .chunking import chunk_text
from .config import settings
from .llm import generate
from .retriever import hybrid_search
from .spellcheck import spell
from .vector_store import vector_store

app = FastAPI(title="RAG Service", version="0.2.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chat_id: str | None = Field(
        default=None,
        description="ID чата. Если не передан — создаётся новый чат автоматически.",
    )


class AskResponse(BaseModel):
    answer: str
    chat_id: str
    original_query: str
    corrected_query: str
    used_chunks: list[dict]


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int


class ChatCreate(BaseModel):
    title: str = "Новый чат"


class ChatRename(BaseModel):
    title: str = Field(..., min_length=1)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "qdrant_points": vector_store.count(),
        "bm25_docs": len(bm25_store.ids),
        "chats": len(chat_store.list_chats()),
    }


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Поддерживаются только .txt файлы")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp1251")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Не удалось декодировать файл (ожидается UTF-8 или CP1251)",
            )

    if not text.strip():
        raise HTTPException(status_code=400, detail="Файл пуст")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Не удалось извлечь чанки из файла")

    ids = vector_store.upsert(chunks, source=file.filename)
    bm25_store.add(ids, chunks)
    spell.rebuild()

    return UploadResponse(
        filename=file.filename,
        chunks_added=len(chunks),
        total_chunks=len(bm25_store.ids),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Корпус пуст — сначала загрузите документ через /upload",
        )

    if req.chat_id:
        chat = chat_store.get_chat(req.chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail=f"Чат {req.chat_id} не найден")
    else:
        title = req.question[:50] + ("..." if len(req.question) > 50 else "")
        chat = chat_store.create_chat(title=title)

    chat_id = chat["id"]

    history = chat_store.get_history(chat_id, last_n=settings.history_length)
    corrected = spell.correct_query(req.question)

    retrieved = hybrid_search(corrected)

    if not retrieved:
        answer = "В загруженных документах ничего подходящего не найдено."
        chat_store.add_message(chat_id, "user", req.question)
        chat_store.add_message(chat_id, "assistant", answer)
        return AskResponse(
            answer=answer,
            chat_id=chat_id,
            original_query=req.question,
            corrected_query=corrected,
            used_chunks=[],
        )

    context_texts = [r.text for r in retrieved]

    answer = await generate(corrected, context_texts, history=history)

    chat_store.add_message(chat_id, "user", req.question)
    chat_store.add_message(chat_id, "assistant", answer)

    return AskResponse(
        answer=answer,
        chat_id=chat_id,
        original_query=req.question,
        corrected_query=corrected,
        used_chunks=[
            {"id": r.id, "score": round(r.score, 4), "text": r.text}
            for r in retrieved
        ],
    )


@app.post("/chats")
async def create_chat(body: ChatCreate):
    return chat_store.create_chat(title=body.title)


@app.get("/chats")
async def list_chats():
    return chat_store.list_chats()


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    chat = chat_store.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Чат не найден")
    return chat


@app.get("/chats/{chat_id}/messages")
async def get_messages(chat_id: str):
    chat = chat_store.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Чат не найден")
    return chat_store.get_all_messages(chat_id)


@app.patch("/chats/{chat_id}")
async def rename_chat(chat_id: str, body: ChatRename):
    result = chat_store.rename_chat(chat_id, body.title)
    if not result:
        raise HTTPException(status_code=404, detail="Чат не найден")
    return result


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    if not chat_store.delete_chat(chat_id):
        raise HTTPException(status_code=404, detail="Чат не найден")
    return {"status": "deleted", "chat_id": chat_id}


@app.delete("/clear")
async def clear():
    vector_store.clear()
    bm25_store.clear()
    spell.rebuild()
    return {"status": "cleared (документы). Чаты сохранены — удаляйте через DELETE /chats/{id}"}
