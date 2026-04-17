from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .bm25_store import bm25_store
from .chunking import chunk_text
from .llm import generate
from .retriever import hybrid_search
from .spellcheck import spell
from .vector_store import vector_store

app = FastAPI(title="RAG Service", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    original_query: str
    corrected_query: str
    used_chunks: list[dict]


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "qdrant_points": vector_store.count(),
        "bm25_docs": len(bm25_store.ids),
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
            raise HTTPException(status_code=400, detail="Не удалось декодировать файл (ожидается UTF-8 или CP1251)")

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
        raise HTTPException(status_code=400, detail="Корпус пуст — сначала загрузите документ через /upload")

    corrected = spell.correct_query(req.question)
    retrieved = hybrid_search(corrected)

    if not retrieved:
        return AskResponse(
            answer="В загруженных документах ничего подходящего не найдено.",
            original_query=req.question,
            corrected_query=corrected,
            used_chunks=[],
        )

    context_texts = [r.text for r in retrieved]
    answer = await generate(corrected, context_texts)

    return AskResponse(
        answer=answer,
        original_query=req.question,
        corrected_query=corrected,
        used_chunks=[
            {"id": r.id, "score": round(r.score, 4), "text": r.text}
            for r in retrieved
        ],
    )


@app.delete("/clear")
async def clear():
    vector_store.clear()
    bm25_store.clear()
    spell.rebuild()
    return {"status": "cleared"}
