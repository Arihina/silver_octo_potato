import httpx

from .config import settings


SYSTEM_PROMPT = (
    "Ты — ассистент, отвечающий на вопросы пользователя СТРОГО на основе "
    "предоставленного контекста. Если в контексте нет ответа, честно скажи, "
    "что информации недостаточно. Отвечай на русском языке, коротко и по делу."
)


def build_prompt(question: str, context_chunks: list[str]) -> str:
    ctx = "\n\n---\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Контекст:\n{ctx}\n\n"
        f"Вопрос: {question}\n\n"
        f"Ответ:"
    )


async def generate(question: str, context_chunks: list[str]) -> str:
    prompt = build_prompt(question, context_chunks)
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{settings.ollama_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
