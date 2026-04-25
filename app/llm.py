import ollama

from .config import settings


SYSTEM_PROMPT = (
    "Ты — ассистент, отвечающий на вопросы пользователя СТРОГО на основе "
    "предоставленного контекста. Если в контексте нет ответа, честно скажи, "
    "что информации недостаточно. Отвечай на русском языке, коротко и по делу. "
    "Учитывай историю диалога при ответе."
)


def _build_context_block(chunks: list[str]) -> str:
    return "\n\n---\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))


def _build_messages(
    question: str,
    context_chunks: list[str],
    history: list[dict],
) -> list[dict]:
    ctx = _build_context_block(context_chunks)
    system_msg = f"{SYSTEM_PROMPT}\n\nКонтекст из базы знаний:\n{ctx}"

    messages = [{"role": "system", "content": system_msg}]

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    return messages


async def generate(
    question: str,
    context_chunks: list[str],
    history: list[dict] | None = None,
) -> str:
    messages = _build_messages(question, context_chunks, history or [])

    response = ollama.chat(
        model=settings.ollama_model,
        messages=messages,
        options={"temperature": 0.2},
    )
    return response["message"]["content"].strip()
