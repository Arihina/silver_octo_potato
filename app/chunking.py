import re
from typing import List

from .config import settings


_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯA-ZЁ0-9])")


def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(paragraph: str) -> List[str]:
    sents = _SENT_SPLIT.split(paragraph)
    return [s.strip() for s in sents if s.strip()]


def _pack(units: List[str], max_chars: int) -> List[str]:
    out: List[str] = []
    buf = ""
    for u in units:
        if not buf:
            buf = u
            continue
        if len(buf) + 1 + len(u) <= max_chars:
            buf = f"{buf} {u}"
        else:
            out.append(buf)
            buf = u
    if buf:
        out.append(buf)
    return out


def chunk_text(text: str) -> List[str]:
    max_c = settings.chunk_max_chars
    min_c = settings.chunk_min_chars

    paragraphs = _split_paragraphs(text)
    chunks: List[str] = []

    for p in paragraphs:
        if len(p) <= max_c:
            chunks.append(p)
        else:
            sents = _split_sentences(p)
            if not sents:
                for i in range(0, len(p), max_c):
                    chunks.append(p[i : i + max_c])
            else:
                chunks.extend(_pack(sents, max_c))

    merged: List[str] = []
    for c in chunks:
        if merged and len(merged[-1]) < min_c and len(merged[-1]) + 1 + len(c) <= max_c:
            merged[-1] = f"{merged[-1]}\n{c}"
        else:
            merged.append(c)

    return merged
