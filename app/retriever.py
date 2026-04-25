from dataclasses import dataclass
from typing import List

from .bm25_store import bm25_store
from .config import settings
from .vector_store import vector_store


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source: str
    score: float


RRF_K = 60


def hybrid_search(query: str) -> List[RetrievedChunk]:
    vec_res = vector_store.search(query, settings.top_k_vector)
    bm25_res = bm25_store.search(query, settings.top_k_bm25)

    fused: dict[str, list] = {}

    for rank, (cid, text, source, _score) in enumerate(vec_res):
        fused.setdefault(cid, [0.0, text, source])
        fused[cid][0] += 1.0 / (RRF_K + rank + 1)

    for rank, (cid, text, source, _score) in enumerate(bm25_res):
        fused.setdefault(cid, [0.0, text, source])
        fused[cid][0] += 1.0 / (RRF_K + rank + 1)

    items = [
        RetrievedChunk(id=cid, text=data[1], source=data[2], score=data[0])
        for cid, data in fused.items()
    ]
    items.sort(key=lambda x: x.score, reverse=True)
    return items[: settings.top_k_final]
