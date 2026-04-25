import uuid
from typing import List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import settings
from .embeddings import embedder


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self):
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(
                    size=embedder.dim,
                    distance=qm.Distance.COSINE,
                ),
            )

    def upsert(self, chunks: List[str], source: str) -> List[str]:
        if not chunks:
            return []
        vectors = embedder.embed_passages(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]
        points = [
            qm.PointStruct(
                id=ids[i],
                vector=vectors[i].tolist(),
                payload={"text": chunks[i], "source": source},
            )
            for i in range(len(chunks))
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return ids

    def search(self, query: str, top_k: int) -> List[Tuple[str, str, float]]:
        qv = embedder.embed_query(query)
        res = self.client.query_points(
            collection_name=self.collection,
            query=qv.tolist(),
            limit=top_k,
            with_payload=True,
        ).points
        return [(str(p.id), p.payload["text"], float(p.score)) for p in res]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count

    def get_ids_by_source(self, source: str) -> list[str]:
        ids = []
        offset = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=qm.Filter(
                    must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=source))]
                ),
                limit=100,
                offset=offset,
                with_payload=False,
            )
            points, next_offset = result
            ids.extend(str(p.id) for p in points)
            if next_offset is None:
                break
            offset = next_offset
        return ids

    def delete_by_source(self, source: str):
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=source))]
                )
            ),
            wait=True,
        )

    def clear(self):
        self.client.delete_collection(self.collection)
        self._ensure_collection()


vector_store = VectorStore()
