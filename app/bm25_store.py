import pickle
import re
from pathlib import Path
from threading import RLock
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from .config import settings


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Store:
    def __init__(self):
        self._lock = RLock()
        self._path: Path = settings.data_path / "bm25.pkl"
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.sources: List[str] = []
        self.tokens: List[List[str]] = []
        self.bm25: BM25Okapi | None = None
        self._load()

    def _load(self):
        if self._path.exists():
            with open(self._path, "rb") as f:
                data = pickle.load(f)
            self.ids = data["ids"]
            self.texts = data["texts"]
            self.sources = data.get("sources", [""] * len(self.ids))
            self.tokens = data["tokens"]
            self._rebuild()

    def _persist(self):
        with open(self._path, "wb") as f:
            pickle.dump(
                {"ids": self.ids, "texts": self.texts, "sources": self.sources, "tokens": self.tokens},
                f,
            )

    def _rebuild(self):
        if self.tokens:
            self.bm25 = BM25Okapi(self.tokens)
        else:
            self.bm25 = None

    def add(self, ids: List[str], texts: List[str], source: str = ""):
        with self._lock:
            for i, t in zip(ids, texts):
                self.ids.append(i)
                self.texts.append(t)
                self.sources.append(source)
                self.tokens.append(tokenize(t))
            self._rebuild()
            self._persist()

    def delete_by_ids(self, ids_to_remove: set[str]):
        with self._lock:
            keep = [(i, t, s, tok) for i, t, s, tok in zip(self.ids, self.texts, self.sources, self.tokens) if i not in ids_to_remove]
            if keep:
                self.ids, self.texts, self.sources, self.tokens = map(list, zip(*keep))
            else:
                self.ids, self.texts, self.sources, self.tokens = [], [], [], []
            self._rebuild()
            self._persist()

    def search(self, query: str, top_k: int) -> List[Tuple[str, str, str, float]]:
        with self._lock:
            if not self.bm25 or not self.ids:
                return []
            q_tokens = tokenize(query)
            if not q_tokens:
                return []
            scores = self.bm25.get_scores(q_tokens)
            idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            return [(self.ids[i], self.texts[i], self.sources[i], float(scores[i])) for i in idxs if scores[i] > 0]

    def vocabulary(self) -> List[str]:
        with self._lock:
            vocab: List[str] = []
            for toks in self.tokens:
                vocab.extend(toks)
            return vocab

    def clear(self):
        with self._lock:
            self.ids.clear()
            self.texts.clear()
            self.sources.clear()
            self.tokens.clear()
            self.bm25 = None
            if self._path.exists():
                self._path.unlink()


bm25_store = BM25Store()
