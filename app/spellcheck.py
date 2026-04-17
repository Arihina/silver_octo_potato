import re
from collections import Counter
from threading import RLock
from typing import List

from spellchecker import SpellChecker

from .bm25_store import bm25_store


_WORD_RE = re.compile(r"\w+", re.UNICODE)


class CorpusSpellChecker:

    def __init__(self):
        self._lock = RLock()
        self._spell: SpellChecker | None = None
        self.rebuild()

    def _detect_lang(self, vocab: List[str]) -> str:
        if not vocab:
            return "ru"
        cyr = sum(1 for w in vocab if re.search(r"[а-яёА-ЯЁ]", w))
        return "ru" if cyr >= len(vocab) / 2 else "en"

    def rebuild(self):
        with self._lock:
            vocab = bm25_store.vocabulary()
            lang = self._detect_lang(vocab)
            try:
                spell = SpellChecker(language=lang)
            except ValueError:
                spell = SpellChecker(language="en")
            if vocab:
                freq = Counter(vocab)
                for word, count in freq.items():
                    for _ in range(count):
                        spell.word_frequency.add(word)
            self._spell = spell

    def correct_query(self, query: str) -> str:
        with self._lock:
            if not self._spell:
                return query

            def repl(m: re.Match) -> str:
                w = m.group(0)
                if w.isdigit() or len(w) <= 2:
                    return w
                lower = w.lower()
                if lower in self._spell:
                    return w
                candidate = self._spell.correction(lower)
                if not candidate or candidate == lower:
                    return w
                if w[0].isupper():
                    candidate = candidate.capitalize()
                return candidate

            return _WORD_RE.sub(repl, query)


spell = CorpusSpellChecker()
