import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Optional

from .config import settings


class ChatStore:
    def __init__(self):
        self._lock = RLock()
        db_path = settings.data_path / "chats.db"
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()
        self._migrate()

    def _init_tables(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, created_at);
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    chunks_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            self._conn.commit()

    def _migrate(self):
        with self._lock:
            cols = [row[1] for row in self._conn.execute("PRAGMA table_info(messages)").fetchall()]
            if "sources" not in cols:
                self._conn.execute("ALTER TABLE messages ADD COLUMN sources TEXT NOT NULL DEFAULT '[]'")
                self._conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_chat(self, title: str = "Новый чат") -> dict:
        with self._lock:
            chat_id = str(uuid.uuid4())
            now = self._now()
            self._conn.execute(
                "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (chat_id, title, now, now),
            )
            self._conn.commit()
            return {"id": chat_id, "title": title, "created_at": now, "updated_at": now}

    def list_chats(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_chat(self, chat_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, title, created_at, updated_at FROM chats WHERE id = ?",
                (chat_id,),
            ).fetchone()
            return dict(row) if row else None

    def rename_chat(self, chat_id: str, title: str) -> Optional[dict]:
        with self._lock:
            now = self._now()
            cur = self._conn.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, chat_id),
            )
            self._conn.commit()
            if cur.rowcount == 0:
                return None
            return self.get_chat(chat_id)

    def delete_chat(self, chat_id: str) -> bool:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            cur = self._conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def add_message(self, chat_id: str, role: str, content: str, sources: list[str] | None = None) -> dict:
        with self._lock:
            msg_id = str(uuid.uuid4())
            now = self._now()
            sources_json = json.dumps(sources or [], ensure_ascii=False)
            self._conn.execute(
                "INSERT INTO messages (id, chat_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (msg_id, chat_id, role, content, sources_json, now),
            )
            self._conn.execute(
                "UPDATE chats SET updated_at = ? WHERE id = ?",
                (now, chat_id),
            )
            self._conn.commit()
            return {"id": msg_id, "chat_id": chat_id, "role": role, "content": content, "sources": sources or [], "created_at": now}

    def get_history(self, chat_id: str, last_n: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT role, content FROM messages
                WHERE chat_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (chat_id, last_n),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    def get_all_messages(self, chat_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, role, content, sources, created_at FROM messages WHERE chat_id = ? ORDER BY created_at",
                (chat_id,),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["sources"] = json.loads(d.get("sources") or "[]")
                result.append(d)
            return result

    def add_file(self, filename: str, chunks_count: int) -> dict:
        with self._lock:
            file_id = str(uuid.uuid4())
            now = self._now()
            self._conn.execute(
                "INSERT INTO files (id, filename, chunks_count, created_at) VALUES (?, ?, ?, ?)",
                (file_id, filename, chunks_count, now),
            )
            self._conn.commit()
            return {"id": file_id, "filename": filename, "chunks_count": chunks_count, "created_at": now}

    def list_files(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, filename, chunks_count, created_at FROM files ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_file(self, file_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, filename, chunks_count, created_at FROM files WHERE id = ?",
                (file_id,),
            ).fetchone()
            return dict(row) if row else None

    def delete_file(self, file_id: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT filename FROM files WHERE id = ?", (file_id,)).fetchone()
            if not row:
                return None
            filename = row["filename"]
            self._conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
            self._conn.commit()
            return filename


chat_store = ChatStore()
