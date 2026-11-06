import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from synapse_memory.models import MemoryItem

DEFAULT_DB_PATH = Path.home() / ".synapse_memory" / "memories.db"


class StorageManager:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
                """
            )
            conn.commit()

    def save_memory(self, item: MemoryItem, vector: np.ndarray) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories 
                (id, content, category, tags, importance, created_at, last_accessed_at, access_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.content,
                    item.category,
                    json.dumps(item.tags),
                    item.importance,
                    item.created_at.isoformat(),
                    item.last_accessed_at.isoformat() if item.last_accessed_at else None,
                    item.access_count,
                    json.dumps(item.metadata),
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding)
                VALUES (?, ?)
                """,
                (item.id, vector.astype(np.float32).tobytes()),
            )
            conn.commit()

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_memory(row)

    def delete_memory(self, memory_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,))
            conn.commit()
            return cur.rowcount > 0

    def list_memories(self, limit: int = 50, category: Optional[str] = None) -> List[MemoryItem]:
        with self._get_connection() as conn:
            if category:
                cur = conn.execute(
                    "SELECT * FROM memories WHERE category = ? ORDER BY created_at DESC LIMIT ?",
                    (category, limit),
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            return [self._row_to_memory(row) for row in cur.fetchall()]

    def get_all_vectors(self, category: Optional[str] = None) -> List[Tuple[MemoryItem, np.ndarray]]:
        """Fetch all memories and their embeddings for ranking calculations."""
        with self._get_connection() as conn:
            query = """
                SELECT m.*, e.embedding 
                FROM memories m
                JOIN memory_embeddings e ON m.id = e.memory_id
            """
            params = []
            if category:
                query += " WHERE m.category = ?"
                params.append(category)

            cur = conn.execute(query, params)
            results = []
            for row in cur.fetchall():
                item = self._row_to_memory(row)
                raw_bytes = row["embedding"]
                vector = np.frombuffer(raw_bytes, dtype=np.float32)
                results.append((item, vector))
            return results

    def touch_memory(self, memory_id: str) -> None:
        """Update last_accessed_at and access_count for recency tracking."""
        with self._get_connection() as conn:
            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                UPDATE memories
                SET last_accessed_at = ?, access_count = access_count + 1
                WHERE id = ?
                """,
                (now_iso, memory_id),
            )
            conn.commit()

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            content=row["content"],
            category=row["category"],
            tags=json.loads(row["tags"]),
            importance=float(row["importance"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]) if row["last_accessed_at"] else None,
            access_count=int(row["access_count"]),
            metadata=json.loads(row["metadata"]),
        )
