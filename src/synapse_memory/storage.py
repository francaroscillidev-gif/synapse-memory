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
                    is_active INTEGER NOT NULL DEFAULT 1,
                    superseded_by TEXT,
                    metadata TEXT NOT NULL
                )
                """
            )
            # Safe schema migration for existing databases
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN superseded_by TEXT")
            except sqlite3.OperationalError:
                pass

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
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active);
                """
            )
            conn.commit()

    def save_memory(self, item: MemoryItem, vector: np.ndarray) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories 
                (id, content, category, tags, importance, created_at, last_accessed_at, access_count, is_active, superseded_by, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    1 if item.is_active else 0,
                    item.superseded_by,
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

    def supersede_memory(self, old_memory_id: str, new_memory_id: str) -> bool:
        """Mark an older memory as superseded/inactive when an updated fact arrives."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE memories
                SET is_active = 0, superseded_by = ?
                WHERE id = ?
                """,
                (new_memory_id, old_memory_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def list_memories(self, limit: int = 50, category: Optional[str] = None, active_only: bool = True) -> List[MemoryItem]:
        with self._get_connection() as conn:
            query = "SELECT * FROM memories"
            clauses = []
            params = []

            if active_only:
                clauses.append("is_active = 1")
            if category:
                clauses.append("category = ?")
                params.append(category)

            if clauses:
                query += " WHERE " + " AND ".join(clauses)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cur = conn.execute(query, params)
            return [self._row_to_memory(row) for row in cur.fetchall()]

    def get_all_vectors(self, category: Optional[str] = None, active_only: bool = True) -> List[Tuple[MemoryItem, np.ndarray]]:
        """Fetch memories and their embeddings for ranking calculations."""
        with self._get_connection() as conn:
            query = """
                SELECT m.*, e.embedding 
                FROM memories m
                JOIN memory_embeddings e ON m.id = e.memory_id
            """
            clauses = []
            params = []

            if active_only:
                clauses.append("m.is_active = 1")
            if category:
                clauses.append("m.category = ?")
                params.append(category)

            if clauses:
                query += " WHERE " + " AND ".join(clauses)

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
        # Check if is_active exists in row
        keys = row.keys()
        is_active = bool(row["is_active"]) if "is_active" in keys else True
        superseded_by = row["superseded_by"] if "superseded_by" in keys else None

        return MemoryItem(
            id=row["id"],
            content=row["content"],
            category=row["category"],
            tags=json.loads(row["tags"]),
            importance=float(row["importance"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]) if row["last_accessed_at"] else None,
            access_count=int(row["access_count"]),
            is_active=is_active,
            superseded_by=superseded_by,
            metadata=json.loads(row["metadata"]),
        )
