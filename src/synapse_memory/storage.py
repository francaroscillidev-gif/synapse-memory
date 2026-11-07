import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from synapse_memory.crypto import CryptoManager
from synapse_memory.models import MemoryItem

DEFAULT_DB_PATH = Path.home() / ".synapse_memory" / "memories.db"


class StorageManager:
    def __init__(self, db_path: Optional[Path] = None, encryption_key: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.crypto = CryptoManager(passphrase=encryption_key)
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
                    workspace TEXT NOT NULL DEFAULT 'default',
                    tier TEXT NOT NULL DEFAULT 'episodic',
                    tags TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    expires_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    superseded_by TEXT,
                    metadata TEXT NOT NULL
                )
                """
            )
            # Safe schema migrations for existing databases
            for col, dtype, dflt in [
                ("is_active", "INTEGER", "1"),
                ("superseded_by", "TEXT", "NULL"),
                ("workspace", "TEXT", "'default'"),
                ("tier", "TEXT", "'episodic'"),
                ("expires_at", "TEXT", "NULL"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {dtype} DEFAULT {dflt}")
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_workspace ON memories(workspace);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(tier);")

            # Initialize FTS5 virtual table for lightning-fast BM25 keyword search
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        id UNINDEXED,
                        content,
                        tags,
                        tokenize = 'porter unicode61'
                    )
                    """
                )
            except sqlite3.OperationalError:
                pass

            conn.commit()

    def save_memory(self, item: MemoryItem, vector: np.ndarray) -> None:
        # Encrypt content and metadata if crypto is enabled
        stored_content = self.crypto.encrypt(item.content)
        stored_metadata = self.crypto.encrypt(json.dumps(item.metadata))

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories 
                (id, content, category, workspace, tier, tags, importance, created_at, last_accessed_at, expires_at, access_count, is_active, superseded_by, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    stored_content,
                    item.category,
                    item.workspace,
                    item.tier,
                    json.dumps(item.tags),
                    item.importance,
                    item.created_at.isoformat(),
                    item.last_accessed_at.isoformat() if item.last_accessed_at else None,
                    item.expires_at.isoformat() if item.expires_at else None,
                    item.access_count,
                    1 if item.is_active else 0,
                    item.superseded_by,
                    stored_metadata,
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding)
                VALUES (?, ?)
                """,
                (item.id, vector.astype(np.float32).tobytes()),
            )

            # Update FTS5 index (only if unencrypted for keyword search)
            try:
                conn.execute("DELETE FROM memories_fts WHERE id = ?", (item.id,))
                if not self.crypto.is_enabled:
                    conn.execute(
                        "INSERT INTO memories_fts (id, content, tags) VALUES (?, ?, ?)",
                        (item.id, item.content, " ".join(item.tags)),
                    )
            except sqlite3.OperationalError:
                pass

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
            try:
                conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
            except sqlite3.OperationalError:
                pass
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

    def prune_superseded(self, workspace: Optional[str] = None) -> int:
        """Permanently delete all inactive/superseded memories (garbage collection)."""
        with self._get_connection() as conn:
            query = "SELECT id FROM memories WHERE is_active = 0"
            params = []
            if workspace:
                query += " AND workspace = ?"
                params.append(workspace)

            cur = conn.execute(query, params)
            ids_to_delete = [r["id"] for r in cur.fetchall()]
            if not ids_to_delete:
                return 0

            placeholders = ",".join("?" for _ in ids_to_delete)
            conn.execute(f"DELETE FROM memory_embeddings WHERE memory_id IN ({placeholders})", ids_to_delete)
            conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids_to_delete)
            try:
                conn.execute(f"DELETE FROM memories_fts WHERE id IN ({placeholders})", ids_to_delete)
            except sqlite3.OperationalError:
                pass
            conn.commit()
            return len(ids_to_delete)

    def prune_expired(self) -> int:
        """Prune working memory items that have exceeded their TTL."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.execute("SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?", (now_iso,))
            ids = [r["id"] for r in cur.fetchall()]
            if not ids:
                return 0
            placeholders = ",".join("?" for _ in ids)
            conn.execute(f"DELETE FROM memory_embeddings WHERE memory_id IN ({placeholders})", ids)
            conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
            try:
                conn.execute(f"DELETE FROM memories_fts WHERE id IN ({placeholders})", ids)
            except sqlite3.OperationalError:
                pass
            conn.commit()
            return len(ids)

    def list_memories(
        self,
        limit: int = 50,
        category: Optional[str] = None,
        workspace: Optional[str] = None,
        tier: Optional[str] = None,
        active_only: bool = True,
    ) -> List[MemoryItem]:
        # Auto-prune expired working memory
        self.prune_expired()
        with self._get_connection() as conn:
            query = "SELECT * FROM memories"
            clauses = []
            params = []

            if active_only:
                clauses.append("is_active = 1")
            if category:
                clauses.append("category = ?")
                params.append(category)
            if workspace:
                clauses.append("workspace = ?")
                params.append(workspace)
            if tier:
                clauses.append("tier = ?")
                params.append(tier)

            if clauses:
                query += " WHERE " + " AND ".join(clauses)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cur = conn.execute(query, params)
            return [self._row_to_memory(row) for row in cur.fetchall()]

    def list_workspaces(self) -> List[str]:
        """Return all distinct workspaces currently in use."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT DISTINCT workspace FROM memories ORDER BY workspace ASC")
            return [r["workspace"] for r in cur.fetchall() if r["workspace"]]

    def search_fts_ids(self, query_text: str, workspace: Optional[str] = None, limit: int = 20) -> List[str]:
        """Execute BM25 keyword search using FTS5 virtual table."""
        if self.crypto.is_enabled:
            return []  # FTS is disabled when zero-knowledge encryption is on
        cleaned_words = [w for w in query_text.replace('"', ' ').replace("'", " ").split() if len(w) > 1]
        if not cleaned_words:
            return []
        fts_query = " OR ".join([f'"{w}"*' for w in cleaned_words[:6]])

        with self._get_connection() as conn:
            try:
                query = """
                    SELECT f.id 
                    FROM memories_fts f
                    JOIN memories m ON f.id = m.id
                    WHERE memories_fts MATCH ? AND m.is_active = 1
                """
                params = [fts_query]
                if workspace:
                    query += " AND m.workspace = ?"
                    params.append(workspace)

                query += " ORDER BY rank LIMIT ?"
                params.append(limit)

                cur = conn.execute(query, params)
                return [r["id"] for r in cur.fetchall()]
            except sqlite3.OperationalError:
                return []

    def get_all_vectors(
        self,
        category: Optional[str] = None,
        workspace: Optional[str] = None,
        tier: Optional[str] = None,
        active_only: bool = True,
    ) -> List[Tuple[MemoryItem, np.ndarray]]:
        """Fetch memories and their embeddings for ranking calculations."""
        # Auto-prune expired items
        self.prune_expired()
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
            if workspace:
                clauses.append("m.workspace = ?")
                params.append(workspace)
            if tier:
                clauses.append("m.tier = ?")
                params.append(tier)

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

    def get_analytics(self, workspace: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate statistical metrics for the analytics dashboard."""
        with self._get_connection() as conn:
            where_ws = " WHERE workspace = ?" if workspace else ""
            params_ws = [workspace] if workspace else []

            total = conn.execute(f"SELECT COUNT(*) as c FROM memories{where_ws}", params_ws).fetchone()["c"]
            active_clause = " WHERE is_active = 1" + (" AND workspace = ?" if workspace else "")
            active = conn.execute(f"SELECT COUNT(*) as c FROM memories{active_clause}", params_ws).fetchone()["c"]
            superseded = total - active

            # Tiers breakdown
            cur_tier = conn.execute(
                f"SELECT tier, COUNT(*) as c FROM memories{where_ws} GROUP BY tier",
                params_ws,
            )
            tiers = {r["tier"]: r["c"] for r in cur_tier.fetchall()}

            # Category breakdown
            cur_cat = conn.execute(
                f"SELECT category, COUNT(*) as c FROM memories{where_ws} GROUP BY category ORDER BY c DESC",
                params_ws,
            )
            categories = {r["category"]: r["c"] for r in cur_cat.fetchall()}

            # Workspaces list
            cur_ws = conn.execute("SELECT workspace, COUNT(*) as c FROM memories GROUP BY workspace")
            workspaces = {r["workspace"]: r["c"] for r in cur_ws.fetchall()}

            # Most accessed
            most_accessed_cur = conn.execute(
                f"SELECT id, content, access_count FROM memories{where_ws} ORDER BY access_count DESC LIMIT 5",
                params_ws,
            )
            most_accessed = []
            for r in most_accessed_cur.fetchall():
                decrypted_content = self.crypto.decrypt(r["content"])
                most_accessed.append({"id": r["id"], "content": decrypted_content[:40], "access_count": r["access_count"]})

            return {
                "total_memories": total,
                "active_memories": active,
                "superseded_memories": superseded,
                "tiers": tiers,
                "categories": categories,
                "workspaces": workspaces,
                "most_accessed": most_accessed,
                "encryption_enabled": self.crypto.is_enabled,
            }

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryItem:
        keys = row.keys()
        is_active = bool(row["is_active"]) if "is_active" in keys else True
        superseded_by = row["superseded_by"] if "superseded_by" in keys else None
        workspace = row["workspace"] if "workspace" in keys else "default"
        tier = row["tier"] if "tier" in keys else "episodic"
        expires_at = datetime.fromisoformat(row["expires_at"]) if ("expires_at" in keys and row["expires_at"]) else None

        # Transparent Decryption
        raw_content = row["content"]
        raw_metadata = row["metadata"]
        decrypted_content = self.crypto.decrypt(raw_content)
        decrypted_metadata_str = self.crypto.decrypt(raw_metadata)

        try:
            metadata = json.loads(decrypted_metadata_str)
        except Exception:
            metadata = {}

        return MemoryItem(
            id=row["id"],
            content=decrypted_content,
            category=row["category"],
            workspace=workspace,
            tier=tier,
            tags=json.loads(row["tags"]),
            importance=float(row["importance"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]) if row["last_accessed_at"] else None,
            expires_at=expires_at,
            access_count=int(row["access_count"]),
            is_active=is_active,
            superseded_by=superseded_by,
            metadata=metadata,
        )
