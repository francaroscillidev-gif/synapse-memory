import math
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from synapse_memory.embedder import LocalEmbedder
from synapse_memory.evolver import MemoryEvolver
from synapse_memory.extractor import SmartExtractor
from synapse_memory.models import MemoryItem, SearchResult
from synapse_memory.security import SecurityRedactor
from synapse_memory.storage import StorageManager


class MemoryEngine:
    def __init__(self, db_path: Optional[Path] = None, embedder: Optional[LocalEmbedder] = None):
        self.storage = StorageManager(db_path=db_path)
        self.embedder = embedder or LocalEmbedder()
        self.evolver = MemoryEvolver(self.storage)

    def store(
        self,
        content: str,
        category: str = "general",
        workspace: str = "default",
        tier: str = "episodic",
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
        memory_id: Optional[str] = None,
        auto_enrich: bool = True,
        auto_evolve: bool = True,
        redact_secrets: bool = True,
    ) -> Tuple[MemoryItem, List[MemoryItem]]:
        """
        Store a new fact, rule, or observation into local memory.
        Returns (created_item, list_of_superseded_items).
        """
        cleaned_content = content.strip()
        redactions_count = 0
        if redact_secrets:
            cleaned_content, redactions_count = SecurityRedactor.redact(cleaned_content)
        tags_set = set([t.strip().lower() for t in (tags or []) if t.strip()])

        # Automatic tag and category enrichment
        if auto_enrich:
            extracted_tags = SmartExtractor.extract_tags(cleaned_content)
            tags_set.update(extracted_tags)
            if category.lower() == "general":
                category = SmartExtractor.infer_category(cleaned_content, default="general")

        metadata = metadata or {}
        if redactions_count > 0:
            metadata["security_redacted"] = redactions_count
        item_id = memory_id or f"mem_{uuid.uuid4().hex[:10]}"

        # Calculate TTL expiration
        expires_at = None
        if ttl_seconds:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        elif tier == "working":
            # Default 1 hour TTL for volatile working memory
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=3600)

        item = MemoryItem(
            id=item_id,
            content=cleaned_content,
            category=category.strip().lower(),
            workspace=workspace.strip().lower(),
            tier=tier,
            tags=sorted(list(tags_set)),
            importance=max(0.0, min(1.0, importance)),
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            metadata=metadata,
        )

        vector = self.embedder.embed_text(item.content)

        # Detect contradictions and supersede older memories if necessary
        superseded: List[MemoryItem] = []
        if auto_evolve and tier != "working":
            superseded = self.evolver.check_and_evolve(new_item=item, new_vector=vector)

        self.storage.save_memory(item, vector)
        return item, superseded

    def recall(
        self,
        query: str,
        category: Optional[str] = None,
        workspace: Optional[str] = None,
        tier: Optional[str] = None,
        limit: int = 5,
        min_score: float = 0.25,
        sim_weight: float = 0.65,
        recency_weight: float = 0.20,
        importance_weight: float = 0.15,
        active_only: bool = True,
    ) -> List[SearchResult]:
        """
        Recall memories with a combined scoring function:
        - Dense Semantic Similarity (via Local ONNX embeddings)
        - Sparse BM25 Keyword Search (via SQLite FTS5)
        - Recency Decay (exponential half-life decay)
        - Explicit Importance
        """
        all_entries = self.storage.get_all_vectors(category=category, workspace=workspace, tier=tier, active_only=active_only)
        if not all_entries:
            return []

        # Sparse keyword search IDs via FTS5
        fts_hits = set(self.storage.search_fts_ids(query, workspace=workspace, limit=limit * 2))

        query_vec = self.embedder.embed_text(query)
        now = datetime.now(timezone.utc)
        results: List[SearchResult] = []

        query_lower = query.lower()

        for item, vec in all_entries:
            # 1. Cosine similarity
            cosine_sim = float(np.dot(query_vec, vec))
            # Normalizziamo su scala 0..1
            sim_score = max(0.0, (cosine_sim + 1.0) / 2.0)

            # BM25 / FTS5 exact keyword hit boost
            fts_boost = 0.12 if item.id in fts_hits else 0.0
            sim_score = min(1.0, sim_score + fts_boost)

            # Keyword tag boost (se i tag sono menzionati esplicitamente nella query)
            tag_hits = sum(1 for t in item.tags if t in query_lower)
            if tag_hits > 0:
                sim_score = min(1.0, sim_score + 0.10 * tag_hits)

            # 2. Recency score (decadimento con emivita di 14 giorni)
            ref_date = item.last_accessed_at or item.created_at
            if ref_date.tzinfo is None:
                ref_date = ref_date.replace(tzinfo=timezone.utc)
            delta_days = max(0.0, (now - ref_date).total_seconds() / 86400.0)
            recency_score = math.exp(-delta_days / 14.0)

            # 3. Importanza
            imp_score = item.importance

            # 4. Final blended score
            final_score = (
                sim_weight * sim_score +
                recency_weight * recency_score +
                importance_weight * imp_score
            )

            if final_score >= min_score:
                fts_note = ", fts=match" if item.id in fts_hits else ""
                rationale = (
                    f"sim={sim_score:.2f}, recency={recency_score:.2f}, imp={imp_score:.2f}{fts_note}"
                )
                results.append(
                    SearchResult(
                        item=item,
                        semantic_score=sim_score,
                        recency_score=recency_score,
                        final_score=final_score,
                        rationale=rationale,
                    )
                )

        # Ordina per final_score decrescente
        results.sort(key=lambda r: r.final_score, reverse=True)
        top_results = results[:limit]

        # Aggiorna lo stato di accesso (recency boost)
        for res in top_results:
            self.storage.touch_memory(res.item.id)

        return top_results

    def format_context_for_prompt(self, query: str, workspace: Optional[str] = None, limit: int = 5) -> str:
        """Returns a ready-to-inject Markdown context block for any AI prompt."""
        matches = self.recall(query=query, workspace=workspace, limit=limit)
        if not matches:
            return ""

        lines = ["### [Synapse Long-Term Memory Context]"]
        for i, match in enumerate(matches, 1):
            category = match.item.category.upper()
            tags = f" (#{', #'.join(match.item.tags)})" if match.item.tags else ""
            lines.append(f"{i}. [{category}] {match.item.content}{tags}")
        lines.append("### [End Memory Context]\n")
        return "\n".join(lines)

    def compact_context(self, query: str, workspace: Optional[str] = None, max_approx_tokens: int = 500) -> str:
        """
        Retrieves matching memories while adhering strictly to a token budget.
        Approximates ~4 characters per token.
        """
        matches = self.recall(query=query, workspace=workspace, limit=15)
        if not matches:
            return ""

        char_budget = max_approx_tokens * 4
        current_chars = 0
        selected = []

        for m in matches:
            entry = f"- [{m.item.category.upper()}] {m.item.content}"
            if current_chars + len(entry) + 1 > char_budget:
                break
            selected.append(entry)
            current_chars += len(entry) + 1

        if not selected:
            return ""

        return "### [Synapse Compact Memory Context]\n" + "\n".join(selected) + "\n### [End Context]\n"
