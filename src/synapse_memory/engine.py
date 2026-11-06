import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from synapse_memory.embedder import LocalEmbedder
from synapse_memory.models import MemoryItem, SearchResult
from synapse_memory.storage import StorageManager


class MemoryEngine:
    def __init__(self, db_path: Optional[Path] = None, embedder: Optional[LocalEmbedder] = None):
        self.storage = StorageManager(db_path=db_path)
        self.embedder = embedder or LocalEmbedder()

    def store(
        self,
        content: str,
        category: str = "general",
        tags: Optional[List[str]] = None,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
        memory_id: Optional[str] = None,
    ) -> MemoryItem:
        """Store a new fact, rule, or observation into local memory."""
        tags = tags or []
        metadata = metadata or {}
        item_id = memory_id or f"mem_{uuid.uuid4().hex[:10]}"

        item = MemoryItem(
            id=item_id,
            content=content.strip(),
            category=category.strip().lower(),
            tags=[t.strip().lower() for t in tags if t.strip()],
            importance=max(0.0, min(1.0, importance)),
            created_at=datetime.now(timezone.utc),
            metadata=metadata,
        )

        vector = self.embedder.embed_text(item.content)
        self.storage.save_memory(item, vector)
        return item

    def recall(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
        min_score: float = 0.25,
        sim_weight: float = 0.65,
        recency_weight: float = 0.20,
        importance_weight: float = 0.15,
    ) -> List[SearchResult]:
        """
        Recall memories with a combined scoring function:
        - Semantic Similarity (via Local ONNX embeddings)
        - Recency Decay (exponential half-life decay)
        - Explicit Importance
        """
        all_entries = self.storage.get_all_vectors(category=category)
        if not all_entries:
            return []

        query_vec = self.embedder.embed_text(query)
        now = datetime.now(timezone.utc)
        results: List[SearchResult] = []

        query_lower = query.lower()

        for item, vec in all_entries:
            # 1. Cosine similarity
            cosine_sim = float(np.dot(query_vec, vec))
            # Normalizziamo su scala 0..1
            sim_score = max(0.0, (cosine_sim + 1.0) / 2.0)

            # Keyword tag boost (se i tag sono menzionati esplicitamente nella query)
            tag_hits = sum(1 for t in item.tags if t in query_lower)
            if tag_hits > 0:
                sim_score = min(1.0, sim_score + 0.10 * tag_hits)

            # 2. Recency score (decadimento con emivita di 14 giorni)
            ref_date = item.last_accessed_at or item.created_at
            if ref_date.tzinfo is None:
                ref_date = ref_date.replace(tzinfo=timezone.utc)
            delta_days = max(0.0, (now - ref_date).total_seconds() / 86400.0)
            # Mezza vita = 14 giorni
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
                rationale = (
                    f"sim={sim_score:.2f}, recency={recency_score:.2f}, imp={imp_score:.2f}"
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

    def format_context_for_prompt(self, query: str, limit: int = 5) -> str:
        """Returns a ready-to-inject Markdown context block for any AI prompt."""
        matches = self.recall(query=query, limit=limit)
        if not matches:
            return ""

        lines = ["### [Synapse Long-Term Memory Context]"]
        for i, match in enumerate(matches, 1):
            category = match.item.category.upper()
            tags = f" (#{', #'.join(match.item.tags)})" if match.item.tags else ""
            lines.append(f"{i}. [{category}] {match.item.content}{tags}")
        lines.append("### [End Memory Context]\n")
        return "\n".join(lines)
