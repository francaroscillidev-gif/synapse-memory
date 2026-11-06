import json
import urllib.request
from typing import Dict, List, Optional
from synapse_memory.engine import MemoryEngine
from synapse_memory.models import MemoryItem


class MemoryConsolidator:
    """Consolidates scattered memory fragments into structured macro-knowledge summaries."""

    def __init__(self, engine: MemoryEngine, ollama_host: str = "http://localhost:11434"):
        self.engine = engine
        self.ollama_host = ollama_host

    def is_ollama_available(self) -> bool:
        """Check if local Ollama daemon is running."""
        try:
            req = urllib.request.Request(f"{self.ollama_host}/api/tags", headers={"User-Agent": "Synapse"})
            with urllib.request.urlopen(req, timeout=1.0) as res:
                return res.status == 200
        except Exception:
            return False

    def consolidate_category(
        self, category: str, model: str = "qwen2.5:1.5b", store_as_memory: bool = True
    ) -> Optional[str]:
        """Consolidate memories within a category into a synthesized knowledge summary."""
        items = self.engine.storage.list_memories(limit=100, category=category, active_only=True)
        if len(items) < 2:
            return None

        # Build raw text
        bullets = [f"- {it.content}" for it in items]
        raw_text = "\n".join(bullets)

        summary = ""
        # Check if Ollama is running locally
        if self.is_ollama_available():
            try:
                summary = self._query_ollama(raw_text, category=category, model=model)
            except Exception:
                summary = ""

        # Fallback to structured rule-based consolidation
        if not summary:
            summary = self._deterministic_consolidation(items, category=category)

        if store_as_memory and summary:
            # Store synthesized macro-memory
            self.engine.store(
                content=f"[Consolidated {category.upper()}] {summary}",
                category=category,
                tags=["consolidated", "summary", category],
                importance=0.9,
                auto_enrich=False,
                auto_evolve=False,
            )

        return summary

    def _query_ollama(self, bullets_text: str, category: str, model: str) -> str:
        prompt = (
            f"You are a knowledge consolidation agent. Combine the following facts into a concise, unified summary for the category '{category}':\n\n"
            f"{bullets_text}\n\n"
            "Summary (concise and factual):"
        )
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.ollama_host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30.0) as res:
            data = json.loads(res.read().decode("utf-8"))
            return data.get("response", "").strip()

    def _deterministic_consolidation(self, items: List[MemoryItem], category: str) -> str:
        """Deterministic fallback that removes duplicates and organizes bullet points."""
        unique_points = []
        seen = set()
        for it in items:
            cleaned = it.content.strip()
            if cleaned not in seen:
                seen.add(cleaned)
                unique_points.append(cleaned)

        return f"Summary of {len(unique_points)} active facts across {category}: " + " | ".join(unique_points[:10])
