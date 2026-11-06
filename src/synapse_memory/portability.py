import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from synapse_memory.engine import MemoryEngine


class PortabilityManager:
    def __init__(self, engine: MemoryEngine):
        self.engine = engine

    def export_json(self, output_path: Path) -> int:
        """Export all memories and metadata into a portable JSON backup file."""
        items = self.engine.storage.list_memories(limit=10000)
        data = {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "count": len(items),
            "memories": [it.model_dump(mode="json") for it in items],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return len(items)

    def import_json(self, input_path: Path) -> int:
        """Import memories from a portable JSON backup file."""
        if not input_path.exists():
            raise FileNotFoundError(f"Backup file not found: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_memories = data.get("memories", [])
        count = 0
        for m in raw_memories:
            content = m.get("content")
            if not content:
                continue
            self.engine.store(
                content=content,
                category=m.get("category", "general"),
                tags=m.get("tags", []),
                importance=float(m.get("importance", 0.5)),
                memory_id=m.get("id"),
                metadata=m.get("metadata", {}),
                auto_enrich=False,
            )
            count += 1

        return count

    def export_markdown(self, output_path: Path) -> int:
        """Export memories as human-readable Markdown notes (e.g. for Obsidian or Notion)."""
        items = self.engine.storage.list_memories(limit=10000)
        lines = [
            "# 🧠 Synapse Memory Export",
            f"*Exported at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
            f"*Total entries: {len(items)}*",
            "",
            "---",
            "",
        ]

        # Group by category
        categories: Dict[str, List[Any]] = {}
        for it in items:
            categories.setdefault(it.category, []).append(it)

        for cat, mems in sorted(categories.items()):
            lines.append(f"## Category: `{cat.upper()}`")
            lines.append("")
            for m in mems:
                tags = " ".join([f"#{t}" for t in m.tags])
                lines.append(f"- **[{m.id}]** (Priority: {int(m.importance*100)}%) {tags}")
                lines.append(f"  > {m.content}")
                lines.append("")
            lines.append("---")
            lines.append("")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return len(items)
