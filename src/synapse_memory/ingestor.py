import re
from pathlib import Path
from typing import List, Tuple

from synapse_memory.engine import MemoryEngine


class FileIngestor:
    """Ingests Markdown, text, and code notes by chunking them into semantic sections."""

    def __init__(self, engine: MemoryEngine):
        self.engine = engine

    def ingest_text(self, text: str, source_name: str, category: str = "document") -> int:
        chunks = self._chunk_content(text)
        count = 0
        for header, chunk in chunks:
            if len(chunk.strip()) < 15:
                continue
            content = f"[{source_name}] {header}\n{chunk}" if header else f"[{source_name}] {chunk}"
            self.engine.store(
                content=content,
                category=category,
                tags=[source_name.lower().replace(".", "_")],
                importance=0.6,
                metadata={"source": source_name, "header": header},
            )
            count += 1
        return count

    def ingest_file(self, file_path: Path) -> int:
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Skip binary files safely
            return 0

        source_name = file_path.name
        # Guess category based on extension
        category = "code" if file_path.suffix in [".py", ".ts", ".js", ".rs", ".go", ".json", ".yaml", ".yml"] else "document"
        return self.ingest_text(content, source_name=source_name, category=category)

    def ingest_directory(self, dir_path: Path, glob_pattern: str = "**/*.md") -> Tuple[int, int]:
        """Ingests all matching text files in a directory."""
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        total_files = 0
        total_chunks = 0

        # Disallow binary/huge directories like .git, node_modules, .venv
        ignored_parts = {".git", "node_modules", ".venv", "__pycache__", "dist", "build"}

        for p in dir_path.glob(glob_pattern):
            if any(part in p.parts for part in ignored_parts):
                continue
            if p.is_file():
                chunks_added = self.ingest_file(p)
                if chunks_added > 0:
                    total_files += 1
                    total_chunks += chunks_added

        return total_files, total_chunks

    def _chunk_content(self, text: str) -> List[Tuple[str, str]]:
        """Split text logically by Markdown headers or paragraph blocks."""
        chunks: List[Tuple[str, str]] = []
        # Check if text has Markdown headers
        header_pattern = re.compile(r"^(#{1,4}\s+.+)$", re.MULTILINE)
        parts = header_pattern.split(text)

        if len(parts) > 1:
            current_header = ""
            for part in parts:
                part_clean = part.strip()
                if not part_clean:
                    continue
                if header_pattern.match(part_clean):
                    current_header = part_clean
                else:
                    # Further split large sections into paragraphs if needed
                    sub_paras = part_clean.split("\n\n")
                    for para in sub_paras:
                        para_clean = para.strip()
                        if len(para_clean) >= 20:
                            chunks.append((current_header, para_clean))
        else:
            # Normal paragraph chunking
            paragraphs = text.split("\n\n")
            for para in paragraphs:
                para_clean = para.strip()
                if len(para_clean) >= 20:
                    chunks.append(("", para_clean))

        return chunks
