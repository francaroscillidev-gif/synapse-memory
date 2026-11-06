import hashlib
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from synapse_memory.engine import MemoryEngine
from synapse_memory.ingestor import FileIngestor


class DirectoryWatcher:
    """Watches a directory for new or modified files and re-syncs memories automatically."""

    def __init__(self, engine: MemoryEngine, dir_path: Path, glob_pattern: str = "**/*.md"):
        self.engine = engine
        self.dir_path = dir_path.resolve()
        self.glob_pattern = glob_pattern
        self.ingestor = FileIngestor(engine)
        self._file_hashes: Dict[str, str] = {}

    def _hash_file(self, file_path: Path) -> Optional[str]:
        try:
            content = file_path.read_bytes()
            return hashlib.sha256(content).hexdigest()
        except Exception:
            return None

    def scan_once(self) -> int:
        """Scan folder once, returning the number of re-ingested files."""
        updated = 0
        ignored = {".git", "node_modules", ".venv", "__pycache__", "dist", "build"}

        for p in self.dir_path.glob(self.glob_pattern):
            if any(part in p.parts for part in ignored):
                continue
            if not p.is_file():
                continue

            current_hash = self._hash_file(p)
            if not current_hash:
                continue

            path_key = str(p)
            if path_key not in self._file_hashes or self._file_hashes[path_key] != current_hash:
                self._file_hashes[path_key] = current_hash
                self.ingestor.ingest_file(p)
                updated += 1

        return updated

    def start_watching(self, interval_seconds: float = 2.0, on_change: Optional[Callable[[int], None]] = None):
        """Run continuous watch loop."""
        print(f"👀 Watching directory '{self.dir_path}' for changes ({self.glob_pattern})...")
        # Initial scan
        initial_count = self.scan_once()
        if on_change and initial_count > 0:
            on_change(initial_count)

        try:
            while True:
                time.sleep(interval_seconds)
                changes = self.scan_once()
                if changes > 0 and on_change:
                    on_change(changes)
        except KeyboardInterrupt:
            print("\n🛑 Stopped directory watcher.")
