import re
from pathlib import Path
from typing import List, Optional, Set

from synapse_memory.engine import MemoryEngine

TRIVIAL_COMMANDS = {
    "ls", "ll", "la", "cd", "pwd", "clear", "exit", "history", "git status", "git diff", "cat", "echo"
}


class ShellHistorySync:
    """Extracts and indexes high-value developer shell commands into long-term memory."""

    def __init__(self, engine: MemoryEngine):
        self.engine = engine

    @staticmethod
    def get_default_history_file() -> Optional[Path]:
        home = Path.home()
        zsh_history = home / ".zsh_history"
        bash_history = home / ".bash_history"
        if zsh_history.exists():
            return zsh_history
        if bash_history.exists():
            return bash_history
        return None

    def sync_history(self, history_file: Optional[Path] = None, max_entries: int = 50) -> int:
        target_file = history_file or self.get_default_history_file()
        if not target_file or not target_file.exists():
            return 0

        try:
            content = target_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return 0

        lines = content.splitlines()
        extracted_commands: List[str] = []
        seen: Set[str] = set()

        for line in reversed(lines):
            # Clean ZSH history timestamps: ': 1629839281:0;command'
            clean_cmd = re.sub(r"^:\s*\d+:\d+;", "", line).strip()
            if not clean_cmd:
                continue

            first_word = clean_cmd.split()[0].lower() if clean_cmd.split() else ""
            if first_word in TRIVIAL_COMMANDS or clean_cmd in seen:
                continue

            # Only index commands of substantial length or containing flags/subcommands
            if len(clean_cmd) >= 12 or any(tok in clean_cmd for tok in ["--", "-", "|", ">", "docker", "kubectl", "git"]):
                seen.add(clean_cmd)
                extracted_commands.append(clean_cmd)
                if len(extracted_commands) >= max_entries:
                    break

        count = 0
        for cmd in extracted_commands:
            self.engine.store(
                content=f"Shell command snippet: `{cmd}`",
                category="code",
                workspace="terminal",
                tags=["shell", "terminal-command"],
                importance=0.6,
                auto_enrich=True,
                auto_evolve=False,
            )
            count += 1

        return count
