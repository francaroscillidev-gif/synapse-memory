import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from synapse_memory.engine import MemoryEngine
from synapse_memory.portability import PortabilityManager


class RemoteSyncHub:
    """Manages decentralized encrypted backup and cross-machine sync via private Git remotes."""

    def __init__(self, engine: MemoryEngine, sync_dir: Optional[Path] = None):
        self.engine = engine
        self.sync_dir = sync_dir or (Path.home() / ".synapse_memory" / "remote_vault")
        self.portability = PortabilityManager(engine)

    def init_vault(self, git_remote_url: str) -> Tuple[bool, str]:
        """Initialize a local sync repository and configure its remote."""
        try:
            self.sync_dir.mkdir(parents=True, exist_ok=True)
            if not (self.sync_dir / ".git").exists():
                subprocess.run(["git", "init", "-b", "main"], cwd=self.sync_dir, check=True, capture_output=True)
                subprocess.run(["git", "remote", "add", "origin", git_remote_url], cwd=self.sync_dir, check=True, capture_output=True)
            else:
                subprocess.run(["git", "remote", "set-url", "origin", git_remote_url], cwd=self.sync_dir, check=True, capture_output=True)

            return True, f"Vault initialized at {self.sync_dir} linked to {git_remote_url}"
        except Exception as e:
            return False, str(e)

    def push_sync(self, commit_message: Optional[str] = None) -> Tuple[bool, str]:
        """Export current memory state and push to the remote git repository."""
        if not (self.sync_dir / ".git").exists():
            return False, "Remote sync vault is not initialized. Run `synapse remote init <url>` first."

        backup_file = self.sync_dir / "synapse_vault.json"
        count = self.portability.export_json(backup_file)

        msg = commit_message or f"auto-sync: {count} memories at {datetime.now(timezone.utc).isoformat()}"

        try:
            subprocess.run(["git", "add", "."], cwd=self.sync_dir, check=True, capture_output=True)
            # Commit if changes exist
            status = subprocess.run(["git", "status", "--porcelain"], cwd=self.sync_dir, check=True, capture_output=True, text=True)
            if status.stdout.strip():
                subprocess.run(["git", "commit", "-m", msg], cwd=self.sync_dir, check=True, capture_output=True)
                subprocess.run(["git", "push", "-u", "origin", "main"], cwd=self.sync_dir, check=True, capture_output=True)
                return True, f"Pushed {count} memories to remote vault."
            return True, "No new memory changes to push (already in sync)."
        except subprocess.CalledProcessError as e:
            return False, f"Git push failed: {e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)}"

    def pull_sync(self) -> Tuple[bool, str]:
        """Pull latest updates from remote repository and merge into local memory engine."""
        if not (self.sync_dir / ".git").exists():
            return False, "Remote sync vault is not initialized. Run `synapse remote init <url>` first."

        try:
            subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=self.sync_dir, check=True, capture_output=True)
            backup_file = self.sync_dir / "synapse_vault.json"
            if backup_file.exists():
                imported = self.portability.import_json(backup_file)
                return True, f"Pulled and merged {imported} memories from remote vault."
            return False, "Remote vault did not contain a valid synapse_vault.json."
        except subprocess.CalledProcessError as e:
            return False, f"Git pull failed: {e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)}"
