import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


class ClientInstaller:
    """Automates 1-click MCP configuration setup for Claude Desktop, Cursor, and Antigravity."""

    @staticmethod
    def get_synapse_command() -> Tuple[str, list]:
        """Detect current python/uv interpreter and command args to launch synapse MCP."""
        uv_path = shutil.which("uv")
        if uv_path:
            return uv_path, ["run", "synapse", "serve-mcp"]
        # Fallback to python -m
        return sys.executable, ["-m", "synapse_memory.mcp_server"]

    @classmethod
    def get_claude_config_path(cls) -> Path:
        system = platform.system()
        if system == "Darwin":
            return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        elif system == "Windows":
            appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
        else:
            return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"

    @classmethod
    def setup_claude(cls) -> Tuple[bool, str]:
        path = cls.get_claude_config_path()
        cmd, args = cls.get_synapse_command()
        server_config = {
            "command": cmd,
            "args": args,
        }
        return cls._inject_mcp_config(path, "synapse-memory", server_config)

    @classmethod
    def setup_cursor(cls, workspace_path: Optional[Path] = None) -> Tuple[bool, str]:
        if workspace_path:
            path = workspace_path / ".cursor" / "mcp.json"
        else:
            path = Path.home() / ".cursor" / "mcp.json"
        cmd, args = cls.get_synapse_command()
        server_config = {
            "command": cmd,
            "args": args,
        }
        return cls._inject_mcp_config(path, "synapse-memory", server_config)

    @classmethod
    def _inject_mcp_config(cls, config_path: Path, server_name: str, server_dict: Dict) -> Tuple[bool, str]:
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}

            if "mcpServers" not in data or not isinstance(data["mcpServers"], dict):
                data["mcpServers"] = {}

            data["mcpServers"][server_name] = server_dict

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True, str(config_path)
        except Exception as e:
            return False, str(e)
