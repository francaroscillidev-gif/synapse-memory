# 🧠 Synapse Memory

> **Local-First, Open Source Long-Term Memory Engine for AI Agents & Humans**  
> Powered by fast ONNX embeddings, SQLite persistence, and the Model Context Protocol (MCP).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

---

## ⚡ What makes Synapse Revolutionary?

Most modern AI agents suffer from **"Context Amnesia"**: they lose personal preferences, codebase guidelines, and project constraints between chat sessions. Existing RAG solutions rely on expensive cloud vector databases or naive vector matching that ignores **time** and **priority**.

**Synapse** provides a **Local-First Episodic & Semantic Memory** with zero cloud leaks:

1. **100% Private & Local:** Runs entirely on your local machine using embedded ONNX models (via FastEmbed) and SQLite. No API keys, no external database servers, no data leakage.
2. **Hybrid Cognitive Ranking:** Combines:
   - **Semantic Similarity** (dense vector cosine distance)
   - **Ebbinghaus Forgetting Curve / Recency Decay** (frequently recalled facts stay fresh)
   - **Explicit Importance & Tag Weighting** (critical architectural rules outrank casual facts)
3. **Universal MCP Server:** Integrates natively with any Model Context Protocol (MCP) host, such as **Claude Desktop**, **Cursor**, **Antigravity**, or custom agent loops.
4. **Instant Terminal CLI:** Intuitive commands for humans and shell scripts.

---

## 🚀 Quick Start

### 1. Installation

Clone and install with [uv](https://github.com/astral-sh/uv) (fastest Python package manager):

```bash
git clone https://github.com/your-username/synapse-memory.git
cd synapse-memory
uv sync
```

Or install in editable mode:
```bash
pip install -e .
```

---

## 💻 CLI Usage

### Store a memory (`remember`)
```bash
synapse remember "We use PostgreSQL for all production database schemas" \
  --category architecture \
  --tag postgres --tag database \
  --importance 0.9
```

### Natural Language Search (`recall`)
```bash
synapse recall "which database do we use in prod?"
```

Output:
```text
          🧠 Synapse Recall: 'which database do we use in prod?'          
┏━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Rank ┃ ID           ┃ Category     ┃ Score    ┃ Content                                             ┃ Rationale               ┃
┡━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1    │ mem_7f09dc09 │ architecture │ 0.90     │ We use PostgreSQL for all production database sche… │ sim=0.89, recency=1.00  │
└──────┴──────────────┴──────────────┴──────────┴─────────────────────────────────────────────────────┴─────────────────────────┘
```

### Generate Context Block for Prompts (`prompt`)
Output clean markdown context directly into your LLM system prompt:
```bash
synapse prompt "Write an API route"
```

### Inspect Stored Memories (`list`)
```bash
synapse list --limit 10
```

### Ingest Files and Documentation (`ingest`)
Chunk and embed notes, Markdown files, or whole project folders:
```bash
# Ingest single file
synapse ingest docs/architecture.md

# Ingest all markdown files in a folder
synapse ingest ./notes --glob "**/*.md"
```

### Export & Backup (`export` / `import`)
Export all memories to portable JSON or Markdown notes (compatible with Obsidian/Notion):
```bash
# Export to JSON backup
synapse export --out backup.json

# Export to human-readable Markdown notes
synapse export --out notes.md

# Restore / merge from backup
synapse import backup.json
```

### Launch Interactive Web Dashboard & Knowledge Graph (`ui`)
Launch the local dark-mode dashboard with interactive force-directed graph view:
```bash
synapse ui
# Opens http://127.0.0.1:8484
```

### Forget / Delete Memory (`forget`)
```bash
synapse forget mem_7f09dc09
```

---

## 🔌 Universal MCP Integration (Claude, Cursor, Antigravity)

Synapse exposes native MCP tools over standard I/O:
* `remember`: Store knowledge or preferences.
* `recall`: Retrieve relevant memories based on semantic similarity and recency.
* `forget`: Delete outdated facts.

### Claude Desktop Configuration
Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "synapse-memory": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/synapse-memory",
        "run",
        "synapse",
        "serve-mcp"
      ]
    }
  }
}
```

---

## 🐍 Python SDK Example

Integrate Synapse into your custom Python agents:

```python
from synapse_memory import MemoryEngine

engine = MemoryEngine()

# 1. Store knowledge
engine.store(
    content="User prefers succinct answers with runnable code snippets",
    category="preference",
    tags=["style", "preferences"],
    importance=0.8,
)

# 2. Inject context into your LLM prompt
prompt_context = engine.format_context_for_prompt("How should I answer this user query?")
system_prompt = f"""You are a helpful coding assistant.
{prompt_context}
"""
```

---

## 🧪 Testing

Run unit tests locally:

```bash
uv run pytest
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).
