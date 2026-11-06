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
3. **Memory Evolution & Self-Correction:** Automatically detects when a new memory contradicts or supersedes an older rule (e.g. *"switched from Postgres to CockroachDB"*), deprecating the outdated memory to eliminate AI hallucination.
4. **Token Budget Compressor:** Compresses recalled knowledge to strictly respect context window limits.
5. **Real-Time Directory Watcher:** Watches note folders or repositories and auto-syncs modified files into memory using SHA256 diffing.
6. **Universal MCP Server:** Integrates natively with any Model Context Protocol (MCP) host, such as **Claude Desktop**, **Cursor**, **Antigravity**, or custom agent loops.
7. **Automatic PII & Secrets Redaction:** Scans and sanitizes API keys (`sk-...`, `ghp_...`), JWTs, AWS credentials, and passwords before indexing to avoid leaks.
8. **1-Click MCP Setup:** Automated auto-injection for Claude Desktop and Cursor.
9. **Memory Health & Consolidation:** Live analytics, garbage collection, and local Ollama-powered knowledge synthesis.
10. **Instant Terminal CLI & Web Graph Dashboard:** Rich CLI + real-time force-directed interactive knowledge graph.

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

### 1-Click MCP Setup for Claude & Cursor (`setup`)
Configure Claude Desktop or Cursor automatically with a single command:
```bash
# Configure Claude Desktop
synapse setup claude

# Configure Cursor IDE
synapse setup cursor
```

### Memory Health, Stats & Garbage Collection (`stats` / `prune`)
```bash
# Inspect health metrics and category counts
synapse stats

# Permanently prune stale/superseded memories
synapse prune
```

### Consolidate Scattered Fragments (`consolidate`)
Synthesize memories within a category into a structured macro-summary (supports local Ollama):
```bash
synapse consolidate architecture
```

### Launch Interactive Web Dashboard & Knowledge Graph (`ui`)
Launch the local dark-mode dashboard with interactive force-directed graph view:
```bash
synapse ui
# Opens http://127.0.0.1:8484
```

### Compress Context within a Token Budget (`compact`)
Extract only the highest-ranking memories compressed to fit within your prompt token budget:
```bash
synapse compact "database guidelines" --tokens 300
```

### Continuous Folder Sync Daemon (`watch`)
Monitor a folder of markdown files or personal notes and auto-sync changes into memory:
```bash
synapse watch ./notes --interval 2.0
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
