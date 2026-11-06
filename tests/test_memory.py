import tempfile
import time
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from synapse_memory.engine import MemoryEngine
from synapse_memory.extractor import SmartExtractor
from synapse_memory.watcher import DirectoryWatcher
from synapse_memory.web_app import app


@pytest.fixture
def temp_engine():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memories.db"
        engine = MemoryEngine(db_path=db_path)
        yield engine


def test_store_and_recall_basic(temp_engine):
    item, _ = temp_engine.store(
        content="Antigravity is an advanced agentic coding assistant.",
        category="knowledge",
        tags=["ai", "assistant"],
        importance=0.9,
    )
    assert item.id.startswith("mem_")
    assert item.content == "Antigravity is an advanced agentic coding assistant."

    # Recall
    results = temp_engine.recall("What is Antigravity?")
    assert len(results) > 0
    top = results[0]
    assert top.item.id == item.id
    assert top.final_score > 0.5


def test_auto_enrichment(temp_engine):
    item, _ = temp_engine.store(
        content="Preferisco usare Docker e Postgres nei container #devops"
    )
    assert "docker" in item.tags
    assert "postgres" in item.tags
    assert "devops" in item.tags
    assert item.category in ["preference", "architecture"]


def test_smart_extractor():
    tags = SmartExtractor.extract_tags("Check out #rust with sqlite and kubernetes")
    assert "rust" in tags
    assert "sqlite" in tags
    assert "kubernetes" in tags

    cat = SmartExtractor.infer_category("Non mi piace usare linguaggi a tipizzazione debole")
    assert cat == "preference"


def test_contradiction_and_superseding(temp_engine):
    # Old rule
    old_item, _ = temp_engine.store(
        content="La porta del backend HTTP deve essere 8080 per lo sviluppo",
        category="architecture",
        tags=["backend", "port"],
        importance=0.7,
    )
    assert old_item.is_active is True

    # New updated rule that supersedes old rule
    new_item, superseded = temp_engine.store(
        content="Ora è 9000 invece di 8080 la porta del backend HTTP per lo sviluppo",
        category="architecture",
        tags=["backend", "port"],
        importance=0.8,
    )

    assert len(superseded) >= 1
    assert superseded[0].id == old_item.id

    # The old memory should now be inactive
    reloaded_old = temp_engine.storage.get_memory(old_item.id)
    assert reloaded_old.is_active is False
    assert reloaded_old.superseded_by == new_item.id

    # Querying should only recall the new active memory by default
    results = temp_engine.recall("qual è la porta del backend?")
    assert len(results) > 0
    assert results[0].item.id == new_item.id


def test_compact_context_budget(temp_engine):
    temp_engine.store(content="Rule 1: Use strict TypeScript types", category="code")
    temp_engine.store(content="Rule 2: Every API must have OpenAPI schema", category="code")
    temp_engine.store(content="Rule 3: Use structured logging with Loguru", category="code")

    compact = temp_engine.compact_context("API and TypeScript rules", max_approx_tokens=50)
    assert "Rule 1" in compact or "Rule 2" in compact
    assert "### [Synapse Compact Memory Context]" in compact


def test_directory_watcher(temp_engine):
    with tempfile.TemporaryDirectory() as tmpdir:
        watch_dir = Path(tmpdir)
        note_file = watch_dir / "setup.md"
        note_file.write_text("# Setup Instructions\nRun uv sync to install packages.\n")

        watcher = DirectoryWatcher(temp_engine, dir_path=watch_dir)
        updated = watcher.scan_once()
        assert updated == 1

        # Second scan without changes should detect 0 updates
        assert watcher.scan_once() == 0

        # Modify file
        note_file.write_text("# Setup Instructions\nRun uv sync and uv run pytest.\n")
        assert watcher.scan_once() == 1


def test_forget_memory(temp_engine):
    item, _ = temp_engine.store(content="Secret message to be forgotten")
    assert temp_engine.storage.get_memory(item.id) is not None

    deleted = temp_engine.storage.delete_memory(item.id)
    assert deleted is True
    assert temp_engine.storage.get_memory(item.id) is None


def test_prompt_context_formatting(temp_engine):
    temp_engine.store(content="Always write comments in Italian", category="rule")
    prompt_block = temp_engine.format_context_for_prompt("How should comments be written?")
    assert "[Synapse Long-Term Memory Context]" in prompt_block
    assert "Always write comments in Italian" in prompt_block


def test_web_app_endpoints():
    client = TestClient(app)
    # Check dashboard HTML
    r = client.get("/")
    assert r.status_code == 200
    assert "Synapse Memory" in r.text

    # Check memories API
    r = client.get("/api/memories")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

    # Check graph API
    r = client.get("/api/graph")
    assert r.status_code == 200
    assert "nodes" in r.json()
    assert "links" in r.json()
