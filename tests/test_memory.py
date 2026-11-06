import tempfile
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from synapse_memory.engine import MemoryEngine
from synapse_memory.extractor import SmartExtractor
from synapse_memory.web_app import app


@pytest.fixture
def temp_engine():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memories.db"
        engine = MemoryEngine(db_path=db_path)
        yield engine


def test_store_and_recall_basic(temp_engine):
    item = temp_engine.store(
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
    item = temp_engine.store(
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


def test_forget_memory(temp_engine):
    item = temp_engine.store(content="Secret message to be forgotten")
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


def test_file_ingestion(temp_engine):
    from synapse_memory.ingestor import FileIngestor

    ingestor = FileIngestor(temp_engine)
    sample_doc = """# Architecture Guidelines
We use FastAPI for internal microservices with pydantic schemas.

## Database Layer
PostgreSQL is strictly required for ACID transactions and persistent state.
"""
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as f:
        f.write(sample_doc)
        f.flush()
        f_path = Path(f.name)

    try:
        count = ingestor.ingest_file(f_path)
        assert count >= 2
        results = temp_engine.recall("What database is required for ACID transactions?")
        assert len(results) > 0
        assert "PostgreSQL" in results[0].item.content
    finally:
        f_path.unlink()


def test_export_and_import(temp_engine):
    from synapse_memory.portability import PortabilityManager

    temp_engine.store(content="Critical secret rule for backup testing", category="backup")
    manager = PortabilityManager(temp_engine)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        backup_path = Path(f.name)

    try:
        exported_count = manager.export_json(backup_path)
        assert exported_count >= 1

        # Re-importing should work without errors
        imported_count = manager.import_json(backup_path)
        assert imported_count == exported_count
    finally:
        if backup_path.exists():
            backup_path.unlink()
