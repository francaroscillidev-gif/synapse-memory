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
