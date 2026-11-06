import tempfile
from pathlib import Path
import pytest
from synapse_memory.engine import MemoryEngine


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
