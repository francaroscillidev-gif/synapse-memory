import tempfile
import time
from pathlib import Path
import numpy as np
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

    # Check analytics API
    r = client.get("/api/analytics")
    assert r.status_code == 200
    assert "total_memories" in r.json()


def test_security_redaction(temp_engine):
    raw_content = "Connect to API using secret key sk-proj-1234567890abcdef1234567890abcdef12345678 and password=SuperSecretPassword123"
    item, _ = temp_engine.store(content=raw_content, category="credentials")

    # Verify secrets were redacted before storage
    assert "sk-proj-1234567890abcdef" not in item.content
    assert "[REDACTED_API_KEY]" in item.content
    assert "[REDACTED_SECRET]" in item.content
    assert item.metadata.get("security_redacted", 0) >= 1


def test_storage_analytics_and_pruning(temp_engine):
    # Store old rule then supersede it
    temp_engine.store(content="Use JavaScript for scripting", category="rule")
    temp_engine.store(content="Ora è Python invece di JavaScript per lo scripting", category="rule")

    stats = temp_engine.storage.get_analytics()
    assert stats["total_memories"] == 2
    assert stats["active_memories"] == 1
    assert stats["superseded_memories"] == 1

    # Prune
    pruned = temp_engine.storage.prune_superseded()
    assert pruned == 1

    stats_after = temp_engine.storage.get_analytics()
    assert stats_after["total_memories"] == 1
    assert stats_after["superseded_memories"] == 0


def test_client_installer(tmp_path):
    from synapse_memory.installer import ClientInstaller

    # Mock cursor config
    success, config_path = ClientInstaller.setup_cursor(workspace_path=tmp_path)
    assert success is True
    assert Path(config_path).exists()


def test_consolidator(temp_engine):
    from synapse_memory.consolidator import MemoryConsolidator

    temp_engine.store(content="Python is great for AI", category="tech")
    temp_engine.store(content="FastAPI is great for web APIs", category="tech")

    consolidator = MemoryConsolidator(temp_engine)
    summary = consolidator.consolidate_category("tech", store_as_memory=True)
    assert summary is not None
    assert "Python" in summary or "FastAPI" in summary


def test_workspace_isolation(temp_engine):
    # Store item in 'project-alpha'
    temp_engine.store(content="Alpha uses port 3000", workspace="project-alpha")
    # Store item in 'project-beta'
    temp_engine.store(content="Beta uses port 4000", workspace="project-beta")

    # Recall in alpha only
    alpha_results = temp_engine.recall("port number", workspace="project-alpha")
    assert len(alpha_results) == 1
    assert "Alpha uses port 3000" in alpha_results[0].item.content

    # Recall in beta only
    beta_results = temp_engine.recall("port number", workspace="project-beta")
    assert len(beta_results) == 1
    assert "Beta uses port 4000" in beta_results[0].item.content

    # List workspaces
    workspaces = temp_engine.storage.list_workspaces()
    assert "project-alpha" in workspaces
    assert "project-beta" in workspaces


def test_fts_hybrid_search(temp_engine):
    # Test exact keyword hit with specific version tag
    temp_engine.store(content="Dependency version is v9.8.7-beta for auth-service")
    results = temp_engine.recall("v9.8.7-beta")
    assert len(results) > 0
    assert "v9.8.7-beta" in results[0].item.content
    assert "fts=match" in results[0].rationale


def test_export_obsidian_vault(temp_engine, tmp_path):
    from synapse_memory.portability import PortabilityManager

    temp_engine.store(content="Obsidian graph connection test note", category="notes", tags=["graph", "obsidian"])
    manager = PortabilityManager(temp_engine)
    vault_dir = tmp_path / "test_vault"

    count = manager.export_obsidian_vault(vault_dir)
    assert count >= 1
    files = list(vault_dir.glob("*.md"))
    assert len(files) >= 1
    content = files[0].read_text()
    assert "[[" in content
    assert "tags:" in content


def test_shell_history_sync(temp_engine, tmp_path):
    from synapse_memory.shell_sync import ShellHistorySync

    history_file = tmp_path / ".zsh_history"
    history_file.write_text(
        ": 1629839281:0;ls\n"
        ": 1629839282:0;docker run -d -p 8080:80 --name my-web nginx:alpine\n"
        ": 1629839283:0;kubectl get pods -n kube-system -o wide\n"
        ": 1629839284:0;pwd\n"
    )

    syncer = ShellHistorySync(temp_engine)
    synced_count = syncer.sync_history(history_file=history_file, max_entries=10)
    assert synced_count == 2  # Only the docker and kubectl commands

    results = temp_engine.recall("nginx", workspace="terminal")
    assert len(results) > 0
    assert "docker run" in results[0].item.content


def test_api_workspaces_endpoint():
    client = TestClient(app)
    r = client.get("/api/workspaces")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_multi_tier_and_ttl_expiration(temp_engine):
    # Store semantic core rule (permanent)
    sem_item, _ = temp_engine.store(content="Always obey user safety limits", tier="semantic", importance=1.0)
    assert sem_item.tier == "semantic"
    assert sem_item.expires_at is None

    # Store working buffer memory with immediate 1-second TTL
    work_item, _ = temp_engine.store(content="Temporary session note", tier="working", ttl_seconds=1)
    assert work_item.tier == "working"
    assert work_item.expires_at is not None

    # Recalling immediately should find it
    assert len(temp_engine.recall("Temporary session note", tier="working")) > 0

    # Wait for TTL to expire
    time.sleep(1.2)
    expired_count = temp_engine.storage.prune_expired()
    assert expired_count >= 1

    # Now it should be gone from storage
    assert temp_engine.storage.get_memory(work_item.id) is None


def test_aes_gcm_encryption_at_rest(tmp_path):
    from synapse_memory.crypto import CryptoManager
    from synapse_memory.storage import StorageManager

    key = "super_secure_passphrase_for_testing"
    crypto = CryptoManager(passphrase=key)
    ciphertext = crypto.encrypt("Highly sensitive medical or financial data")
    assert ciphertext.startswith("enc:v1:")
    assert crypto.decrypt(ciphertext) == "Highly sensitive medical or financial data"

    # Test encrypted storage
    db_file = tmp_path / "encrypted.db"
    enc_storage = StorageManager(db_path=db_file, encryption_key=key)
    test_vec = np.zeros(384, dtype=np.float32)

    from synapse_memory.models import MemoryItem
    item = MemoryItem(id="enc_1", content="Top Secret Mission Alpha")
    enc_storage.save_memory(item, test_vec)

    # Read raw SQLite content directly from disk
    import sqlite3
    raw_conn = sqlite3.connect(db_file)
    row = raw_conn.execute("SELECT content FROM memories WHERE id = 'enc_1'").fetchone()
    raw_stored_text = row[0]
    # The raw disk file must contain the ciphertext prefix, NOT the plaintext
    assert raw_stored_text.startswith("enc:v1:")
    assert "Top Secret Mission Alpha" not in raw_stored_text

    # Read through encrypted storage -> transparently decrypted
    reloaded = enc_storage.get_memory("enc_1")
    assert reloaded.content == "Top Secret Mission Alpha"


def test_vector_quantization_int8():
    from synapse_memory.quantizer import VectorQuantizer

    # Create dummy 384-dimensional unit vector
    vec = np.random.randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec)

    quant_bytes, scale, min_val = VectorQuantizer.quantize_to_int8(vec)
    assert len(quant_bytes) == 384  # 1 byte per dimension (vs 1536 bytes in float32)

    dequant = VectorQuantizer.dequantize_from_int8(quant_bytes, scale, min_val)
    cosine_sim = VectorQuantizer.fast_cosine_similarity(vec, dequant)
    # Quantization error should be minimal (similarity > 0.99)
    assert cosine_sim > 0.99
