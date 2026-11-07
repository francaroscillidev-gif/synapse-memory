import asyncio
from typing import List, Optional
from mcp.server.mcpserver import MCPServer

from synapse_memory.engine import MemoryEngine

mcp_app = MCPServer(name="synapse-memory")
engine = MemoryEngine()


@mcp_app.tool()
def remember(
    content: str,
    category: str = "general",
    workspace: str = "default",
    tier: str = "episodic",
    tags: Optional[List[str]] = None,
    importance: float = 0.5,
) -> str:
    """Store a new fact, user preference, instruction, or knowledge in persistent local memory.
    
    Args:
        content: The text content of the memory.
        category: Category (e.g. 'preference', 'project', 'tool', 'general').
        workspace: Workspace namespace (e.g. 'default', 'frontend', 'security', 'work').
        tier: Memory tier ('working', 'episodic', or 'semantic').
        tags: Optional list of keyword tags.
        importance: Weight between 0.0 and 1.0 (defaults to 0.5).
    """
    item, superseded = engine.store(
        content=content,
        category=category,
        workspace=workspace,
        tier=tier,
        tags=tags or [],
        importance=importance,
    )
    msg = f"Successfully stored memory [{item.id}] in workspace '{item.workspace}', category '{item.category}' (tier: {item.tier}):\n\"{item.content}\""
    if superseded:
        old_ids = ", ".join([f"[{s.id}]" for s in superseded])
        msg += f"\n\nNote: This memory superseded/updated older conflicting memory: {old_ids}"
    return msg


@mcp_app.tool()
def compact_prompt(
    query: str,
    workspace: Optional[str] = None,
    max_tokens: int = 400,
) -> str:
    """Retrieve long-term memory context formatted and compressed to fit within a token budget."""
    return engine.compact_context(query=query, workspace=workspace, max_approx_tokens=max_tokens)


@mcp_app.tool()
def recall(
    query: str,
    category: Optional[str] = None,
    workspace: Optional[str] = None,
    tier: Optional[str] = None,
    limit: int = 5,
) -> str:
    """Query local memory using semantic hybrid search, recency, and importance ranking.
    
    Args:
        query: Search term or question.
        category: Filter by specific category (optional).
        workspace: Filter by workspace (optional).
        tier: Filter by tier ('working', 'episodic', 'semantic') (optional).
        limit: Maximum results to return (default: 5).
    """
    results = engine.recall(
        query=query,
        category=category,
        workspace=workspace,
        tier=tier,
        limit=limit,
    )
    if not results:
        target = f" in workspace '{workspace}'" if workspace else ""
        return f"No relevant memories found for query: '{query}'{target}"

    formatted = []
    for i, res in enumerate(results, 1):
        formatted.append(
            f"{i}. [ID: {res.item.id}] [{res.item.workspace.upper()}:{res.item.category.upper()}] (Score: {res.final_score:.2f}, Tier: {res.item.tier})\n"
            f"   Content: {res.item.content}\n"
            f"   Tags: {', '.join(res.item.tags) if res.item.tags else 'None'}\n"
            f"   Factors: {res.rationale}"
        )

    return "\n\n".join(formatted)


@mcp_app.tool()
def list_workspaces() -> str:
    """List all available workspaces and the count of memories stored in each."""
    stats = engine.storage.get_workspace_stats()
    if not stats:
        return "No memories stored in any workspace yet."
    lines = ["Available Workspaces:"]
    for s in stats:
        lines.append(f"- {s['workspace']}: {s['count']} memories (latest activity: {s['last_activity'] or 'N/A'})")
    return "\n".join(lines)


@mcp_app.tool()
def forget(memory_id: str) -> str:
    """Permanently delete a memory item by its ID."""
    success = engine.storage.delete_memory(memory_id)
    if success:
        return f"Memory '{memory_id}' deleted successfully."
    else:
        return f"Memory '{memory_id}' not found."


def run_mcp():
    """Run the MCP server over stdio."""
    asyncio.run(mcp_app.run_stdio_async())


if __name__ == "__main__":
    run_mcp()
