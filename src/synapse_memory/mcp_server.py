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
    tags: Optional[List[str]] = None,
    importance: float = 0.5,
) -> str:
    """Store a new fact, user preference, instruction, or knowledge in persistent local memory."""
    item = engine.store(
        content=content,
        category=category,
        tags=tags or [],
        importance=importance,
    )
    return f"Successfully stored memory [{item.id}] in category '{item.category}':\n\"{item.content}\""


@mcp_app.tool()
def recall(
    query: str,
    category: Optional[str] = None,
    limit: int = 5,
) -> str:
    """Query local memory using semantic hybrid search, recency, and importance ranking."""
    results = engine.recall(query=query, category=category, limit=limit)
    if not results:
        return f"No relevant memories found for query: '{query}'"

    formatted = []
    for i, res in enumerate(results, 1):
        formatted.append(
            f"{i}. [ID: {res.item.id}] [{res.item.category.upper()}] (Relevance Score: {res.final_score:.2f})\n"
            f"   Content: {res.item.content}\n"
            f"   Tags: {', '.join(res.item.tags) if res.item.tags else 'None'}\n"
            f"   Factors: {res.rationale}"
        )

    return "\n\n".join(formatted)


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
