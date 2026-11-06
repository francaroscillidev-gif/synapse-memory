from typing import List, Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from synapse_memory.engine import MemoryEngine
from synapse_memory.mcp_server import run_mcp

app = typer.Typer(
    name="synapse",
    help="🧠 Synapse: Local-First Long-Term Memory Engine for AI Agents & Humans",
    add_completion=False,
)
console = Console()
engine = MemoryEngine()


@app.command()
def remember(
    content: str = typer.Argument(..., help="The fact, note, or observation to remember"),
    category: str = typer.Option("general", "--category", "-c", help="Category (e.g. preference, code, fact)"),
    tags: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tags/entities linked to this memory"),
    importance: float = typer.Option(0.5, "--importance", "-i", help="Importance score (0.0 to 1.0)"),
):
    """Save a new memory locally with automatic semantic embedding."""
    with console.status("[bold cyan]Encoding and storing memory...[/bold cyan]"):
        item = engine.store(
            content=content,
            category=category,
            tags=tags or [],
            importance=importance,
        )

    console.print(
        Panel.fit(
            f"[bold green]✔ Memory Preserved![/bold green]\n"
            f"[bold]ID:[/bold] {item.id}\n"
            f"[bold]Category:[/bold] {item.category}\n"
            f"[bold]Importance:[/bold] {item.importance}\n"
            f"[bold]Tags:[/bold] {', '.join(item.tags) if item.tags else 'None'}\n"
            f"[bold]Content:[/bold] {item.content}",
            title="🧠 Synapse Memory",
            border_style="green",
        )
    )


@app.command()
def recall(
    query: str = typer.Argument(..., help="Natural language query to search"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of items to retrieve"),
    min_score: float = typer.Option(0.2, "--min-score", help="Minimum relevance score threshold"),
):
    """Retrieve relevant memories using hybrid ranking (Semantic + Recency + Importance)."""
    with console.status("[bold cyan]Searching synaptic connections...[/bold cyan]"):
        results = engine.recall(query=query, category=category, limit=limit, min_score=min_score)

    if not results:
        console.print(f"[yellow]No memories found for query:[/yellow] '{query}'")
        return

    table = Table(title=f"🧠 Synapse Recall: '{query}'", border_style="cyan")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("ID", style="bold yellow", width=12)
    table.add_column("Category", style="magenta", width=12)
    table.add_column("Score", style="bold green", width=8)
    table.add_column("Content", style="white")
    table.add_column("Rationale", style="dim", width=25)

    for i, res in enumerate(results, 1):
        table.add_row(
            str(i),
            res.item.id,
            res.item.category,
            f"{res.final_score:.2f}",
            res.item.content,
            res.rationale,
        )

    console.print(table)


@app.command()
def prompt(
    query: str = typer.Argument(..., help="Query to extract LLM prompt context for"),
    limit: int = typer.Option(5, "--limit", "-l", help="Max memories in prompt"),
):
    """Format matching memories into a clean Markdown block ready for LLM System Prompts."""
    block = engine.format_context_for_prompt(query, limit=limit)
    if not block:
        console.print("[dim]No relevant context found for this prompt.[/dim]")
    else:
        console.print(Panel(block, title="Prompt Context Block", border_style="blue"))


@app.command(name="list")
def list_all(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of memories to show"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
):
    """List recent memories stored in local SQLite database."""
    items = engine.storage.list_memories(limit=limit, category=category)
    if not items:
        console.print("[yellow]Memory is currently empty. Use `synapse remember` to add some![/yellow]")
        return

    table = Table(title="🧠 Stored Memories", border_style="blue")
    table.add_column("ID", style="yellow", width=12)
    table.add_column("Category", style="magenta", width=12)
    table.add_column("Importance", justify="right", width=10)
    table.add_column("Created", style="dim", width=19)
    table.add_column("Content", style="white")

    for it in items:
        table.add_row(
            it.id,
            it.category,
            f"{it.importance:.2f}",
            it.created_at.strftime("%Y-%m-%d %H:%M"),
            it.content,
        )

    console.print(table)


@app.command()
def forget(
    memory_id: str = typer.Argument(..., help="Memory ID to delete"),
):
    """Delete a memory item permanently."""
    success = engine.storage.delete_memory(memory_id)
    if success:
        console.print(f"[bold green]✔ Memory '{memory_id}' deleted.[/bold green]")
    else:
        console.print(f"[bold red]✘ Memory '{memory_id}' not found.[/bold red]")


@app.command(name="serve-mcp")
def serve_mcp():
    """Start the Model Context Protocol (MCP) server on standard I/O for Claude Desktop / Cursor / Agents."""
    console.print("[bold green]Starting Synapse MCP Server on stdio...[/bold green]", err=True)
    run_mcp()


def main():
    app()


if __name__ == "__main__":
    main()
