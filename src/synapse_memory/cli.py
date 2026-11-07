import time
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
    workspace: str = typer.Option("default", "--workspace", "-w", help="Workspace namespace (e.g. work, personal, mobile)"),
    tier: str = typer.Option("episodic", "--tier", help="Cognitive tier: 'working' (short-term buffer), 'episodic' (events), 'semantic' (core beliefs)"),
    ttl: Optional[int] = typer.Option(None, "--ttl", help="Time-to-live in seconds (useful for working memory)"),
    tags: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tags/entities linked to this memory"),
    importance: float = typer.Option(0.5, "--importance", "-i", help="Importance score (0.0 to 1.0)"),
):
    """Save a new memory locally with automatic semantic embedding."""
    with console.status("[bold cyan]Encoding and storing memory...[/bold cyan]"):
        item, superseded = engine.store(
            content=content,
            category=category,
            workspace=workspace,
            tier=tier,
            ttl_seconds=ttl,
            tags=tags or [],
            importance=importance,
        )

    msg = (
        f"[bold green]✔ Memory Preserved![/bold green]\n"
        f"[bold]ID:[/bold] {item.id}\n"
        f"[bold]Tier:[/bold] {item.tier}\n"
        f"[bold]Workspace:[/bold] {item.workspace}\n"
        f"[bold]Category:[/bold] {item.category}\n"
        f"[bold]Importance:[/bold] {item.importance}\n"
        f"[bold]Expires:[/bold] {item.expires_at.strftime('%Y-%m-%d %H:%M:%S') if item.expires_at else 'Never'}\n"
        f"[bold]Tags:[/bold] {', '.join(item.tags) if item.tags else 'None'}\n"
        f"[bold]Content:[/bold] {item.content}"
    )
    if superseded:
        old_ids = ", ".join([f"[yellow]{s.id}[/yellow]" for s in superseded])
        msg += f"\n\n[bold yellow]⚡ Memory Evolution:[/bold yellow] This memory superseded/updated older conflicting memory: {old_ids}"

    console.print(
        Panel.fit(
            msg,
            title="🧠 Synapse Memory",
            border_style="green",
        )
    )


@app.command()
def recall(
    query: str = typer.Argument(..., help="Natural language query to search"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Filter by workspace namespace"),
    tier: Optional[str] = typer.Option(None, "--tier", help="Filter by memory tier ('working', 'episodic', 'semantic')"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of items to retrieve"),
    min_score: float = typer.Option(0.2, "--min-score", help="Minimum relevance score threshold"),
):
    """Retrieve relevant memories using hybrid ranking (Semantic + Recency + Importance)."""
    with console.status("[bold cyan]Searching synaptic connections...[/bold cyan]"):
        results = engine.recall(query=query, category=category, workspace=workspace, tier=tier, limit=limit, min_score=min_score)

    if not results:
        console.print(f"[yellow]No memories found for query:[/yellow] '{query}'")
        return

    table = Table(title=f"🧠 Synapse Recall: '{query}'" + (f" (Workspace: {workspace})" if workspace else ""), border_style="cyan")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("ID", style="bold yellow", width=12)
    table.add_column("Tier", style="cyan", width=9)
    table.add_column("Workspace", style="blue", width=10)
    table.add_column("Category", style="magenta", width=12)
    table.add_column("Score", style="bold green", width=8)
    table.add_column("Content", style="white")
    table.add_column("Rationale", style="dim", width=25)

    for i, res in enumerate(results, 1):
        table.add_row(
            str(i),
            res.item.id,
            res.item.tier,
            res.item.workspace,
            res.item.category,
            f"{res.final_score:.2f}",
            res.item.content,
            res.rationale,
        )

    console.print(table)


@app.command()
def prompt(
    query: str = typer.Argument(..., help="Query to extract LLM prompt context for"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace filter"),
    limit: int = typer.Option(5, "--limit", "-l", help="Max memories in prompt"),
):
    """Format matching memories into a clean Markdown block ready for LLM System Prompts."""
    block = engine.format_context_for_prompt(query, workspace=workspace, limit=limit)
    if not block:
        console.print("[dim]No relevant context found for this prompt.[/dim]")
    else:
        console.print(Panel(block, title="Prompt Context Block", border_style="blue"))


@app.command(name="list")
def list_all(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of memories to show"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Filter by workspace"),
    tier: Optional[str] = typer.Option(None, "--tier", help="Filter by memory tier ('working', 'episodic', 'semantic')"),
):
    """List recent memories stored in local SQLite database."""
    items = engine.storage.list_memories(limit=limit, category=category, workspace=workspace, tier=tier)
    if not items:
        console.print("[yellow]Memory is currently empty. Use `synapse remember` to add some![/yellow]")
        return

    table = Table(title="🧠 Stored Memories", border_style="blue")
    table.add_column("ID", style="yellow", width=12)
    table.add_column("Tier", style="cyan", width=9)
    table.add_column("Workspace", style="blue", width=10)
    table.add_column("Category", style="magenta", width=12)
    table.add_column("Importance", justify="right", width=10)
    table.add_column("Created", style="dim", width=19)
    table.add_column("Content", style="white")

    for it in items:
        table.add_row(
            it.id,
            it.tier,
            it.workspace,
            it.category,
            f"{it.importance:.2f}",
            it.created_at.strftime("%Y-%m-%d %H:%M"),
            it.content,
        )

    console.print(table)


@app.command(name="workspaces")
def list_workspaces_cli():
    """List all active workspace namespaces in memory."""
    ws_list = engine.storage.list_workspaces()
    if not ws_list:
        console.print("[dim]No workspaces found (default is empty).[/dim]")
        return

    table = Table(title="📁 Synapse Workspaces", border_style="cyan")
    table.add_column("Workspace Name", style="bold green")
    for ws in ws_list:
        table.add_row(ws)
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


@app.command(name="ui")
def launch_ui(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind to"),
    port: int = typer.Option(8484, "--port", "-p", help="Port for the web dashboard"),
):
    """Launch the interactive local Web Dashboard & REST API."""
    from synapse_memory.web_app import start_server
    console.print(f"[bold cyan]🚀 Opening Synapse Dashboard at http://{host}:{port}[/bold cyan]")
    start_server(host=host, port=port)


@app.command(name="ingest")
def ingest(
    path: str = typer.Argument(..., help="Path to file or directory to ingest into memory"),
    glob_pattern: str = typer.Option("**/*.md", "--glob", "-g", help="Glob pattern for directories"),
):
    """Ingest notes, Markdown files, or directories into long-term memory."""
    from pathlib import Path
    from synapse_memory.ingestor import FileIngestor

    target_path = Path(path).resolve()
    ingestor = FileIngestor(engine)

    if not target_path.exists():
        console.print(f"[bold red]✘ Path does not exist:[/bold red] {target_path}")
        raise typer.Exit(1)

    if target_path.is_file():
        with console.status(f"[bold cyan]Ingesting file {target_path.name}...[/bold cyan]"):
            count = ingestor.ingest_file(target_path)
        console.print(f"[bold green]✔ Ingested {count} memory chunks from '{target_path.name}'[/bold green]")
    elif target_path.is_dir():
        with console.status(f"[bold cyan]Ingesting directory {target_path.name}...[/bold cyan]"):
            files, chunks = ingestor.ingest_directory(target_path, glob_pattern=glob_pattern)
        console.print(f"[bold green]✔ Ingested {chunks} chunks across {files} files[/bold green]")


@app.command(name="export")
def export_memories(
    output_path: str = typer.Option("synapse_backup.json", "--out", "-o", help="Output file path (.json or .md)"),
    obsidian_vault: Optional[str] = typer.Option(None, "--obsidian", help="Export as Obsidian vault folder with wikilinks"),
):
    """Export all memories to JSON backup, single Markdown file, or interconnected Obsidian Vault."""
    from pathlib import Path
    from synapse_memory.portability import PortabilityManager

    manager = PortabilityManager(engine)

    if obsidian_vault:
        vault_path = Path(obsidian_vault).resolve()
        with console.status(f"[bold cyan]Exporting Obsidian Vault to {vault_path}...[/bold cyan]"):
            count = manager.export_obsidian_vault(vault_path)
        console.print(f"[bold green]✔ Exported {count} notes with bidirectional graph links to Obsidian Vault:[/bold green] {vault_path}")
        return

    out = Path(output_path).resolve()
    if out.suffix == ".md":
        count = manager.export_markdown(out)
        console.print(f"[bold green]✔ Exported {count} memories to Markdown:[/bold green] {out}")
    else:
        count = manager.export_json(out)
        console.print(f"[bold green]✔ Exported {count} memories to JSON backup:[/bold green] {out}")


@app.command(name="import")
def import_memories(
    input_path: str = typer.Argument(..., help="Path to JSON backup file"),
):
    """Import memories from a previously exported JSON backup file."""
    from pathlib import Path
    from synapse_memory.portability import PortabilityManager

    inp = Path(input_path).resolve()
    manager = PortabilityManager(engine)

    if not inp.exists():
        console.print(f"[bold red]✘ Backup file not found:[/bold red] {inp}")
        raise typer.Exit(1)

    with console.status("[bold cyan]Restoring memories from backup...[/bold cyan]"):
        count = manager.import_json(inp)
    console.print(f"[bold green]✔ Successfully restored {count} memories from '{inp.name}'[/bold green]")


@app.command(name="compact")
def compact_context(
    query: str = typer.Argument(..., help="Query to generate compressed context for"),
    tokens: int = typer.Option(400, "--tokens", "-t", help="Target token budget"),
):
    """Generate compressed memory context strictly adhering to a token budget."""
    block = engine.compact_context(query=query, max_approx_tokens=tokens)
    if not block:
        console.print("[dim]No matching memories to compact.[/dim]")
    else:
        console.print(Panel(block, title=f"⚡ Compact Memory Context (~{tokens} Tokens)", border_style="cyan"))


@app.command(name="watch")
def watch_directory(
    path: str = typer.Argument(..., help="Directory path to watch"),
    interval: float = typer.Option(2.0, "--interval", "-i", help="Scan interval in seconds"),
    glob_pattern: str = typer.Option("**/*.md", "--glob", "-g", help="Glob pattern for files"),
):
    """Continuously watch a directory and auto-sync modified files into memory."""
    from pathlib import Path
    from synapse_memory.watcher import DirectoryWatcher

    dir_path = Path(path).resolve()
    if not dir_path.is_dir():
        console.print(f"[bold red]✘ Not a directory:[/bold red] {dir_path}")
        raise typer.Exit(1)

    watcher = DirectoryWatcher(engine, dir_path=dir_path, glob_pattern=glob_pattern)
    console.print(f"[bold cyan]👀 Starting Synapse Watcher on '{dir_path}'...[/bold cyan]")
    
    def on_change(count: int):
        console.print(f"[bold green]✔ Auto-synced {count} updated file(s) into memory at {time.strftime('%H:%M:%S')}[/bold green]")

    watcher.start_watching(interval_seconds=interval, on_change=on_change)


@app.command(name="setup")
def setup_client(
    target: str = typer.Argument(..., help="Client to configure: 'claude' or 'cursor'"),
):
    """1-Click MCP Setup for Claude Desktop or Cursor."""
    from synapse_memory.installer import ClientInstaller

    target_clean = target.strip().lower()
    if target_clean in ["claude", "claude-desktop"]:
        success, path_or_err = ClientInstaller.setup_claude()
        if success:
            console.print(f"[bold green]✔ Configured Claude Desktop MCP at:[/bold green] {path_or_err}")
            console.print("[dim]Restart Claude Desktop to activate Synapse Memory tools.[/dim]")
        else:
            console.print(f"[bold red]✘ Failed to configure Claude Desktop:[/bold red] {path_or_err}")
    elif target_clean == "cursor":
        success, path_or_err = ClientInstaller.setup_cursor()
        if success:
            console.print(f"[bold green]✔ Configured Cursor MCP at:[/bold green] {path_or_err}")
            console.print("[dim]Restart Cursor to activate Synapse Memory tools.[/dim]")
        else:
            console.print(f"[bold red]✘ Failed to configure Cursor:[/bold red] {path_or_err}")
    else:
        console.print(f"[bold red]Unknown target '{target}'. Use 'claude' or 'cursor'.[/bold red]")


@app.command(name="stats")
def show_stats():
    """Display analytics and health metrics for your memory store."""
    stats = engine.storage.get_analytics()
    table = Table(title="🧠 Synapse Memory Stats", border_style="cyan")
    table.add_column("Metric", style="bold yellow")
    table.add_column("Value", style="green")

    table.add_row("Total Memories", str(stats["total_memories"]))
    table.add_row("Active Memories", str(stats["active_memories"]))
    table.add_row("Superseded / Stale", str(stats["superseded_memories"]))

    console.print(table)

    if stats["categories"]:
        cat_table = Table(title="📁 Categories Breakdown", border_style="magenta")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", style="white")
        for cat, count in stats["categories"].items():
            cat_table.add_row(cat, str(count))
        console.print(cat_table)


@app.command(name="prune")
def prune_superseded():
    """Permanently delete inactive / superseded memories (Garbage Collection)."""
    count = engine.storage.prune_superseded()
    if count > 0:
        console.print(f"[bold green]✔ Pruned {count} stale/superseded memories from database.[/bold green]")
    else:
        console.print("[dim]Memory store is clean: no stale memories found.[/dim]")


@app.command(name="consolidate")
def consolidate(
    category: str = typer.Argument(..., help="Category to consolidate into structured summary"),
    model: str = typer.Option("qwen2.5:1.5b", "--model", "-m", help="Local Ollama model if available"),
):
    """Synthesize scattered memory fragments within a category into a structured summary."""
    from synapse_memory.consolidator import MemoryConsolidator

    consolidator = MemoryConsolidator(engine)
    with console.status(f"[bold cyan]Consolidating memories for '{category}'...[/bold cyan]"):
        summary = consolidator.consolidate_category(category=category, model=model)

    if not summary:
        console.print(f"[yellow]Not enough memories in '{category}' to consolidate (minimum 2 needed).[/yellow]")
    else:
        console.print(Panel(summary, title=f"🧠 Consolidated Summary: {category.upper()}", border_style="green"))


@app.command(name="shell-sync")
def shell_sync(
    max_commands: int = typer.Option(40, "--max", "-m", help="Maximum commands to index"),
    history_file: Optional[str] = typer.Option(None, "--file", "-f", help="Custom history file path"),
):
    """Index high-value developer terminal commands into the 'terminal' workspace."""
    from pathlib import Path
    from synapse_memory.shell_sync import ShellHistorySync

    syncer = ShellHistorySync(engine)
    h_path = Path(history_file).resolve() if history_file else None

    with console.status("[bold cyan]Scanning shell history for high-value commands...[/bold cyan]"):
        count = syncer.sync_history(history_file=h_path, max_entries=max_commands)

    if count > 0:
        console.print(f"[bold green]✔ Synchronized {count} shell commands into workspace 'terminal'![/bold green]")
        console.print("[dim]Recall them anytime with: `synapse recall 'docker command' --workspace terminal`[/dim]")
    else:
        console.print("[yellow]No new or substantial shell commands found to index.[/yellow]")


remote_app = typer.Typer(help="🌐 Decentralized cross-machine backup and sync via private Git remotes")
app.add_typer(remote_app, name="remote")


@remote_app.command(name="init")
def remote_init(
    git_url: str = typer.Argument(..., help="Remote git repository URL (e.g. git@github.com:user/my-vault.git)"),
):
    """Initialize a remote vault synchronization folder."""
    from synapse_memory.remote_sync import RemoteSyncHub

    hub = RemoteSyncHub(engine)
    success, msg = hub.init_vault(git_url)
    if success:
        console.print(f"[bold green]✔ {msg}[/bold green]")
    else:
        console.print(f"[bold red]✘ {msg}[/bold red]")


@remote_app.command(name="push")
def remote_push(
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Custom commit message"),
):
    """Export current memory state and push to remote sync repository."""
    from synapse_memory.remote_sync import RemoteSyncHub

    hub = RemoteSyncHub(engine)
    with console.status("[bold cyan]Syncing and pushing memories to remote vault...[/bold cyan]"):
        success, msg = hub.push_sync(commit_message=message)
    if success:
        console.print(f"[bold green]✔ {msg}[/bold green]")
    else:
        console.print(f"[bold red]✘ {msg}[/bold red]")


@remote_app.command(name="pull")
def remote_pull():
    """Pull latest memory state from remote repository and merge into local database."""
    from synapse_memory.remote_sync import RemoteSyncHub

    hub = RemoteSyncHub(engine)
    with console.status("[bold cyan]Pulling memories from remote vault...[/bold cyan]"):
        success, msg = hub.pull_sync()
    if success:
        console.print(f"[bold green]✔ {msg}[/bold green]")
    else:
        console.print(f"[bold red]✘ {msg}[/bold red]")


def main():
    app()


if __name__ == "__main__":
    main()
