import json
from typing import Any, Dict
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route
import uvicorn

from synapse_memory.engine import MemoryEngine

engine = MemoryEngine()

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Synapse Memory Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background-color: #0b0f19; color: #f3f4f6; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .glass { background: rgba(17, 24, 39, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .glow-cyan { box-shadow: 0 0 20px rgba(6, 182, 212, 0.25); }
  </style>
</head>
<body class="min-h-screen p-6 md:p-10">
  <div class="max-w-6xl mx-auto space-y-8">
    
    <!-- Header -->
    <header class="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-gray-800 gap-4">
      <div>
        <div class="flex items-center space-x-3">
          <span class="text-3xl">🧠</span>
          <h1 class="text-2xl md:text-3xl font-extrabold tracking-tight bg-gradient-to-r from-cyan-400 to-indigo-400 bg-clip-text text-transparent">Synapse Memory</h1>
          <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800">Local-First</span>
        </div>
        <p class="text-sm text-gray-400 mt-1">Cognitive Long-Term Memory Engine for AI Agents & Personal Workspace</p>
      </div>
      <div class="flex items-center space-x-3">
        <button onclick="refreshAll()" class="px-4 py-2 text-xs font-medium rounded-lg glass hover:bg-gray-800 text-gray-200 transition">↻ Refresh</button>
      </div>
    </header>

    <!-- Main Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
      
      <!-- Left Panel: Input & Query -->
      <div class="lg:col-span-5 space-y-6">
        
        <!-- Store Memory Card -->
        <div class="glass p-6 rounded-2xl space-y-4">
          <h2 class="text-lg font-bold text-gray-100 flex items-center space-x-2">
            <span>✨</span> <span>Remember New Fact</span>
          </h2>
          <div>
            <label class="text-xs font-medium text-gray-400">Content / Instruction / Preference</label>
            <textarea id="memContent" rows="3" placeholder="e.g. Always generate modern TypeScript with Zod validation..." class="w-full mt-1.5 p-3 rounded-xl bg-gray-950/80 border border-gray-800 text-sm focus:border-cyan-500 focus:outline-none transition"></textarea>
          </div>
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="text-xs font-medium text-gray-400">Category</label>
              <input id="memCategory" type="text" placeholder="general (or auto-detect)" class="w-full mt-1 p-2 rounded-lg bg-gray-950/80 border border-gray-800 text-sm focus:border-cyan-500 focus:outline-none">
            </div>
            <div>
              <label class="text-xs font-medium text-gray-400">Importance (0.0 to 1.0)</label>
              <input id="memImportance" type="number" min="0" max="1" step="0.1" value="0.7" class="w-full mt-1 p-2 rounded-lg bg-gray-950/80 border border-gray-800 text-sm focus:border-cyan-500 focus:outline-none">
            </div>
          </div>
          <div>
            <label class="text-xs font-medium text-gray-400">Tags (comma separated)</label>
            <input id="memTags" type="text" placeholder="e.g. code, typescript, zod" class="w-full mt-1 p-2 rounded-lg bg-gray-950/80 border border-gray-800 text-sm focus:border-cyan-500 focus:outline-none">
          </div>
          <button onclick="storeMemory()" id="btnStore" class="w-full py-2.5 px-4 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold text-sm rounded-xl shadow-lg transition glow-cyan">
            Store in Persistent Local Memory
          </button>
        </div>

        <!-- Semantic Recall Card -->
        <div class="glass p-6 rounded-2xl space-y-4">
          <h2 class="text-lg font-bold text-gray-100 flex items-center space-x-2">
            <span>🔍</span> <span>Semantic Cognitive Recall</span>
          </h2>
          <div>
            <input id="searchQuery" type="text" placeholder="Ask anything, e.g. 'what language do we prefer?'" onkeydown="if(event.key==='Enter') doSearch()" class="w-full p-3 rounded-xl bg-gray-950/80 border border-gray-800 text-sm focus:border-cyan-500 focus:outline-none">
          </div>
          <button onclick="doSearch()" class="w-full py-2 px-4 rounded-xl glass hover:bg-gray-800 text-cyan-400 font-medium text-sm transition">
            Run Hybrid Recall
          </button>
        </div>

      </div>

      <!-- Right Panel: Results & Memories List -->
      <div class="lg:col-span-7 space-y-6">
        
        <!-- Search Results Panel (if active) -->
        <div id="searchResultsCard" class="hidden glass p-6 rounded-2xl border-cyan-500/30 space-y-4 glow-cyan">
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-bold text-cyan-300">Cognitive Recall Matches</h2>
            <button onclick="closeSearch()" class="text-xs text-gray-400 hover:text-white">✕ Clear search</button>
          </div>
          <div id="searchResultsList" class="space-y-3"></div>
        </div>

        <!-- Stored Memories Feed -->
        <div class="glass p-6 rounded-2xl space-y-4">
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-bold text-gray-100 flex items-center space-x-2">
              <span>📚</span> <span>All Stored Memories</span>
            </h2>
            <span id="memCount" class="text-xs text-gray-400">Loading...</span>
          </div>
          <div id="memoriesFeed" class="space-y-3 max-h-[600px] overflow-y-auto pr-1"></div>
        </div>

      </div>

    </div>
  </div>

  <script>
    async function fetchMemories() {
      const res = await fetch('/api/memories');
      const data = await res.json();
      const feed = document.getElementById('memoriesFeed');
      const count = document.getElementById('memCount');
      count.textContent = `${data.length} items`;
      
      if (!data.length) {
        feed.innerHTML = '<div class="text-sm text-gray-500 py-8 text-center">No memories stored yet. Add one on the left!</div>';
        return;
      }

      feed.innerHTML = data.map(m => `
        <div class="p-4 rounded-xl bg-gray-900/60 border border-gray-800 hover:border-gray-700 transition space-y-2">
          <div class="flex items-center justify-between text-xs">
            <div class="flex items-center space-x-2">
              <span class="px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 font-semibold uppercase text-[10px]">${m.category}</span>
              <span class="text-gray-400">Score priority: <b>${(m.importance * 100).toFixed(0)}%</b></span>
            </div>
            <div class="flex items-center space-x-3">
              <span class="text-gray-500">${new Date(m.created_at).toLocaleString()}</span>
              <button onclick="deleteMemory('${m.id}')" class="text-red-400 hover:text-red-300">🗑</button>
            </div>
          </div>
          <p class="text-sm text-gray-200 leading-relaxed">${escapeHtml(m.content)}</p>
          ${m.tags.length ? `
            <div class="flex flex-wrap gap-1.5 pt-1">
              ${m.tags.map(t => `<span class="px-2 py-0.5 rounded text-[11px] bg-gray-800 text-gray-300">#${escapeHtml(t)}</span>`).join('')}
            </div>
          ` : ''}
        </div>
      `).join('');
    }

    async function storeMemory() {
      const content = document.getElementById('memContent').value.trim();
      if (!content) return alert('Please enter some memory content');

      const category = document.getElementById('memCategory').value.trim() || 'general';
      const importance = parseFloat(document.getElementById('memImportance').value) || 0.5;
      const tagsRaw = document.getElementById('memTags').value.trim();
      const tags = tagsRaw ? tagsRaw.split(',').map(s => s.trim()).filter(Boolean) : [];

      const btn = document.getElementById('btnStore');
      btn.disabled = true;
      btn.textContent = 'Encoding & Storing...';

      try {
        const res = await fetch('/api/remember', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content, category, importance, tags })
        });
        if (res.ok) {
          document.getElementById('memContent').value = '';
          document.getElementById('memCategory').value = '';
          document.getElementById('memTags').value = '';
          await fetchMemories();
        } else {
          alert('Failed to store memory');
        }
      } finally {
        btn.disabled = false;
        btn.textContent = 'Store in Persistent Local Memory';
      }
    }

    async function doSearch() {
      const query = document.getElementById('searchQuery').value.trim();
      if (!query) return;

      const res = await fetch('/api/recall', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, limit: 5 })
      });
      const data = await res.json();
      
      const card = document.getElementById('searchResultsCard');
      const list = document.getElementById('searchResultsList');
      card.classList.remove('hidden');

      if (!data.length) {
        list.innerHTML = `<div class="text-sm text-yellow-400">No matching memories for "${escapeHtml(query)}"</div>`;
        return;
      }

      list.innerHTML = data.map((r, i) => `
        <div class="p-3.5 rounded-xl bg-cyan-950/30 border border-cyan-800/50 space-y-1.5">
          <div class="flex items-center justify-between text-xs">
            <span class="font-bold text-cyan-300">#${i+1} [${r.item.category.toUpperCase()}] • Match Score: ${(r.final_score * 100).toFixed(0)}%</span>
            <span class="text-gray-400 font-mono text-[11px]">${r.rationale}</span>
          </div>
          <p class="text-sm text-gray-100">${escapeHtml(r.item.content)}</p>
        </div>
      `).join('');
    }

    function closeSearch() {
      document.getElementById('searchResultsCard').classList.add('hidden');
      document.getElementById('searchQuery').value = '';
    }

    async function deleteMemory(id) {
      if (!confirm('Are you sure you want to forget this memory?')) return;
      await fetch(`/api/memories/${id}`, { method: 'DELETE' });
      await fetchMemories();
    }

    function refreshAll() {
      fetchMemories();
    }

    function escapeHtml(str) {
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    fetchMemories();
  </script>
</body>
</html>
"""


async def get_dashboard(request):
    return HTMLResponse(HTML_DASHBOARD)


async def api_list_memories(request):
    limit = int(request.query_params.get("limit", 100))
    category = request.query_params.get("category")
    items = engine.storage.list_memories(limit=limit, category=category)
    return JSONResponse([it.model_dump(mode="json") for it in items])


async def api_remember(request):
    data = await request.json()
    content = data.get("content", "")
    if not content:
        return JSONResponse({"error": "Content is required"}, status_code=400)

    category = data.get("category", "general")
    tags = data.get("tags", [])
    importance = float(data.get("importance", 0.5))

    item = engine.store(
        content=content,
        category=category,
        tags=tags,
        importance=importance,
    )
    return JSONResponse(item.model_dump(mode="json"))


async def api_recall(request):
    data = await request.json()
    query = data.get("query", "")
    if not query:
        return JSONResponse({"error": "Query is required"}, status_code=400)

    category = data.get("category")
    limit = int(data.get("limit", 5))

    results = engine.recall(query=query, category=category, limit=limit)
    return JSONResponse([r.model_dump(mode="json") for r in results])


async def api_delete_memory(request):
    memory_id = request.path_params["id"]
    success = engine.storage.delete_memory(memory_id)
    if success:
        return JSONResponse({"status": "deleted", "id": memory_id})
    return JSONResponse({"error": "Not found"}, status_code=404)


async def api_prompt_context(request):
    data = await request.json()
    query = data.get("query", "")
    limit = int(data.get("limit", 5))
    block = engine.format_context_for_prompt(query, limit=limit)
    return JSONResponse({"context_block": block})


routes = [
    Route("/", endpoint=get_dashboard, methods=["GET"]),
    Route("/api/memories", endpoint=api_list_memories, methods=["GET"]),
    Route("/api/remember", endpoint=api_remember, methods=["POST"]),
    Route("/api/recall", endpoint=api_recall, methods=["POST"]),
    Route("/api/memories/{id}", endpoint=api_delete_memory, methods=["DELETE"]),
    Route("/api/prompt", endpoint=api_prompt_context, methods=["POST"]),
]

app = Starlette(debug=True, routes=routes)


def start_server(host: str = "127.0.0.1", port: int = 8484):
    print(f"🧠 Synapse UI & REST API running at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    start_server()
