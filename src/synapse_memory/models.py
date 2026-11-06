from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    id: str = Field(..., description="Unique identifier for the memory chunk")
    content: str = Field(..., description="The textual content / fact / observation")
    category: str = Field(default="general", description="Category: e.g. preference, code, fact, decision, context")
    workspace: str = Field(default="default", description="Workspace namespace (e.g. default, work, personal)")
    tags: List[str] = Field(default_factory=list, description="Tags or entities linked to this memory")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Priority/importance of this memory (0.0 to 1.0)")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    is_active: bool = Field(default=True, description="Whether this memory is currently active or superseded")
    superseded_by: Optional[str] = Field(default=None, description="ID of memory that replaced this one")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    item: MemoryItem
    semantic_score: float
    recency_score: float
    final_score: float
    rationale: str = ""
