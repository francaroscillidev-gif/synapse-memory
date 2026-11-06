import re
from typing import List, Optional, Tuple
import numpy as np

from synapse_memory.models import MemoryItem
from synapse_memory.storage import StorageManager

UPDATE_PHRASES = [
    r"\bnon\s+più\b",
    r"\bora\s+è\b",
    r"\binvece\s+di\b",
    r"\bsostituit[oa]\b",
    r"\baggiornat[oa]\b",
    r"\bpassat[oa]\s+a\b",
    r"\babbiamo\s+cambiato\b",
    r"\bno\s+longer\b",
    r"\binstead\s+of\b",
    r"\breplaced\b",
    r"\bnow\s+use\b",
    r"\bswitched\s+to\b",
    r"\bdeprecated\b",
]

UPDATE_REGEX = re.compile("|".join(UPDATE_PHRASES), re.IGNORECASE)


class MemoryEvolver:
    """Detects when new facts supersede or contradict older facts, maintaining consistency."""

    def __init__(self, storage: StorageManager):
        self.storage = storage

    def check_and_evolve(
        self, new_item: MemoryItem, new_vector: np.ndarray, threshold: float = 0.72
    ) -> List[MemoryItem]:
        """
        Check if new_item supersedes any existing active memory.
        Returns a list of superseded memories.
        """
        active_entries = self.storage.get_all_vectors(category=new_item.category, active_only=True)
        superseded: List[MemoryItem] = []

        has_update_signal = bool(UPDATE_REGEX.search(new_item.content))

        for old_item, old_vector in active_entries:
            if old_item.id == new_item.id:
                continue

            # Calculate semantic similarity
            cos_sim = float(np.dot(new_vector, old_vector))

            # Shared tags boost probability of addressing the exact same topic
            shared_tags = set(new_item.tags).intersection(set(old_item.tags))

            # Trigger condition:
            # High semantic overlap AND (explicit update phrasing OR shared specific entity tag with high overlap)
            is_conflict = (
                (has_update_signal and cos_sim >= threshold) or
                (len(shared_tags) >= 2 and cos_sim >= 0.85) or
                (cos_sim >= 0.92)  # Near-duplicate or direct update
            )

            if is_conflict:
                self.storage.supersede_memory(old_memory_id=old_item.id, new_memory_id=new_item.id)
                superseded.append(old_item)

        return superseded
