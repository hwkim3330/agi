"""
Working Memory - Current task context (CoALA-based)

Holds:
- Current goal
- Recent actions
- Immediate observations
- Active sub-tasks
"""

from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional


class WorkingMemory:
    """
    Short-term memory for current task context.

    Based on CoALA's working memory concept:
    - Limited capacity (like human working memory)
    - Fast access
    - Automatically expires old items
    """

    def __init__(self, max_items: int = 100):
        self.max_items = max_items
        self._store: OrderedDict[str, Dict] = OrderedDict()
        self._important_keys: set = set()

    def set(self, key: str, value: Any, important: bool = False):
        """Store item in working memory"""
        # Remove oldest if at capacity
        while len(self._store) >= self.max_items:
            oldest_key = next(iter(self._store))
            if oldest_key not in self._important_keys:
                del self._store[oldest_key]
            else:
                # Move to end and try next
                self._store.move_to_end(oldest_key)

        self._store[key] = {
            "value": value,
            "timestamp": datetime.now(),
            "access_count": 0
        }

        if important:
            self._important_keys.add(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get item from working memory"""
        if key in self._store:
            self._store[key]["access_count"] += 1
            self._store.move_to_end(key)  # LRU
            return self._store[key]["value"]
        return default

    def get_all(self) -> Dict[str, Any]:
        """Get all items"""
        return {k: v["value"] for k, v in self._store.items()}

    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get most recent n items"""
        items = list(self._store.items())[-n:]
        return [{"key": k, "value": v["value"], "time": v["timestamp"]} for k, v in items]

    def get_important(self) -> Dict[str, Any]:
        """Get items marked as important"""
        return {k: self._store[k]["value"] for k in self._important_keys if k in self._store}

    def get_context(self, max_length: int = 2000) -> str:
        """Get working memory as context string"""
        parts = []
        for key, data in list(self._store.items())[-20:]:  # Last 20 items
            value_str = str(data["value"])[:200]
            parts.append(f"- {key}: {value_str}")

        context = "\n".join(parts)
        return context[:max_length]

    def contains(self, key: str) -> bool:
        """Check if key exists"""
        return key in self._store

    def remove(self, key: str):
        """Remove item"""
        if key in self._store:
            del self._store[key]
        self._important_keys.discard(key)

    def clear(self):
        """Clear all items"""
        self._store.clear()
        self._important_keys.clear()

    def mark_important(self, key: str):
        """Mark existing item as important"""
        if key in self._store:
            self._important_keys.add(key)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"WorkingMemory({len(self._store)} items, {len(self._important_keys)} important)"
