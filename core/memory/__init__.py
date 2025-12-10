"""
Agentic Memory System based on:
- CoALA (Cognitive Architectures for Language Agents)
- A-Mem (Agentic Memory for LLM Agents)

Memory Types:
- Working Memory: Current task context
- Long-term Memory: Episodic + Semantic
- Procedural Memory: Skills and patterns
"""

from .working import WorkingMemory
from .long_term import LongTermMemory
from .procedural import ProceduralMemory
from .connection_graph import ConnectionGraph


class MemorySystem:
    """
    Unified Memory System combining all memory types.

    Based on CoALA's modular memory architecture and
    A-Mem's contextual connection graph.
    """

    def __init__(self, data_dir: str = "data/memories"):
        self.data_dir = data_dir
        self.working = WorkingMemory()
        self.long_term = LongTermMemory(data_dir=f"{data_dir}/long_term")
        self.procedural = ProceduralMemory(data_dir=f"{data_dir}/procedural")
        self.connections = ConnectionGraph(data_dir=f"{data_dir}/connections")

    def store(self, content: str, memory_type: str = "long_term", **kwargs):
        """Store content in appropriate memory"""
        if memory_type == "working":
            key = kwargs.get("key", "default")
            self.working.set(key, content)
        elif memory_type == "long_term":
            memory_id = self.long_term.store(content, kwargs.get("metadata", {}))
            # Auto-connect to related memories
            self._auto_connect(memory_id, content)
            return memory_id
        elif memory_type == "procedural":
            return self.procedural.store_pattern(
                goal=kwargs.get("goal", ""),
                actions=kwargs.get("actions", []),
                success=kwargs.get("success", True)
            )

    def retrieve(self, query: str, memory_type: str = "all", k: int = 5):
        """Retrieve relevant memories"""
        results = []

        if memory_type in ["all", "long_term"]:
            results.extend(self.long_term.retrieve(query, k=k))

        if memory_type in ["all", "procedural"]:
            patterns = self.procedural.retrieve(query, k=k)
            results.extend(patterns)

        if memory_type in ["all", "connections"]:
            connected = self.connections.get_related(query, k=k)
            results.extend(connected)

        return results[:k]

    def _auto_connect(self, memory_id: str, content: str):
        """Automatically connect new memory to related existing memories"""
        # Find similar memories
        similar = self.long_term.retrieve(content, k=3)

        for mem in similar:
            if mem.get("id") != memory_id:
                self.connections.connect(
                    source_id=memory_id,
                    target_id=mem.get("id"),
                    relation_type="similar_to",
                    strength=mem.get("similarity", 0.5)
                )

    def consolidate(self):
        """Consolidate memories (compress, extract patterns)"""
        # Move important working memory to long-term
        important_items = self.working.get_important()
        for key, value in important_items.items():
            self.long_term.store(str(value), {"source": "working_memory", "key": key})

        # Extract patterns from long-term to procedural
        self.procedural.extract_patterns(self.long_term)

        # Prune weak connections
        self.connections.prune(threshold=0.1)

    def get_context(self, goal: str, max_tokens: int = 4000) -> str:
        """Get relevant context for a goal"""
        context_parts = []

        # Working memory (most recent)
        working_context = self.working.get_context()
        context_parts.append(f"## Current Context\n{working_context}")

        # Relevant long-term memories
        memories = self.long_term.retrieve(goal, k=3)
        if memories:
            mem_text = "\n".join([f"- {m.get('content', '')[:200]}" for m in memories])
            context_parts.append(f"## Relevant Memories\n{mem_text}")

        # Relevant patterns
        patterns = self.procedural.retrieve(goal, k=2)
        if patterns:
            pattern_text = "\n".join([f"- Goal: {p.get('goal', '')} -> {len(p.get('actions', []))} actions" for p in patterns])
            context_parts.append(f"## Known Patterns\n{pattern_text}")

        return "\n\n".join(context_parts)

    def clear_working(self):
        """Clear working memory"""
        self.working.clear()

    def save(self):
        """Save all memories to disk"""
        self.long_term.save()
        self.procedural.save()
        self.connections.save()

    def load(self):
        """Load all memories from disk"""
        self.long_term.load()
        self.procedural.load()
        self.connections.load()


__all__ = [
    "MemorySystem",
    "WorkingMemory",
    "LongTermMemory",
    "ProceduralMemory",
    "ConnectionGraph"
]
