"""
Connection Graph - A-Mem inspired contextual memory links

Based on A-Mem (arXiv:2502.12110):
- Autonomous generation of contextual descriptions
- Intelligent connection establishment
- Memory relationship graph
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set


@dataclass
class Connection:
    """A connection between two memories"""
    source_id: str
    target_id: str
    relation_type: str  # similar_to, leads_to, contradicts, supports, etc.
    strength: float  # 0.0 to 1.0
    created_at: str
    context: Optional[str] = None


class ConnectionGraph:
    """
    Graph of connections between memories.

    Based on A-Mem's approach to establishing
    contextual links between memories.
    """

    RELATION_TYPES = [
        "similar_to",      # Semantically similar
        "leads_to",        # Causal relationship
        "contradicts",     # Contradictory information
        "supports",        # Supporting evidence
        "part_of",         # Hierarchical relationship
        "related_to",      # General relationship
        "learned_from",    # Learning source
        "used_with",       # Co-occurrence
    ]

    def __init__(self, data_dir: str = "data/memories/connections"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self._connections: List[Connection] = []
        self._adjacency: Dict[str, Set[str]] = {}  # memory_id -> connected_ids

        self.load()

    def connect(self, source_id: str, target_id: str, relation_type: str,
                strength: float = 0.5, context: Optional[str] = None) -> Connection:
        """Create a connection between two memories"""

        if relation_type not in self.RELATION_TYPES:
            relation_type = "related_to"

        # Check for existing connection
        existing = self._find_connection(source_id, target_id)
        if existing:
            # Strengthen existing connection
            existing.strength = min(1.0, existing.strength + 0.1)
            return existing

        connection = Connection(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=max(0.0, min(1.0, strength)),
            created_at=datetime.now().isoformat(),
            context=context
        )

        self._connections.append(connection)

        # Update adjacency
        if source_id not in self._adjacency:
            self._adjacency[source_id] = set()
        self._adjacency[source_id].add(target_id)

        if target_id not in self._adjacency:
            self._adjacency[target_id] = set()
        self._adjacency[target_id].add(source_id)

        return connection

    def get_connections(self, memory_id: str) -> List[Connection]:
        """Get all connections for a memory"""
        return [c for c in self._connections
                if c.source_id == memory_id or c.target_id == memory_id]

    def get_related(self, memory_id: str, k: int = 5,
                    relation_type: Optional[str] = None) -> List[Dict]:
        """Get related memory IDs"""

        connections = self.get_connections(memory_id)

        if relation_type:
            connections = [c for c in connections if c.relation_type == relation_type]

        # Sort by strength
        connections.sort(key=lambda c: c.strength, reverse=True)

        results = []
        for conn in connections[:k]:
            related_id = conn.target_id if conn.source_id == memory_id else conn.source_id
            results.append({
                "id": related_id,
                "relation": conn.relation_type,
                "strength": conn.strength,
                "context": conn.context
            })

        return results

    def get_path(self, source_id: str, target_id: str, max_depth: int = 3) -> List[str]:
        """Find path between two memories (BFS)"""

        if source_id == target_id:
            return [source_id]

        visited = {source_id}
        queue = [(source_id, [source_id])]

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            for neighbor in self._adjacency.get(current, set()):
                if neighbor == target_id:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []  # No path found

    def weaken(self, source_id: str, target_id: str, amount: float = 0.1):
        """Weaken a connection"""
        conn = self._find_connection(source_id, target_id)
        if conn:
            conn.strength = max(0.0, conn.strength - amount)

    def strengthen(self, source_id: str, target_id: str, amount: float = 0.1):
        """Strengthen a connection"""
        conn = self._find_connection(source_id, target_id)
        if conn:
            conn.strength = min(1.0, conn.strength + amount)

    def prune(self, threshold: float = 0.1):
        """Remove weak connections"""
        self._connections = [c for c in self._connections if c.strength >= threshold]

        # Rebuild adjacency
        self._adjacency = {}
        for conn in self._connections:
            if conn.source_id not in self._adjacency:
                self._adjacency[conn.source_id] = set()
            self._adjacency[conn.source_id].add(conn.target_id)

            if conn.target_id not in self._adjacency:
                self._adjacency[conn.target_id] = set()
            self._adjacency[conn.target_id].add(conn.source_id)

    def _find_connection(self, source_id: str, target_id: str) -> Optional[Connection]:
        """Find existing connection between two memories"""
        for conn in self._connections:
            if (conn.source_id == source_id and conn.target_id == target_id) or \
               (conn.source_id == target_id and conn.target_id == source_id):
                return conn
        return None

    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        return {
            "total_connections": len(self._connections),
            "total_nodes": len(self._adjacency),
            "avg_connections_per_node": len(self._connections) * 2 / max(1, len(self._adjacency)),
            "relation_types": {
                rt: sum(1 for c in self._connections if c.relation_type == rt)
                for rt in self.RELATION_TYPES
            }
        }

    def save(self):
        """Save to disk"""
        data = {
            "connections": [
                {
                    "source_id": c.source_id,
                    "target_id": c.target_id,
                    "relation_type": c.relation_type,
                    "strength": c.strength,
                    "created_at": c.created_at,
                    "context": c.context
                }
                for c in self._connections
            ]
        }

        filepath = os.path.join(self.data_dir, "graph.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load from disk"""
        filepath = os.path.join(self.data_dir, "graph.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)

            self._connections = [
                Connection(**c) for c in data.get("connections", [])
            ]

            # Rebuild adjacency
            for conn in self._connections:
                if conn.source_id not in self._adjacency:
                    self._adjacency[conn.source_id] = set()
                self._adjacency[conn.source_id].add(conn.target_id)

                if conn.target_id not in self._adjacency:
                    self._adjacency[conn.target_id] = set()
                self._adjacency[conn.target_id].add(conn.source_id)

    def __len__(self) -> int:
        return len(self._connections)

    def __repr__(self) -> str:
        return f"ConnectionGraph({len(self._connections)} connections, {len(self._adjacency)} nodes)"
