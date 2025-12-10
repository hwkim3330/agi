"""
Long-term Memory - Episodic + Semantic (CoALA-based)

Stores:
- Facts and knowledge (semantic)
- Past experiences (episodic)
- Learned concepts
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False


class LongTermMemory:
    """
    Persistent long-term memory with semantic search.

    Features:
    - Semantic similarity search (when embeddings available)
    - Keyword-based fallback search
    - Metadata filtering
    - Automatic persistence
    """

    def __init__(self, data_dir: str = "data/memories/long_term"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self._memories: Dict[str, Dict] = {}
        self._embeddings: Dict[str, Any] = {}

        # Initialize embedding model if available
        self._encoder = None
        if HAS_EMBEDDINGS:
            try:
                self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                pass

        self.load()

    def store(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Store new memory"""
        memory_id = str(uuid.uuid4())[:12]

        memory = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": None
        }

        self._memories[memory_id] = memory

        # Generate embedding if available
        if self._encoder:
            self._embeddings[memory_id] = self._encoder.encode(content)

        return memory_id

    def retrieve(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Retrieve relevant memories"""

        # Filter by metadata first
        candidates = list(self._memories.values())
        if filters:
            candidates = [m for m in candidates if self._matches_filters(m, filters)]

        if not candidates:
            return []

        # Semantic search if embeddings available
        if self._encoder and self._embeddings:
            query_embedding = self._encoder.encode(query)
            scored = []

            for mem in candidates:
                if mem["id"] in self._embeddings:
                    similarity = self._cosine_similarity(
                        query_embedding,
                        self._embeddings[mem["id"]]
                    )
                    scored.append((mem, similarity))

            scored.sort(key=lambda x: x[1], reverse=True)
            results = [{"**mem, "similarity": sim} for mem, sim in scored[:k]]

        else:
            # Keyword fallback
            query_words = set(query.lower().split())
            scored = []

            for mem in candidates:
                content_words = set(mem["content"].lower().split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    scored.append((mem, overlap))

            scored.sort(key=lambda x: x[1], reverse=True)
            results = [{"**mem, "similarity": score/len(query_words)} for mem, score in scored[:k]]

        # Update access counts
        for r in results:
            mem_id = r.get("id")
            if mem_id in self._memories:
                self._memories[mem_id]["access_count"] += 1
                self._memories[mem_id]["last_accessed"] = datetime.now().isoformat()

        return results

    def get(self, memory_id: str) -> Optional[Dict]:
        """Get specific memory by ID"""
        return self._memories.get(memory_id)

    def update(self, memory_id: str, content: Optional[str] = None, metadata: Optional[Dict] = None):
        """Update existing memory"""
        if memory_id not in self._memories:
            return

        if content:
            self._memories[memory_id]["content"] = content
            if self._encoder:
                self._embeddings[memory_id] = self._encoder.encode(content)

        if metadata:
            self._memories[memory_id]["metadata"].update(metadata)

    def delete(self, memory_id: str):
        """Delete memory"""
        self._memories.pop(memory_id, None)
        self._embeddings.pop(memory_id, None)

    def _matches_filters(self, memory: Dict, filters: Dict) -> bool:
        """Check if memory matches metadata filters"""
        metadata = memory.get("metadata", {})
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True

    def _cosine_similarity(self, a, b) -> float:
        """Calculate cosine similarity between vectors"""
        if not HAS_EMBEDDINGS:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def save(self):
        """Save memories to disk"""
        memories_file = os.path.join(self.data_dir, "memories.json")
        with open(memories_file, "w") as f:
            json.dump(self._memories, f, indent=2, default=str)

        if HAS_EMBEDDINGS and self._embeddings:
            embeddings_file = os.path.join(self.data_dir, "embeddings.npy")
            np.save(embeddings_file, {k: v for k, v in self._embeddings.items()})

    def load(self):
        """Load memories from disk"""
        memories_file = os.path.join(self.data_dir, "memories.json")
        if os.path.exists(memories_file):
            with open(memories_file, "r") as f:
                self._memories = json.load(f)

        if HAS_EMBEDDINGS:
            embeddings_file = os.path.join(self.data_dir, "embeddings.npy")
            if os.path.exists(embeddings_file):
                try:
                    self._embeddings = np.load(embeddings_file, allow_pickle=True).item()
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self._memories)

    def __repr__(self) -> str:
        return f"LongTermMemory({len(self._memories)} memories)"
