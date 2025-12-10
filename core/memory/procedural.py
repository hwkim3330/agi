"""
Procedural Memory - Skills and Action Patterns

Stores:
- Successful action sequences
- Learned skills
- CAPTCHA solving strategies
- Navigation patterns
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class ProceduralMemory:
    """
    Memory for learned procedures and action patterns.

    This is the "skill library" that stores reusable
    action sequences for common goals.
    """

    def __init__(self, data_dir: str = "data/memories/procedural"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self._patterns: List[Dict] = []
        self._skills: Dict[str, Dict] = {}

        self.load()

    def store_pattern(self, goal: str, actions: List[Dict], success: bool = True,
                      metadata: Optional[Dict] = None) -> str:
        """Store a successful action pattern"""

        pattern = {
            "id": f"pattern_{len(self._patterns)}",
            "goal": goal,
            "actions": actions,
            "success": success,
            "success_count": 1 if success else 0,
            "failure_count": 0 if success else 1,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat()
        }

        # Check for similar existing pattern
        similar = self._find_similar_pattern(goal, actions)
        if similar:
            # Update existing pattern
            if success:
                similar["success_count"] += 1
            else:
                similar["failure_count"] += 1
            similar["last_used"] = datetime.now().isoformat()
            return similar["id"]

        self._patterns.append(pattern)
        return pattern["id"]

    def store_skill(self, name: str, description: str, executor: str,
                    parameters: Optional[Dict] = None):
        """Store a named skill"""
        self._skills[name] = {
            "name": name,
            "description": description,
            "executor": executor,  # e.g., "vla", "browser", "code"
            "parameters": parameters or {},
            "usage_count": 0,
            "created_at": datetime.now().isoformat()
        }

    def retrieve(self, goal: str, k: int = 5) -> List[Dict]:
        """Retrieve patterns relevant to goal"""

        # Simple keyword matching
        goal_words = set(goal.lower().split())
        scored = []

        for pattern in self._patterns:
            pattern_words = set(pattern["goal"].lower().split())
            overlap = len(goal_words & pattern_words)

            if overlap > 0:
                # Score by overlap and success rate
                success_rate = pattern["success_count"] / max(1, pattern["success_count"] + pattern["failure_count"])
                score = overlap * success_rate
                scored.append((pattern, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:k]]

    def get_skill(self, name: str) -> Optional[Dict]:
        """Get a skill by name"""
        skill = self._skills.get(name)
        if skill:
            skill["usage_count"] += 1
        return skill

    def get_best_pattern(self, goal: str) -> Optional[Dict]:
        """Get the single best pattern for a goal"""
        patterns = self.retrieve(goal, k=1)
        return patterns[0] if patterns else None

    def _find_similar_pattern(self, goal: str, actions: List[Dict]) -> Optional[Dict]:
        """Find existing pattern with same goal and similar actions"""

        for pattern in self._patterns:
            if pattern["goal"].lower() == goal.lower():
                # Check if actions are similar
                if len(pattern["actions"]) == len(actions):
                    return pattern

        return None

    def extract_patterns(self, long_term_memory):
        """Extract patterns from long-term memory experiences"""
        # This would analyze past experiences and extract common patterns
        # For now, this is a placeholder
        pass

    def get_all_skills(self) -> List[str]:
        """Get list of all skill names"""
        return list(self._skills.keys())

    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        total_patterns = len(self._patterns)
        successful = sum(1 for p in self._patterns if p["success_count"] > p["failure_count"])

        return {
            "total_patterns": total_patterns,
            "successful_patterns": successful,
            "total_skills": len(self._skills),
            "most_used_skills": sorted(
                self._skills.items(),
                key=lambda x: x[1]["usage_count"],
                reverse=True
            )[:5]
        }

    def save(self):
        """Save to disk"""
        patterns_file = os.path.join(self.data_dir, "patterns.json")
        with open(patterns_file, "w") as f:
            json.dump(self._patterns, f, indent=2)

        skills_file = os.path.join(self.data_dir, "skills.json")
        with open(skills_file, "w") as f:
            json.dump(self._skills, f, indent=2)

    def load(self):
        """Load from disk"""
        patterns_file = os.path.join(self.data_dir, "patterns.json")
        if os.path.exists(patterns_file):
            with open(patterns_file, "r") as f:
                self._patterns = json.load(f)

        skills_file = os.path.join(self.data_dir, "skills.json")
        if os.path.exists(skills_file):
            with open(skills_file, "r") as f:
                self._skills = json.load(f)

    def __len__(self) -> int:
        return len(self._patterns)

    def __repr__(self) -> str:
        return f"ProceduralMemory({len(self._patterns)} patterns, {len(self._skills)} skills)"
