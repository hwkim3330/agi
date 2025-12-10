"""
Task Decomposer - Break complex tasks into sub-tasks

Based on LLM Agent Survey's task decomposition approaches
"""

from typing import Any, Dict, List, Optional


class TaskDecomposer:
    """
    Decomposes complex goals into manageable sub-tasks.

    Uses heuristics and patterns to determine:
    - How to break down the task
    - Which agent type is best for each sub-task
    - Dependencies between sub-tasks
    """

    # Agent type mappings based on task keywords
    AGENT_MAPPINGS = {
        "vla": [
            "browse", "click", "navigate", "scroll", "screenshot",
            "captcha", "form", "website", "page", "button",
            "login", "search", "web"
        ],
        "trinity": [
            "analyze", "compare", "evaluate", "explain", "summarize",
            "question", "answer", "opinion", "recommend", "decide",
            "pros cons", "complex"
        ],
        "research": [
            "find", "search", "lookup", "research", "discover",
            "arxiv", "news", "article", "paper", "learn",
            "information", "knowledge"
        ]
    }

    def __init__(self):
        pass

    async def decompose(self, goal: str, context: Optional[Dict] = None,
                        past_experiences: Optional[List] = None) -> List[Dict]:
        """
        Decompose a goal into sub-tasks.

        Returns list of dicts with:
        - description: what to do
        - agent_type: which agent should handle it
        - dependencies: list of sub-task IDs this depends on
        """

        goal_lower = goal.lower()

        # Check for simple single-step tasks
        if self._is_simple_task(goal_lower):
            agent_type = self._determine_agent(goal_lower)
            return [{
                "description": goal,
                "agent_type": agent_type,
                "dependencies": []
            }]

        # Check past experiences for similar patterns
        if past_experiences:
            similar = self._find_similar_decomposition(goal, past_experiences)
            if similar:
                return similar

        # Complex task decomposition
        sub_tasks = self._decompose_complex(goal, context)

        return sub_tasks

    def _is_simple_task(self, goal: str) -> bool:
        """Check if task is simple enough to not need decomposition"""
        # Short tasks are usually simple
        if len(goal.split()) < 10:
            return True

        # Single action keywords
        simple_keywords = ["click", "scroll", "type", "answer", "summarize"]
        for keyword in simple_keywords:
            if goal.startswith(keyword):
                return True

        return False

    def _determine_agent(self, goal: str) -> str:
        """Determine best agent type for a goal"""

        scores = {agent: 0 for agent in self.AGENT_MAPPINGS}

        for agent, keywords in self.AGENT_MAPPINGS.items():
            for keyword in keywords:
                if keyword in goal:
                    scores[agent] += 1

        # Return highest scoring agent, default to trinity
        best_agent = max(scores.items(), key=lambda x: x[1])
        return best_agent[0] if best_agent[1] > 0 else "trinity"

    def _decompose_complex(self, goal: str, context: Optional[Dict]) -> List[Dict]:
        """Decompose complex task into sub-tasks"""

        sub_tasks = []
        goal_lower = goal.lower()

        # Pattern: Research then analyze
        if any(kw in goal_lower for kw in ["learn about", "understand", "research"]):
            sub_tasks.append({
                "description": f"Research information about: {goal}",
                "agent_type": "research",
                "dependencies": []
            })
            sub_tasks.append({
                "description": f"Analyze and synthesize findings about: {goal}",
                "agent_type": "trinity",
                "dependencies": [sub_tasks[0].get("id", "0")]
            })

        # Pattern: Browse then extract
        elif any(kw in goal_lower for kw in ["browse", "visit", "go to", "open"]):
            sub_tasks.append({
                "description": f"Navigate to relevant website for: {goal}",
                "agent_type": "vla",
                "dependencies": []
            })
            sub_tasks.append({
                "description": f"Extract relevant information",
                "agent_type": "vla",
                "dependencies": [sub_tasks[0].get("id", "0")]
            })

        # Pattern: Compare multiple things
        elif "compare" in goal_lower or "vs" in goal_lower:
            # Extract items to compare
            sub_tasks.append({
                "description": f"Research first item in comparison",
                "agent_type": "research",
                "dependencies": []
            })
            sub_tasks.append({
                "description": f"Research second item in comparison",
                "agent_type": "research",
                "dependencies": []
            })
            sub_tasks.append({
                "description": f"Compare and analyze both items: {goal}",
                "agent_type": "trinity",
                "dependencies": ["0", "1"]
            })

        # Pattern: Multi-step web task
        elif any(kw in goal_lower for kw in ["login", "fill", "submit", "form"]):
            sub_tasks.append({
                "description": f"Navigate to the page",
                "agent_type": "vla",
                "dependencies": []
            })
            sub_tasks.append({
                "description": f"Fill in required information",
                "agent_type": "vla",
                "dependencies": ["0"]
            })
            sub_tasks.append({
                "description": f"Submit and verify completion",
                "agent_type": "vla",
                "dependencies": ["1"]
            })

        # Default: single task
        else:
            agent_type = self._determine_agent(goal_lower)
            sub_tasks.append({
                "description": goal,
                "agent_type": agent_type,
                "dependencies": []
            })

        # Add IDs
        for i, task in enumerate(sub_tasks):
            task["id"] = str(i)

        return sub_tasks

    def _find_similar_decomposition(self, goal: str, experiences: List) -> Optional[List[Dict]]:
        """Find similar decomposition from past experiences"""
        # Simple keyword matching for now
        goal_words = set(goal.lower().split())

        for exp in experiences:
            if isinstance(exp, dict) and "goal" in exp:
                exp_words = set(exp["goal"].lower().split())
                overlap = len(goal_words & exp_words) / max(len(goal_words), 1)

                if overlap > 0.6 and "plan" in exp:
                    # Adapt past plan
                    return exp["plan"]

        return None
