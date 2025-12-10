"""
Planner - Plan Generation and Adaptation

Creates and modifies execution plans with feedback
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import asyncio


@dataclass
class SubTask:
    """Minimal SubTask for planning"""
    id: str
    description: str
    agent_type: str
    dependencies: List[str]


class Planner:
    """
    Creates and adapts execution plans.

    Features:
    - Plan generation from sub-tasks
    - Alternative plan creation on failure
    - Plan optimization
    """

    def __init__(self):
        self._alternatives_tried = {}

    async def create_alternative(self, failed_task: Any, error: str,
                                 available_agents: List[str]) -> Optional[SubTask]:
        """
        Create an alternative approach when a task fails.

        Based on AgentOrchestra's closed-loop feedback adaptation.
        """

        task_id = getattr(failed_task, 'id', str(id(failed_task)))

        # Track alternatives tried
        if task_id not in self._alternatives_tried:
            self._alternatives_tried[task_id] = []

        tried = self._alternatives_tried[task_id]

        # Strategy 1: Try different agent
        original_agent = getattr(failed_task, 'agent_type', 'unknown')
        for agent in available_agents:
            if agent != original_agent and agent not in tried:
                self._alternatives_tried[task_id].append(agent)
                return SubTask(
                    id=f"{task_id}_alt_{len(tried)}",
                    description=f"[Retry with {agent}] {failed_task.description}",
                    agent_type=agent,
                    dependencies=[]
                )

        # Strategy 2: Simplify the task
        if "simplify" not in tried:
            self._alternatives_tried[task_id].append("simplify")
            simplified_desc = self._simplify_task(failed_task.description)
            return SubTask(
                id=f"{task_id}_simplified",
                description=simplified_desc,
                agent_type=original_agent,
                dependencies=[]
            )

        # Strategy 3: Break into smaller steps
        if "split" not in tried:
            self._alternatives_tried[task_id].append("split")
            # Return first part of split task
            return SubTask(
                id=f"{task_id}_part1",
                description=f"[Part 1] {failed_task.description[:100]}...",
                agent_type=original_agent,
                dependencies=[]
            )

        # No more alternatives
        return None

    def _simplify_task(self, description: str) -> str:
        """Simplify a task description"""
        # Remove complex modifiers
        simplifications = [
            ("thoroughly ", ""),
            ("comprehensively ", ""),
            ("in detail ", ""),
            ("completely ", ""),
            ("all ", ""),
        ]

        result = description
        for old, new in simplifications:
            result = result.replace(old, new)

        return f"[Simplified] {result}"

    async def optimize_plan(self, sub_tasks: List[Dict]) -> List[Dict]:
        """
        Optimize a plan for better execution.

        - Identify parallelizable tasks
        - Reorder for efficiency
        - Remove redundant tasks
        """

        # Group tasks by dependencies
        no_deps = [t for t in sub_tasks if not t.get("dependencies")]
        with_deps = [t for t in sub_tasks if t.get("dependencies")]

        # Tasks with no dependencies can run in parallel
        optimized = []

        # Add independent tasks first (marked for parallel execution)
        if len(no_deps) > 1:
            for task in no_deps:
                task["parallel_group"] = 0
            optimized.extend(no_deps)
        else:
            optimized.extend(no_deps)

        # Add dependent tasks in order
        completed_ids = {t.get("id") for t in no_deps}

        while with_deps:
            # Find tasks whose dependencies are satisfied
            ready = []
            still_waiting = []

            for task in with_deps:
                deps = set(task.get("dependencies", []))
                if deps.issubset(completed_ids):
                    ready.append(task)
                else:
                    still_waiting.append(task)

            if not ready:
                # Circular dependency or missing - add remaining
                optimized.extend(still_waiting)
                break

            optimized.extend(ready)
            completed_ids.update(t.get("id") for t in ready)
            with_deps = still_waiting

        return optimized

    def estimate_complexity(self, sub_tasks: List[Dict]) -> Dict:
        """Estimate plan complexity"""

        total_tasks = len(sub_tasks)
        agents_used = set(t.get("agent_type") for t in sub_tasks)
        max_depth = self._calculate_dependency_depth(sub_tasks)

        return {
            "total_tasks": total_tasks,
            "agents_used": list(agents_used),
            "max_dependency_depth": max_depth,
            "estimated_complexity": "simple" if total_tasks < 3 else "medium" if total_tasks < 7 else "complex"
        }

    def _calculate_dependency_depth(self, sub_tasks: List[Dict]) -> int:
        """Calculate maximum dependency chain depth"""

        task_map = {t.get("id", str(i)): t for i, t in enumerate(sub_tasks)}
        depths = {}

        def get_depth(task_id: str) -> int:
            if task_id in depths:
                return depths[task_id]

            task = task_map.get(task_id)
            if not task or not task.get("dependencies"):
                depths[task_id] = 0
                return 0

            max_dep_depth = max(
                get_depth(dep) for dep in task.get("dependencies", [])
            )
            depths[task_id] = max_dep_depth + 1
            return depths[task_id]

        for task_id in task_map:
            get_depth(task_id)

        return max(depths.values()) if depths else 0
