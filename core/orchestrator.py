"""
Orchestrator - Planning Agent based on AgentOrchestra (arXiv:2506.12508)

Hierarchical multi-agent framework with:
- Task decomposition
- Dynamic role allocation
- Closed-loop feedback
- State management
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from core.memory import MemorySystem
from core.planning import TaskDecomposer, Planner


class TaskStatus(Enum):
    """Task execution states"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass
class SubTask:
    """A decomposed sub-task"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    agent_type: str = ""  # vla, trinity, research
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "agent_type": self.agent_type,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": str(self.result)[:200] if self.result else None,
        }


@dataclass
class Plan:
    """A plan containing multiple sub-tasks"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    sub_tasks: List[SubTask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    def get_ready_tasks(self) -> List[SubTask]:
        """Get tasks that are ready to execute (dependencies satisfied)"""
        completed_ids = {t.id for t in self.sub_tasks if t.status == TaskStatus.COMPLETED}
        ready = []
        for task in self.sub_tasks:
            if task.status == TaskStatus.PENDING:
                if all(dep in completed_ids for dep in task.dependencies):
                    ready.append(task)
        return ready

    def is_complete(self) -> bool:
        return all(t.status == TaskStatus.COMPLETED for t in self.sub_tasks)

    def is_blocked(self) -> bool:
        return any(t.status == TaskStatus.BLOCKED for t in self.sub_tasks)


class Orchestrator:
    """
    Central Planning Agent that coordinates sub-agents.

    Based on AgentOrchestra's two-tier architecture:
    - Top-level planning agent (this class)
    - Modular sub-agents (VLA, Trinity, Research)
    """

    def __init__(self, memory: Optional[MemorySystem] = None):
        self.memory = memory or MemorySystem()
        self.decomposer = TaskDecomposer()
        self.planner = Planner()
        self.agents: Dict[str, Any] = {}  # Registered sub-agents
        self.current_plan: Optional[Plan] = None
        self.execution_history: List[Plan] = []

    def register_agent(self, agent_type: str, agent: Any):
        """Register a sub-agent for task execution"""
        self.agents[agent_type] = agent
        print(f"[Orchestrator] Registered agent: {agent_type}")

    async def execute(self, goal: str, context: Optional[Dict] = None) -> Dict:
        """
        Main execution loop:
        1. Decompose goal into sub-tasks
        2. Create execution plan
        3. Execute with feedback loop
        4. Return final result
        """
        print(f"\n{'='*60}")
        print(f"[Orchestrator] Goal: {goal}")
        print(f"{'='*60}\n")

        # Store goal in working memory
        self.memory.working.set("current_goal", goal)
        self.memory.working.set("context", context or {})

        # 1. Task Decomposition
        sub_tasks = await self._decompose_task(goal, context)

        # 2. Create Plan
        self.current_plan = Plan(goal=goal, sub_tasks=sub_tasks)
        print(f"[Orchestrator] Created plan with {len(sub_tasks)} sub-tasks")

        # 3. Execute with feedback loop
        result = await self._execute_plan()

        # 4. Store in history and long-term memory
        self.execution_history.append(self.current_plan)
        await self._store_experience(goal, result)

        return result

    async def _decompose_task(self, goal: str, context: Optional[Dict]) -> List[SubTask]:
        """Decompose goal into manageable sub-tasks"""

        # Retrieve relevant past experiences
        relevant_memories = self.memory.long_term.retrieve(goal, k=3)

        # Use decomposer to break down task
        decomposition = await self.decomposer.decompose(
            goal=goal,
            context=context,
            past_experiences=relevant_memories
        )

        sub_tasks = []
        for i, item in enumerate(decomposition):
            task = SubTask(
                description=item["description"],
                agent_type=item["agent_type"],
                dependencies=item.get("dependencies", [])
            )
            sub_tasks.append(task)
            print(f"  [{i+1}] {task.description} -> {task.agent_type}")

        return sub_tasks

    async def _execute_plan(self) -> Dict:
        """Execute plan with closed-loop feedback"""

        results = {}
        max_iterations = 50  # Safety limit
        iteration = 0

        while not self.current_plan.is_complete() and iteration < max_iterations:
            iteration += 1

            # Get tasks ready to execute
            ready_tasks = self.current_plan.get_ready_tasks()

            if not ready_tasks:
                if self.current_plan.is_blocked():
                    print("[Orchestrator] Plan is blocked!")
                    break
                await asyncio.sleep(0.1)
                continue

            # Execute ready tasks in parallel
            tasks_to_run = []
            for task in ready_tasks:
                task.status = TaskStatus.IN_PROGRESS
                tasks_to_run.append(self._execute_task(task))

            # Wait for all parallel tasks
            task_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

            # Process results with feedback
            for task, result in zip(ready_tasks, task_results):
                if isinstance(result, Exception):
                    task.status = TaskStatus.FAILED
                    task.result = str(result)
                    print(f"[Orchestrator] Task failed: {task.description[:50]}...")

                    # Adapt plan based on failure
                    await self._adapt_plan(task, result)
                else:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    task.result = result
                    results[task.id] = result
                    print(f"[Orchestrator] Task completed: {task.description[:50]}...")

        # Synthesize final result
        final_result = await self._synthesize_results(results)

        return {
            "success": self.current_plan.is_complete(),
            "goal": self.current_plan.goal,
            "result": final_result,
            "tasks_completed": len([t for t in self.current_plan.sub_tasks if t.status == TaskStatus.COMPLETED]),
            "tasks_total": len(self.current_plan.sub_tasks)
        }

    async def _execute_task(self, task: SubTask) -> Any:
        """Execute a single sub-task using appropriate agent"""

        agent = self.agents.get(task.agent_type)
        if not agent:
            raise ValueError(f"No agent registered for type: {task.agent_type}")

        # Get context from working memory
        context = self.memory.working.get_all()

        # Execute through agent
        result = await agent.execute(
            task=task.description,
            context=context
        )

        # Update working memory with result
        self.memory.working.set(f"result_{task.id}", result)

        return result

    async def _adapt_plan(self, failed_task: SubTask, error: Exception):
        """Adapt plan based on task failure (closed-loop feedback)"""

        # Mark dependent tasks as blocked
        for task in self.current_plan.sub_tasks:
            if failed_task.id in task.dependencies:
                task.status = TaskStatus.BLOCKED

        # Try to create alternative approach
        alternative = await self.planner.create_alternative(
            failed_task=failed_task,
            error=str(error),
            available_agents=list(self.agents.keys())
        )

        if alternative:
            # Insert alternative task
            self.current_plan.sub_tasks.append(alternative)
            print(f"[Orchestrator] Created alternative: {alternative.description[:50]}...")

    async def _synthesize_results(self, results: Dict) -> Any:
        """Combine results from all sub-tasks"""

        if not results:
            return None

        # Simple synthesis: combine all results
        synthesis = {
            "individual_results": results,
            "summary": None
        }

        # If we have a synthesizer agent, use it
        if "synthesizer" in self.agents:
            synthesis["summary"] = await self.agents["synthesizer"].synthesize(results)
        else:
            # Basic summary
            synthesis["summary"] = f"Completed {len(results)} tasks successfully"

        return synthesis

    async def _store_experience(self, goal: str, result: Dict):
        """Store execution experience in long-term memory"""

        experience = {
            "goal": goal,
            "success": result.get("success", False),
            "plan": [t.to_dict() for t in self.current_plan.sub_tasks],
            "timestamp": datetime.now().isoformat()
        }

        self.memory.long_term.store(
            content=json.dumps(experience),
            metadata={
                "type": "execution_experience",
                "goal": goal,
                "success": result.get("success", False)
            }
        )

    def get_status(self) -> Dict:
        """Get current orchestrator status"""
        return {
            "registered_agents": list(self.agents.keys()),
            "current_plan": {
                "goal": self.current_plan.goal if self.current_plan else None,
                "tasks": [t.to_dict() for t in self.current_plan.sub_tasks] if self.current_plan else []
            },
            "history_count": len(self.execution_history)
        }


# Convenience function to create configured orchestrator
def create_orchestrator(memory: Optional[MemorySystem] = None) -> Orchestrator:
    """Factory function to create a configured Orchestrator"""
    return Orchestrator(memory=memory)
