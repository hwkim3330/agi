"""
SPICE Reasoner - Task Solver

The Reasoner role in SPICE self-play:
1. Receives tasks from Challenger (WITHOUT seeing source document)
2. Attempts to solve using only its learned knowledge
3. Generates answers that can be verified against ground truth
4. Learns from correct/incorrect outcomes

Based on Meta's SPICE paper (arXiv:2510.24684)
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .challenger import Task


@dataclass
class Attempt:
    """A single attempt at solving a task"""
    task_id: str
    response: str
    score: float  # 0.0 to 1.0
    reasoning_trace: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "response": self.response,
            "score": self.score,
            "reasoning_trace": self.reasoning_trace,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class Reasoner:
    """
    SPICE Reasoner - Solves tasks without seeing source documents.

    The Reasoner must rely on:
    - Knowledge learned during training
    - Reasoning capabilities
    - Pattern recognition

    Key insight from SPICE: The Reasoner never sees the source
    document, forcing it to generalize learned knowledge rather
    than memorize specific documents.
    """

    REASONING_PROMPT = """You are a reasoning agent. Answer the following question.

Think step by step before giving your final answer.

Question: {question}

Your response should follow this format:
REASONING: [your step-by-step reasoning]
ANSWER: [your final answer]"""

    def __init__(self, llm_model: str = "local", data_dir: str = "data/spice"):
        self.llm_model = llm_model
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.attempts: List[Attempt] = []
        self.performance_history: List[Dict] = []
        self._llm = None

        self._load_history()

    async def solve(self, task: Task) -> Attempt:
        """
        Attempt to solve a task.

        Args:
            task: Task from Challenger (only question, no ground truth shown)

        Returns:
            Attempt object with response and reasoning
        """
        # Generate response
        response, reasoning = await self._generate_response(task.question)

        # Score against ground truth
        score = self._score_response(response, task.ground_truth)

        attempt = Attempt(
            task_id=task.id,
            response=response,
            score=score,
            reasoning_trace=reasoning,
            timestamp=datetime.now().isoformat(),
            metadata={
                "task_type": task.task_type,
                "difficulty": task.difficulty
            }
        )

        self.attempts.append(attempt)
        self._update_performance(attempt)
        self._save_history()

        return attempt

    async def _generate_response(self, question: str) -> Tuple[str, str]:
        """Generate response for question"""

        # Try LLM-based generation
        if self.llm_model != "local":
            return await self._llm_generate(question)

        # Fallback to simple heuristic
        return self._heuristic_generate(question)

    async def _llm_generate(self, question: str) -> Tuple[str, str]:
        """Use LLM to generate response"""
        try:
            if self._llm is None:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                model_id = "Qwen/Qwen2.5-1.5B-Instruct"
                self._llm = {
                    "tokenizer": AutoTokenizer.from_pretrained(model_id),
                    "model": AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.bfloat16,
                        device_map="auto"
                    )
                }

            prompt = self.REASONING_PROMPT.format(question=question)

            inputs = self._llm["tokenizer"](prompt, return_tensors="pt").to(
                self._llm["model"].device
            )

            outputs = self._llm["model"].generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.3  # Lower temp for reasoning
            )

            response = self._llm["tokenizer"].decode(outputs[0], skip_special_tokens=True)

            # Parse response
            reasoning = ""
            answer = response

            if "REASONING:" in response and "ANSWER:" in response:
                parts = response.split("ANSWER:")
                reasoning = parts[0].split("REASONING:")[-1].strip()
                answer = parts[1].strip()

            return answer, reasoning

        except Exception as e:
            print(f"[Reasoner] LLM generation failed: {e}")
            return self._heuristic_generate(question)

    def _heuristic_generate(self, question: str) -> Tuple[str, str]:
        """Simple heuristic response (fallback)"""

        # Extract key information from question
        question_lower = question.lower()

        # Very basic response generation
        if "summarize" in question_lower:
            reasoning = "Extracting main points from the provided text."
            answer = "The text discusses the main topic and its key aspects."

        elif "what can be inferred" in question_lower:
            reasoning = "Analyzing the text for implicit information."
            answer = "Based on the information provided, we can infer related concepts."

        elif "explain" in question_lower:
            reasoning = "Breaking down the concept into understandable parts."
            answer = "The concept can be explained through its key components."

        else:
            reasoning = "Processing the question to extract relevant information."
            answer = "Based on the available information, the answer relates to the topic discussed."

        return answer, reasoning

    def _score_response(self, response: str, ground_truth: str) -> float:
        """
        Score response against ground truth.

        Uses multiple metrics:
        - Exact match
        - Token overlap (F1)
        - Semantic similarity (if available)
        """
        if not response or not ground_truth:
            return 0.0

        response_lower = response.lower().strip()
        truth_lower = ground_truth.lower().strip()

        # Exact match
        if response_lower == truth_lower:
            return 1.0

        # Token overlap (simple F1)
        response_tokens = set(response_lower.split())
        truth_tokens = set(truth_lower.split())

        if not response_tokens or not truth_tokens:
            return 0.0

        overlap = response_tokens & truth_tokens
        precision = len(overlap) / len(response_tokens)
        recall = len(overlap) / len(truth_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)

        # Bonus for substring match
        if truth_lower in response_lower or response_lower in truth_lower:
            f1 = min(1.0, f1 + 0.2)

        return f1

    def _update_performance(self, attempt: Attempt):
        """Update performance tracking"""

        self.performance_history.append({
            "timestamp": attempt.timestamp,
            "task_type": attempt.metadata.get("task_type"),
            "difficulty": attempt.metadata.get("difficulty"),
            "score": attempt.score
        })

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""

        if not self.performance_history:
            return {"total_attempts": 0}

        scores = [p["score"] for p in self.performance_history]

        # By task type
        by_type = {}
        for p in self.performance_history:
            task_type = p.get("task_type", "unknown")
            if task_type not in by_type:
                by_type[task_type] = []
            by_type[task_type].append(p["score"])

        type_avg = {k: sum(v)/len(v) for k, v in by_type.items()}

        # By difficulty
        by_difficulty = {"easy": [], "medium": [], "hard": []}
        for p in self.performance_history:
            diff = p.get("difficulty", 0.5)
            if diff < 0.33:
                by_difficulty["easy"].append(p["score"])
            elif diff < 0.66:
                by_difficulty["medium"].append(p["score"])
            else:
                by_difficulty["hard"].append(p["score"])

        diff_avg = {k: sum(v)/len(v) if v else 0 for k, v in by_difficulty.items()}

        return {
            "total_attempts": len(scores),
            "average_score": sum(scores) / len(scores),
            "best_score": max(scores),
            "worst_score": min(scores),
            "by_task_type": type_avg,
            "by_difficulty": diff_avg,
            "recent_trend": sum(scores[-10:]) / len(scores[-10:]) if len(scores) >= 10 else None
        }

    def get_current_level(self) -> float:
        """
        Estimate current capability level for curriculum.

        SPICE insight: Track Reasoner's capability to
        adaptively select appropriate difficulty tasks.
        """
        if len(self.performance_history) < 5:
            return 0.5  # Default middle difficulty

        # Use recent performance
        recent = self.performance_history[-20:]
        avg_score = sum(p["score"] for p in recent) / len(recent)

        # Map score to difficulty level
        # High score = can handle higher difficulty
        return min(1.0, avg_score + 0.1)

    def get_training_samples(self, min_score: float = 0.5) -> List[Dict]:
        """
        Get successful attempts for training.

        SPICE insight: Only use successful attempts
        (above threshold) for updating model weights.
        """
        samples = []

        for attempt in self.attempts:
            if attempt.score >= min_score:
                samples.append({
                    "task_id": attempt.task_id,
                    "response": attempt.response,
                    "reasoning": attempt.reasoning_trace,
                    "score": attempt.score
                })

        return samples

    def _save_history(self):
        """Save attempt history"""
        history_file = os.path.join(self.data_dir, "reasoner_history.json")
        data = {
            "attempts": [a.to_dict() for a in self.attempts[-1000:]],  # Keep last 1000
            "performance_history": self.performance_history[-1000:]
        }
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_history(self):
        """Load attempt history"""
        history_file = os.path.join(self.data_dir, "reasoner_history.json")
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                data = json.load(f)

            self.attempts = [
                Attempt(**a) for a in data.get("attempts", [])
            ]
            self.performance_history = data.get("performance_history", [])

            print(f"[Reasoner] Loaded {len(self.attempts)} attempts")

    def __repr__(self) -> str:
        stats = self.get_performance_stats()
        avg = stats.get("average_score", 0)
        return f"Reasoner(attempts={stats['total_attempts']}, avg_score={avg:.2f})"
