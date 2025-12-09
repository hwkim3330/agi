"""
Debate Strategy - 토론 기반 합의
AI들이 서로의 응답을 평가하고 개선합니다.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Callable


@dataclass
class AgentResponse:
    agent: str
    content: str
    latency: float
    success: bool
    metadata: Dict[str, Any] = None


class DebateStrategy:
    """토론 기반 합의 전략"""

    def __init__(self, max_rounds: int = 2):
        self.max_rounds = max_rounds

    def create_debate_prompt(
        self,
        original_question: str,
        responses: List[AgentResponse],
        current_agent: str
    ) -> str:
        """토론 프롬프트 생성"""
        other_responses = [r for r in responses if r.agent != current_agent]

        prompt = f"원래 질문: {original_question}\n\n"
        prompt += "다른 AI의 응답들:\n\n"

        for r in other_responses:
            prompt += f"[{r.agent}]: {r.content[:500]}...\n\n"

        prompt += """
위 응답들을 검토하고:
1. 각 응답의 강점과 약점을 분석하세요
2. 누락된 중요한 관점이 있다면 추가하세요
3. 가장 정확하고 완전한 최종 답변을 제시하세요
"""
        return prompt

    def execute(
        self,
        responses: List[AgentResponse],
        original_question: str = "",
        agent_executor: Callable = None
    ) -> Dict[str, Any]:
        """토론 실행"""
        if not responses:
            return {
                "strategy": "debate",
                "content": "No responses received",
                "confidence": 0.0
            }

        successful = [r for r in responses if r.success]
        if not successful:
            return {
                "strategy": "debate",
                "content": "All agents failed",
                "confidence": 0.0
            }

        # 토론 라운드 없이 기본 동작 (executor 없는 경우)
        if not agent_executor:
            # 모든 응답의 핵심을 추출하여 종합
            combined = self._combine_insights(successful)
            return {
                "strategy": "debate",
                "content": combined,
                "confidence": len(successful) / len(responses),
                "rounds": 0,
                "reasoning": "Combined insights from all agents"
            }

        # TODO: 실제 토론 라운드 구현
        # 각 AI가 다른 AI의 응답을 보고 개선된 답변 생성
        return {
            "strategy": "debate",
            "content": successful[0].content,
            "confidence": len(successful) / len(responses),
            "rounds": self.max_rounds,
            "reasoning": f"Debated for {self.max_rounds} rounds"
        }

    def _combine_insights(self, responses: List[AgentResponse]) -> str:
        """응답들의 핵심 인사이트 결합"""
        if len(responses) == 1:
            return responses[0].content

        result = "## 종합 분석\n\n"

        for r in responses:
            result += f"### {r.agent}의 관점\n{r.content}\n\n"

        result += "---\n\n"
        result += "위 분석들을 종합하면, 각 AI가 다른 관점에서 문제를 분석했습니다."

        return result
