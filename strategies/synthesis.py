"""
Synthesis Strategy - 응답 통합 합의
모든 응답을 통합하여 종합적인 결과를 생성합니다.
"""
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class AgentResponse:
    agent: str
    content: str
    latency: float
    success: bool
    metadata: Dict[str, Any] = None


class SynthesisStrategy:
    """응답 통합 전략"""

    AGENT_ROLES = {
        "claude": "기술 전문가",
        "gemini": "데이터 분석가",
        "gpt": "창의적 문제해결사"
    }

    def execute(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """모든 응답 통합"""
        if not responses:
            return {
                "strategy": "synthesis",
                "content": "No responses received",
                "confidence": 0.0
            }

        successful = [r for r in responses if r.success]
        if not successful:
            return {
                "strategy": "synthesis",
                "content": "All agents failed",
                "confidence": 0.0
            }

        # 통합 문서 생성
        sections = []
        for response in successful:
            role = self.AGENT_ROLES.get(response.agent.lower(), "Agent")
            sections.append(
                f"## {role} ({response.agent})\n\n{response.content}"
            )

        synthesized = "\n\n---\n\n".join(sections)

        # 신뢰도
        confidence = len(successful) / len(responses)

        return {
            "strategy": "synthesis",
            "content": synthesized,
            "confidence": confidence,
            "contributors": [r.agent for r in successful],
            "reasoning": f"Synthesized {len(successful)} responses"
        }
