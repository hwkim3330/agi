"""
Vote Strategy - 투표 기반 합의
최고 점수 응답을 선택합니다.
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


class VoteStrategy:
    """투표 기반 합의 전략"""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "claude": 1.0,
            "gemini": 1.0,
            "gpt": 1.0
        }

    def calculate_score(self, response: AgentResponse) -> float:
        """응답 점수 계산"""
        if not response.success:
            return 0.0

        # 기본 점수
        score = 1.0

        # 콘텐츠 길이 (적당한 길이 선호)
        content_len = len(response.content)
        if 100 <= content_len <= 5000:
            score *= 1.0
        elif content_len < 100:
            score *= content_len / 100
        else:
            score *= 5000 / content_len

        # 응답 시간 (빠를수록 좋음)
        time_score = 1.0 / (1.0 + response.latency / 30.0)
        score *= (0.7 + 0.3 * time_score)

        # 에이전트 가중치
        agent_weight = self.weights.get(response.agent.lower(), 1.0)
        score *= agent_weight

        return score

    def execute(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """최고 점수 응답 선택"""
        if not responses:
            return {
                "strategy": "vote",
                "content": "No responses received",
                "confidence": 0.0,
                "winner": None
            }

        successful = [r for r in responses if r.success]
        if not successful:
            return {
                "strategy": "vote",
                "content": "All agents failed",
                "confidence": 0.0,
                "winner": None
            }

        # 점수 계산
        scores = {r.agent: self.calculate_score(r) for r in successful}

        # 최고 점수 선택
        winner = max(scores, key=scores.get)
        winner_response = next(r for r in successful if r.agent == winner)

        # 신뢰도: 성공률 * 최고점수 정규화
        confidence = (len(successful) / len(responses)) * min(scores[winner], 1.0)

        return {
            "strategy": "vote",
            "content": winner_response.content,
            "confidence": confidence,
            "winner": winner,
            "scores": scores,
            "reasoning": f"Selected {winner} with score {scores[winner]:.3f}"
        }
