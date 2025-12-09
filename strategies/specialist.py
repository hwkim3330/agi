"""
Specialist Strategy - 전문가 자동 선택
질문 유형에 따라 적합한 AI를 자동 선택합니다.
"""
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


@dataclass
class AgentResponse:
    agent: str
    content: str
    latency: float
    success: bool
    metadata: Dict[str, Any] = None


class SpecialistStrategy:
    """전문가 자동 선택 전략"""

    # 각 에이전트의 전문 분야 키워드
    SPECIALTIES = {
        "claude": {
            "keywords": [
                "코드", "code", "debug", "디버그", "버그", "프로그래밍",
                "python", "javascript", "rust", "함수", "클래스",
                "알고리즘", "보안", "security", "취약점", "리뷰",
                "리팩토링", "아키텍처", "설계", "시스템"
            ],
            "weight": 1.2,
            "role": "기술 전문가"
        },
        "gemini": {
            "keywords": [
                "연구", "research", "논문", "데이터", "분석",
                "통계", "팩트", "fact", "검증", "이미지",
                "비디오", "멀티모달", "시장", "트렌드"
            ],
            "weight": 1.1,
            "role": "데이터 분석가"
        },
        "gpt": {
            "keywords": [
                "창의", "creative", "아이디어", "브레인스토밍",
                "전략", "strategy", "기획", "마케팅", "스토리",
                "글쓰기", "writing", "컨텐츠", "혁신"
            ],
            "weight": 1.0,
            "role": "창의적 문제해결사"
        }
    }

    def classify_question(self, question: str) -> Tuple[str, float]:
        """질문 유형 분류 및 최적 에이전트 선택"""
        question_lower = question.lower()
        scores = {}

        for agent, spec in self.SPECIALTIES.items():
            score = 0
            for keyword in spec["keywords"]:
                if keyword.lower() in question_lower:
                    score += 1

            # 가중치 적용
            scores[agent] = score * spec["weight"]

        # 최고 점수 에이전트
        if max(scores.values()) > 0:
            best = max(scores, key=scores.get)
            confidence = scores[best] / (sum(scores.values()) + 1)
            return best, confidence
        else:
            # 키워드 매칭 없으면 모든 에이전트 사용
            return "all", 0.5

    def execute(
        self,
        responses: List[AgentResponse],
        question: str = ""
    ) -> Dict[str, Any]:
        """전문가 선택 실행"""
        if not responses:
            return {
                "strategy": "specialist",
                "content": "No responses received",
                "confidence": 0.0
            }

        successful = [r for r in responses if r.success]
        if not successful:
            return {
                "strategy": "specialist",
                "content": "All agents failed",
                "confidence": 0.0
            }

        # 질문 분류
        recommended, classify_confidence = self.classify_question(question)

        if recommended == "all":
            # 모든 응답 통합
            return self._synthesize_all(successful)

        # 추천 에이전트 응답 찾기
        specialist_response = next(
            (r for r in successful if r.agent.lower() == recommended),
            None
        )

        if specialist_response:
            return {
                "strategy": "specialist",
                "content": specialist_response.content,
                "confidence": classify_confidence,
                "specialist": recommended,
                "role": self.SPECIALTIES[recommended]["role"],
                "reasoning": f"Selected {recommended} as specialist for this question"
            }
        else:
            # 추천 에이전트 응답 없으면 통합
            return self._synthesize_all(successful)

    def _synthesize_all(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """모든 응답 통합"""
        sections = []
        for r in responses:
            spec = self.SPECIALTIES.get(r.agent.lower(), {})
            role = spec.get("role", "Agent")
            sections.append(f"## {role} ({r.agent})\n\n{r.content}")

        return {
            "strategy": "specialist",
            "content": "\n\n---\n\n".join(sections),
            "confidence": len(responses) / 3,
            "specialist": "all",
            "reasoning": "No specific specialist identified, synthesized all responses"
        }
