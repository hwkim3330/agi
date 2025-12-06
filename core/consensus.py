#!/usr/bin/env python3
"""
AGI Trinity - Consensus Engine
합의 엔진 모듈

다중 에이전트 응답에서 최적의 결과를 도출하는 합의 알고리즘을 구현합니다.
"""
import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from collections import Counter


class ConsensusStrategy(Enum):
    """합의 전략"""
    VOTE = "vote"           # 투표 기반 (가장 좋은 응답 선택)
    SYNTHESIS = "synthesis"  # 통합 (모든 응답 결합)
    FANOUT = "fanout"       # 개별 표시 (합의 없음)
    WEIGHTED = "weighted"    # 가중치 기반
    SEMANTIC = "semantic"    # 의미 기반 클러스터링


@dataclass
class ConsensusResult:
    """합의 결과"""
    strategy: str
    content: str
    confidence: float
    reasoning: str
    individual_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConsensusEngine:
    """
    합의 엔진

    다중 에이전트의 응답을 분석하고 최적의 결과를 도출합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._scoring_weights = self.config.get("scoring_weights", {
            "content_length": 0.2,
            "response_time": 0.15,
            "success_rate": 0.25,
            "confidence": 0.25,
            "uniqueness": 0.15
        })

    def calculate_consensus(
        self,
        responses: List[Dict[str, Any]],
        strategy: str = "vote"
    ) -> ConsensusResult:
        """
        합의 결과를 계산합니다.

        Args:
            responses: 에이전트 응답 목록
            strategy: 합의 전략

        Returns:
            ConsensusResult: 합의 결과
        """
        if not responses:
            return ConsensusResult(
                strategy=strategy,
                content="No responses received",
                confidence=0.0,
                reasoning="Empty response list"
            )

        # 성공한 응답만 필터링
        successful = [r for r in responses if r.get("success", False)]

        if not successful:
            return ConsensusResult(
                strategy=strategy,
                content="All agents failed to respond",
                confidence=0.0,
                reasoning="No successful responses available"
            )

        # 전략에 따른 합의 계산
        strategy_handlers = {
            "vote": self._vote_consensus,
            "synthesis": self._synthesis_consensus,
            "fanout": self._fanout_consensus,
            "weighted": self._weighted_consensus,
            "semantic": self._semantic_consensus
        }

        handler = strategy_handlers.get(strategy, self._vote_consensus)
        return handler(successful, responses)

    def _vote_consensus(
        self,
        successful: List[Dict[str, Any]],
        all_responses: List[Dict[str, Any]]
    ) -> ConsensusResult:
        """
        투표 기반 합의 - 가장 우수한 응답 선택
        """
        scores = {}

        for response in successful:
            agent_name = response.get("agent_name", "unknown")
            score = self._calculate_response_score(response)
            scores[agent_name] = score

        # 최고 점수 응답 선택
        best_agent = max(scores, key=scores.get)
        best_response = next(
            r for r in successful
            if r.get("agent_name") == best_agent
        )

        confidence = len(successful) / len(all_responses)

        return ConsensusResult(
            strategy="vote",
            content=best_response.get("content", ""),
            confidence=confidence,
            reasoning=f"Selected {best_agent} with score {scores[best_agent]:.3f}",
            individual_scores=scores,
            metadata={
                "winner": best_agent,
                "total_responses": len(all_responses),
                "successful_responses": len(successful)
            }
        )

    def _synthesis_consensus(
        self,
        successful: List[Dict[str, Any]],
        all_responses: List[Dict[str, Any]]
    ) -> ConsensusResult:
        """
        통합 합의 - 모든 응답 결합
        """
        sections = []
        scores = {}

        for response in successful:
            agent_name = response.get("agent_name", "unknown")
            role = response.get("metadata", {}).get("role", "Agent")
            content = response.get("content", "")

            scores[agent_name] = self._calculate_response_score(response)

            sections.append(f"## {role} ({agent_name})\n\n{content}")

        synthesized_content = "\n\n---\n\n".join(sections)

        # 요약 생성
        summary = self._generate_synthesis_summary(successful)

        final_content = f"# Trinity Synthesis\n\n{summary}\n\n---\n\n{synthesized_content}"

        confidence = len(successful) / len(all_responses)

        return ConsensusResult(
            strategy="synthesis",
            content=final_content,
            confidence=confidence,
            reasoning=f"Synthesized insights from {len(successful)} agents",
            individual_scores=scores,
            metadata={
                "agents_included": [r.get("agent_name") for r in successful],
                "total_responses": len(all_responses)
            }
        )

    def _fanout_consensus(
        self,
        successful: List[Dict[str, Any]],
        all_responses: List[Dict[str, Any]]
    ) -> ConsensusResult:
        """
        개별 표시 - 합의 없이 모든 응답 표시
        """
        sections = []
        for response in successful:
            agent_name = response.get("agent_name", "unknown")
            content = response.get("content", "")
            latency = response.get("latency", 0)
            sections.append(f"### {agent_name} ({latency:.2f}s)\n\n{content}")

        return ConsensusResult(
            strategy="fanout",
            content="\n\n".join(sections),
            confidence=1.0,
            reasoning="Fanout mode - all responses displayed individually",
            metadata={"response_count": len(successful)}
        )

    def _weighted_consensus(
        self,
        successful: List[Dict[str, Any]],
        all_responses: List[Dict[str, Any]]
    ) -> ConsensusResult:
        """
        가중치 기반 합의 - 에이전트별 가중치 적용
        """
        # 에이전트별 기본 가중치
        agent_weights = {
            "claude": 1.2,   # 기술 분석
            "gemini": 1.1,   # 연구/분석
            "codex": 1.0     # 창의성
        }

        weighted_scores = {}
        for response in successful:
            agent_name = response.get("agent_name", "unknown")
            base_score = self._calculate_response_score(response)
            weight = agent_weights.get(agent_name, 1.0)
            weighted_scores[agent_name] = base_score * weight

        best_agent = max(weighted_scores, key=weighted_scores.get)
        best_response = next(
            r for r in successful
            if r.get("agent_name") == best_agent
        )

        return ConsensusResult(
            strategy="weighted",
            content=best_response.get("content", ""),
            confidence=len(successful) / len(all_responses),
            reasoning=f"Weighted selection: {best_agent} (score: {weighted_scores[best_agent]:.3f})",
            individual_scores=weighted_scores
        )

    def _semantic_consensus(
        self,
        successful: List[Dict[str, Any]],
        all_responses: List[Dict[str, Any]]
    ) -> ConsensusResult:
        """
        의미 기반 합의 - 텍스트 유사도 클러스터링
        """
        contents = [r.get("content", "") for r in successful]

        # 간단한 유사도 계산 (실제로는 WASM 모듈 사용)
        similarities = self._calculate_pairwise_similarity(contents)

        # 가장 다른 에이전트들의 응답이 모두 동의하는 내용 추출
        common_themes = self._extract_common_themes(contents)

        if common_themes:
            synthesized = f"## Common Themes\n\n{common_themes}\n\n"
        else:
            synthesized = ""

        # 최고 유사도 응답 선택
        avg_similarities = {}
        for i, response in enumerate(successful):
            agent_name = response.get("agent_name", "unknown")
            avg_sim = sum(similarities[i]) / (len(similarities[i]) or 1)
            avg_similarities[agent_name] = avg_sim

        best_agent = max(avg_similarities, key=avg_similarities.get)
        best_response = next(
            r for r in successful
            if r.get("agent_name") == best_agent
        )

        synthesized += best_response.get("content", "")

        return ConsensusResult(
            strategy="semantic",
            content=synthesized,
            confidence=max(avg_similarities.values()) if avg_similarities else 0,
            reasoning=f"Semantic analysis: {best_agent} highest agreement",
            individual_scores=avg_similarities
        )

    def _calculate_response_score(self, response: Dict[str, Any]) -> float:
        """응답 점수 계산"""
        content = response.get("content", "")
        latency = response.get("latency", 0)
        confidence = response.get("confidence", 0.5)

        # 콘텐츠 길이 점수 (너무 짧거나 긴 것은 감점)
        content_len = len(content)
        if content_len < 100:
            length_score = content_len / 100
        elif content_len > 10000:
            length_score = 10000 / content_len
        else:
            length_score = 1.0

        # 응답 시간 점수 (빠를수록 좋음)
        time_score = 1.0 / (1.0 + latency / 30)

        # 가중 합산
        weights = self._scoring_weights
        score = (
            length_score * weights.get("content_length", 0.2) +
            time_score * weights.get("response_time", 0.15) +
            confidence * weights.get("confidence", 0.25) +
            (1.0 if response.get("success") else 0.0) * weights.get("success_rate", 0.25)
        )

        return min(1.0, max(0.0, score))

    def _generate_synthesis_summary(self, responses: List[Dict[str, Any]]) -> str:
        """합성 요약 생성"""
        agent_count = len(responses)
        total_length = sum(len(r.get("content", "")) for r in responses)
        avg_latency = sum(r.get("latency", 0) for r in responses) / agent_count

        return (
            f"**Summary**: Synthesized from {agent_count} agents | "
            f"Total: {total_length:,} chars | "
            f"Avg latency: {avg_latency:.2f}s"
        )

    def _calculate_pairwise_similarity(self, texts: List[str]) -> List[List[float]]:
        """텍스트 쌍 유사도 계산 (간단한 자카드 유사도)"""
        n = len(texts)
        similarities = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    similarities[i][j] = 1.0
                else:
                    similarities[i][j] = self._jaccard_similarity(texts[i], texts[j])

        return similarities

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """자카드 유사도 계산"""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _extract_common_themes(self, texts: List[str]) -> str:
        """공통 주제 추출"""
        # 모든 텍스트에서 단어 빈도 계산
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-zA-Z가-힣]{3,}\b', text.lower())
            all_words.extend(words)

        # 불용어 제외
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'can', 'this',
                    'that', 'these', 'those', 'and', 'but', 'or', 'for', 'with',
                    'not', 'you', 'your', 'they', 'their', 'its', 'from', 'about'}

        filtered_words = [w for w in all_words if w not in stopwords]

        # 빈도 계산
        counter = Counter(filtered_words)
        common = counter.most_common(10)

        if not common:
            return ""

        themes = [f"- {word} ({count})" for word, count in common]
        return "\n".join(themes)


# WASM 연동 인터페이스
class WasmConsensusEngine(ConsensusEngine):
    """
    WASM 최적화된 합의 엔진

    Rust로 작성된 WASM 모듈을 사용하여 고성능 합의 계산을 수행합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._wasm_module = None
        self._load_wasm_module()

    def _load_wasm_module(self):
        """WASM 모듈 로드 시도"""
        try:
            # wasmer 또는 wasmtime을 통한 WASM 로드
            # 실제 구현에서는 wasm-consensus 모듈 로드
            pass
        except Exception:
            # WASM 사용 불가시 Python 폴백
            self._wasm_module = None

    def _calculate_pairwise_similarity(self, texts: List[str]) -> List[List[float]]:
        """WASM 최적화된 유사도 계산"""
        if self._wasm_module:
            # WASM 모듈 사용
            return self._wasm_module.calculate_similarity(texts)
        else:
            # Python 폴백
            return super()._calculate_pairwise_similarity(texts)
