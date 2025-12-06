#!/usr/bin/env python3
"""
AGI Trinity - Consensus Engine Tests
합의 엔진 테스트
"""
import pytest
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.consensus import (
    ConsensusEngine,
    ConsensusStrategy,
    ConsensusResult
)


class TestConsensusEngine:
    """합의 엔진 테스트 클래스"""

    @pytest.fixture
    def engine(self):
        """합의 엔진 인스턴스"""
        return ConsensusEngine()

    @pytest.fixture
    def sample_responses(self):
        """샘플 응답 데이터"""
        return [
            {
                "agent_name": "claude",
                "content": "This is a technical analysis of the problem. The solution involves implementing a microservice architecture.",
                "success": True,
                "latency": 1.5,
                "confidence": 0.95,
                "metadata": {"role": "Technical Expert"}
            },
            {
                "agent_name": "gemini",
                "content": "Research shows that microservices provide better scalability. Data supports this approach.",
                "success": True,
                "latency": 2.0,
                "confidence": 0.88,
                "metadata": {"role": "Data Analyst"}
            },
            {
                "agent_name": "codex",
                "content": "Creative solution: Consider a hybrid approach combining microservices with serverless functions for optimal flexibility.",
                "success": True,
                "latency": 1.8,
                "confidence": 0.82,
                "metadata": {"role": "Creative Problem Solver"}
            }
        ]

    def test_vote_consensus(self, engine, sample_responses):
        """투표 기반 합의 테스트"""
        result = engine.calculate_consensus(sample_responses, "vote")

        assert isinstance(result, ConsensusResult)
        assert result.strategy == "vote"
        assert result.confidence > 0
        assert len(result.individual_scores) == 3
        assert result.content  # 콘텐츠가 비어있지 않아야 함

    def test_synthesis_consensus(self, engine, sample_responses):
        """통합 합의 테스트"""
        result = engine.calculate_consensus(sample_responses, "synthesis")

        assert result.strategy == "synthesis"
        assert "claude" in result.content.lower() or "gemini" in result.content.lower()
        assert result.confidence == 1.0  # 모든 에이전트 성공

    def test_fanout_consensus(self, engine, sample_responses):
        """개별 표시 테스트"""
        result = engine.calculate_consensus(sample_responses, "fanout")

        assert result.strategy == "fanout"
        assert result.confidence == 1.0

    def test_semantic_consensus(self, engine, sample_responses):
        """의미 기반 합의 테스트"""
        result = engine.calculate_consensus(sample_responses, "semantic")

        assert result.strategy == "semantic"
        assert result.confidence > 0

    def test_empty_responses(self, engine):
        """빈 응답 처리 테스트"""
        result = engine.calculate_consensus([], "vote")

        assert result.confidence == 0.0
        assert "No responses" in result.content or "No responses" in result.reasoning

    def test_all_failed_responses(self, engine):
        """모든 에이전트 실패 시 테스트"""
        failed_responses = [
            {"agent_name": "claude", "content": "", "success": False, "latency": 0, "confidence": 0},
            {"agent_name": "gemini", "content": "", "success": False, "latency": 0, "confidence": 0}
        ]
        result = engine.calculate_consensus(failed_responses, "vote")

        assert result.confidence == 0.0

    def test_partial_success(self, engine, sample_responses):
        """일부 성공 시 테스트"""
        sample_responses[0]["success"] = False
        result = engine.calculate_consensus(sample_responses, "vote")

        # 2/3 성공
        assert result.confidence == pytest.approx(2/3, rel=0.01)

    def test_weighted_consensus(self, engine, sample_responses):
        """가중치 기반 합의 테스트"""
        result = engine.calculate_consensus(sample_responses, "weighted")

        assert result.strategy == "weighted"
        # Claude에 가중치가 높으므로 선택될 가능성이 높음
        assert result.content

    def test_jaccard_similarity(self, engine):
        """자카드 유사도 테스트"""
        text1 = "hello world programming"
        text2 = "hello programming python"

        sim = engine._jaccard_similarity(text1, text2)
        assert 0 <= sim <= 1
        assert sim > 0  # 공통 단어가 있음

    def test_identical_texts_similarity(self, engine):
        """동일 텍스트 유사도 테스트"""
        text = "identical text here"
        sim = engine._jaccard_similarity(text, text)
        assert sim == 1.0

    def test_completely_different_texts(self, engine):
        """완전히 다른 텍스트 유사도 테스트"""
        text1 = "apple banana cherry"
        text2 = "xyz uvw rst"
        sim = engine._jaccard_similarity(text1, text2)
        assert sim == 0.0

    def test_extract_common_themes(self, engine, sample_responses):
        """공통 주제 추출 테스트"""
        texts = [r["content"] for r in sample_responses]
        themes = engine._extract_common_themes(texts)

        assert isinstance(themes, str)
        # microservices가 공통 주제로 나올 가능성이 높음


class TestConsensusScoring:
    """합의 점수 계산 테스트"""

    @pytest.fixture
    def engine(self):
        return ConsensusEngine()

    def test_response_score_success(self, engine):
        """성공 응답 점수 테스트"""
        response = {
            "content": "A" * 500,  # 적당한 길이
            "latency": 1.0,
            "confidence": 0.9,
            "success": True
        }
        score = engine._calculate_response_score(response)
        assert 0 <= score <= 1
        assert score > 0.5  # 좋은 응답이므로 0.5 이상

    def test_response_score_short_content(self, engine):
        """짧은 콘텐츠 점수 테스트"""
        response = {
            "content": "short",
            "latency": 1.0,
            "confidence": 0.9,
            "success": True
        }
        score = engine._calculate_response_score(response)
        assert score < 0.8  # 짧은 콘텐츠는 감점

    def test_response_score_high_latency(self, engine):
        """높은 레이턴시 점수 테스트"""
        response = {
            "content": "A" * 500,
            "latency": 100.0,  # 매우 느림
            "confidence": 0.9,
            "success": True
        }
        score = engine._calculate_response_score(response)
        # 레이턴시가 높으면 점수 감점


class TestConsensusStrategies:
    """합의 전략 테스트"""

    def test_strategy_enum(self):
        """전략 열거형 테스트"""
        assert ConsensusStrategy.VOTE.value == "vote"
        assert ConsensusStrategy.SYNTHESIS.value == "synthesis"
        assert ConsensusStrategy.FANOUT.value == "fanout"
        assert ConsensusStrategy.WEIGHTED.value == "weighted"
        assert ConsensusStrategy.SEMANTIC.value == "semantic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
