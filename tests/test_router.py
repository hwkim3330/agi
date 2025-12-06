#!/usr/bin/env python3
"""
AGI Trinity - Request Router Tests
요청 라우터 테스트
"""
import pytest
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.router import (
    RequestRouter,
    AdaptiveRouter,
    RoutingStrategy,
    RoutingDecision,
    AgentHealth
)


class TestRequestRouter:
    """요청 라우터 테스트 클래스"""

    @pytest.fixture
    def router(self):
        return RequestRouter()

    @pytest.fixture
    def available_agents(self):
        return ["claude", "gemini", "codex"]

    def test_route_round_robin(self, router, available_agents):
        """순차 분배 테스트"""
        decision = router.route(
            "test prompt",
            available_agents,
            RoutingStrategy.ROUND_ROBIN
        )

        assert isinstance(decision, RoutingDecision)
        assert len(decision.selected_agents) == len(available_agents)
        assert decision.strategy_used == "round_robin"

    def test_route_random(self, router, available_agents):
        """무작위 라우팅 테스트"""
        decision = router.route(
            "test prompt",
            available_agents,
            RoutingStrategy.RANDOM
        )

        assert len(decision.selected_agents) <= len(available_agents)
        assert decision.strategy_used == "random"

    def test_route_specialty_code(self, router, available_agents):
        """전문성 기반 라우팅 - 코드 관련 테스트"""
        decision = router.route(
            "review this code and debug the error",
            available_agents,
            RoutingStrategy.SPECIALTY
        )

        # Claude가 코드 관련 키워드에 높은 점수를 받아야 함
        assert "claude" in decision.selected_agents[:2]
        assert decision.strategy_used == "specialty"

    def test_route_specialty_research(self, router, available_agents):
        """전문성 기반 라우팅 - 연구 관련 테스트"""
        decision = router.route(
            "research the market trends and analyze data",
            available_agents,
            RoutingStrategy.SPECIALTY
        )

        # Gemini가 연구/분석 키워드에 높은 점수를 받아야 함
        assert "gemini" in decision.selected_agents[:2]

    def test_route_specialty_creative(self, router, available_agents):
        """전문성 기반 라우팅 - 창의성 관련 테스트"""
        decision = router.route(
            "brainstorm creative ideas and innovative solutions",
            available_agents,
            RoutingStrategy.SPECIALTY
        )

        # Codex가 창의성 키워드에 높은 점수를 받아야 함
        assert "codex" in decision.selected_agents[:2]

    def test_route_hybrid(self, router, available_agents):
        """복합 전략 라우팅 테스트"""
        decision = router.route(
            "optimize the code performance",
            available_agents,
            RoutingStrategy.HYBRID
        )

        assert decision.strategy_used == "hybrid"
        assert len(decision.scores) == len(available_agents)

    def test_route_max_agents(self, router, available_agents):
        """최대 에이전트 수 제한 테스트"""
        decision = router.route(
            "test prompt",
            available_agents,
            RoutingStrategy.ROUND_ROBIN,
            max_agents=2
        )

        assert len(decision.selected_agents) == 2

    def test_route_empty_agents(self, router):
        """빈 에이전트 목록 테스트"""
        decision = router.route(
            "test prompt",
            [],
            RoutingStrategy.ROUND_ROBIN
        )

        assert len(decision.selected_agents) == 0
        assert "No agents" in decision.reasoning


class TestAgentHealth:
    """에이전트 건강 상태 테스트"""

    @pytest.fixture
    def router(self):
        return RequestRouter()

    def test_update_health_success(self, router):
        """성공 응답 후 건강 상태 업데이트"""
        router.update_agent_health("claude", True, 1.5)

        health = router.get_agent_health("claude")
        assert health is not None
        assert health.is_healthy
        assert health.error_count == 0

    def test_update_health_failure(self, router):
        """실패 응답 후 건강 상태 업데이트"""
        router.update_agent_health("claude", False, 1.5, "Connection timeout")

        health = router.get_agent_health("claude")
        assert health.error_count == 1
        assert health.last_error == "Connection timeout"

    def test_health_degradation(self, router):
        """연속 실패 시 건강 상태 저하"""
        for _ in range(5):
            router.update_agent_health("claude", False, 1.5)

        health = router.get_agent_health("claude")
        assert not health.is_healthy
        assert health.error_count >= 5

    def test_health_recovery(self, router):
        """실패 후 회복"""
        # 먼저 실패
        router.update_agent_health("claude", False, 1.5)
        router.update_agent_health("claude", False, 1.5)

        # 그 다음 성공
        for _ in range(5):
            router.update_agent_health("claude", True, 1.0)

        health = router.get_agent_health("claude")
        assert health.is_healthy
        assert health.error_count < 3

    def test_load_tracking(self, router):
        """부하 추적 테스트"""
        router.increment_load("claude")
        router.increment_load("claude")

        health = router.get_agent_health("claude")
        assert health.current_load == 2

        router.decrement_load("claude")
        health = router.get_agent_health("claude")
        assert health.current_load == 1


class TestAdaptiveRouter:
    """적응형 라우터 테스트"""

    @pytest.fixture
    def router(self):
        return AdaptiveRouter()

    @pytest.fixture
    def available_agents(self):
        return ["claude", "gemini", "codex"]

    def test_record_feedback(self, router):
        """피드백 기록 테스트"""
        router.record_feedback(
            "code review request",
            "claude",
            0.9,
            user_selected=True
        )

        # 학습된 가중치가 업데이트되어야 함
        assert len(router._feedback_history) == 1

    def test_adaptive_routing(self, router, available_agents):
        """적응형 라우팅 테스트"""
        # 피드백 학습
        for _ in range(10):
            router.record_feedback("code debug", "claude", 0.95)
            router.record_feedback("data analysis", "gemini", 0.90)
            router.record_feedback("creative idea", "codex", 0.85)

        # 코드 관련 쿼리는 Claude를 선호해야 함
        decision = router.route(
            "debug this code",
            available_agents,
            RoutingStrategy.HYBRID
        )

        # 학습 후에는 Claude가 높은 순위에 있어야 함
        assert "claude" in decision.selected_agents[:2]

    def test_learning_weight_update(self, router):
        """학습 가중치 업데이트 테스트"""
        router.record_feedback("python programming", "claude", 0.95)

        # 'python' 키워드에 대한 가중치가 생성되어야 함
        assert "python" in router._learned_weights or "programming" in router._learned_weights


class TestRoutingDecision:
    """라우팅 결정 테스트"""

    def test_routing_decision_dataclass(self):
        """RoutingDecision 데이터클래스 테스트"""
        decision = RoutingDecision(
            selected_agents=["claude", "gemini"],
            strategy_used="hybrid",
            reasoning="Test reasoning",
            scores={"claude": 0.9, "gemini": 0.8}
        )

        assert len(decision.selected_agents) == 2
        assert decision.strategy_used == "hybrid"
        assert decision.scores["claude"] == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
