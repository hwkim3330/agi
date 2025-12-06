#!/usr/bin/env python3
"""
AGI Trinity - Agent Adapter Tests
에이전트 어댑터 테스트
"""
import pytest
import asyncio
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import (
    BaseAgentAdapter,
    AgentConfig,
    AgentResponse,
    AgentStatus,
    ClaudeAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    get_adapter
)


class TestAgentResponse:
    """에이전트 응답 테스트"""

    def test_response_creation(self):
        """응답 객체 생성 테스트"""
        response = AgentResponse(
            agent_name="claude",
            success=True,
            content="Test response",
            latency=1.5,
            confidence=0.95,
            tokens_used=100
        )

        assert response.agent_name == "claude"
        assert response.success is True
        assert response.latency == 1.5
        assert response.confidence == 0.95

    def test_response_to_dict(self):
        """응답 딕셔너리 변환 테스트"""
        response = AgentResponse(
            agent_name="claude",
            success=True,
            content="Test response",
            latency=1.5
        )

        data = response.to_dict()
        assert isinstance(data, dict)
        assert data["agent_name"] == "claude"
        assert "timestamp" in data

    def test_response_with_error(self):
        """에러 응답 테스트"""
        response = AgentResponse(
            agent_name="claude",
            success=False,
            content="",
            latency=0.5,
            error="Connection failed"
        )

        assert response.success is False
        assert response.error == "Connection failed"


class TestAgentConfig:
    """에이전트 설정 테스트"""

    def test_config_creation(self):
        """설정 객체 생성 테스트"""
        config = AgentConfig(
            name="test_agent",
            role="Tester",
            specialty="Testing",
            mode="batch",
            cmd=["echo", "test"],
            timeout_s=60
        )

        assert config.name == "test_agent"
        assert config.timeout_s == 60
        assert config.max_retries == 3  # 기본값

    def test_config_with_strengths(self):
        """강점 목록 포함 설정 테스트"""
        config = AgentConfig(
            name="test_agent",
            role="Tester",
            specialty="Testing",
            mode="batch",
            cmd=["echo"],
            strengths=["Strength 1", "Strength 2"]
        )

        assert len(config.strengths) == 2


class TestAgentStatus:
    """에이전트 상태 테스트"""

    def test_status_enum(self):
        """상태 열거형 테스트"""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.SUCCESS.value == "success"
        assert AgentStatus.ERROR.value == "error"
        assert AgentStatus.TIMEOUT.value == "timeout"


class TestClaudeAdapter:
    """Claude 어댑터 테스트"""

    @pytest.fixture
    def adapter(self):
        return ClaudeAdapter()

    def test_adapter_creation(self, adapter):
        """어댑터 생성 테스트"""
        assert adapter.name == "claude"
        assert adapter.role == "Technical Expert"
        assert adapter.status == AgentStatus.IDLE

    def test_adapter_config(self, adapter):
        """어댑터 설정 테스트"""
        assert adapter.config.name == "claude"
        assert adapter.config.timeout_s == 180
        assert len(adapter.config.strengths) > 0

    def test_adapter_repr(self, adapter):
        """어댑터 표현 테스트"""
        repr_str = repr(adapter)
        assert "ClaudeAdapter" in repr_str
        assert "claude" in repr_str

    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """건강 체크 테스트 (Claude CLI 없으면 False)"""
        # CI 환경에서는 Claude CLI가 없을 수 있음
        result = await adapter.health_check()
        assert isinstance(result, bool)

    def test_metrics_initial(self, adapter):
        """초기 메트릭 테스트"""
        metrics = adapter.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["success_rate"] == 0


class TestGeminiAdapter:
    """Gemini 어댑터 테스트"""

    @pytest.fixture
    def adapter(self):
        return GeminiAdapter()

    def test_adapter_creation(self, adapter):
        """어댑터 생성 테스트"""
        assert adapter.name == "gemini"
        assert adapter.role == "Data Analyst"

    def test_adapter_config(self, adapter):
        """어댑터 설정 테스트"""
        assert adapter.config.specialty == "Research, analysis, fact-checking, data interpretation"


class TestOpenAIAdapter:
    """OpenAI 어댑터 테스트"""

    @pytest.fixture
    def adapter(self):
        return OpenAIAdapter()

    def test_adapter_creation(self, adapter):
        """어댑터 생성 테스트"""
        assert adapter.name == "codex"
        assert adapter.role == "Creative Problem Solver"

    def test_adapter_model(self, adapter):
        """모델 설정 테스트"""
        assert adapter._model == "gpt-4"


class TestGetAdapter:
    """get_adapter 함수 테스트"""

    def test_get_claude(self):
        """Claude 어댑터 가져오기"""
        adapter = get_adapter("claude")
        assert isinstance(adapter, ClaudeAdapter)

    def test_get_gemini(self):
        """Gemini 어댑터 가져오기"""
        adapter = get_adapter("gemini")
        assert isinstance(adapter, GeminiAdapter)

    def test_get_codex(self):
        """Codex 어댑터 가져오기"""
        adapter = get_adapter("codex")
        assert isinstance(adapter, OpenAIAdapter)

    def test_get_openai_alias(self):
        """OpenAI 별칭으로 가져오기"""
        adapter = get_adapter("openai")
        assert isinstance(adapter, OpenAIAdapter)

    def test_get_unknown_agent(self):
        """알 수 없는 에이전트"""
        with pytest.raises(ValueError) as excinfo:
            get_adapter("unknown_agent")
        assert "Unknown agent" in str(excinfo.value)

    def test_case_insensitive(self):
        """대소문자 구분 없이 가져오기"""
        adapter1 = get_adapter("CLAUDE")
        adapter2 = get_adapter("Claude")
        adapter3 = get_adapter("claude")

        assert isinstance(adapter1, ClaudeAdapter)
        assert isinstance(adapter2, ClaudeAdapter)
        assert isinstance(adapter3, ClaudeAdapter)


class TestAgentMetrics:
    """에이전트 메트릭 테스트"""

    @pytest.fixture
    def adapter(self):
        return ClaudeAdapter()

    def test_metrics_update(self, adapter):
        """메트릭 업데이트 테스트"""
        response = AgentResponse(
            agent_name="claude",
            success=True,
            content="Test",
            latency=1.0
        )

        adapter._update_metrics(response)
        metrics = adapter.get_metrics()

        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 1
        assert metrics["average_latency"] == 1.0

    def test_metrics_multiple_updates(self, adapter):
        """다중 메트릭 업데이트 테스트"""
        for i in range(5):
            response = AgentResponse(
                agent_name="claude",
                success=True,
                content="Test",
                latency=float(i + 1)
            )
            adapter._update_metrics(response)

        metrics = adapter.get_metrics()
        assert metrics["total_requests"] == 5
        assert metrics["success_rate"] == 1.0
        assert metrics["average_latency"] == 3.0  # (1+2+3+4+5)/5

    def test_metrics_with_failures(self, adapter):
        """실패 포함 메트릭 테스트"""
        adapter._update_metrics(AgentResponse(
            agent_name="claude", success=True, content="", latency=1.0
        ))
        adapter._update_metrics(AgentResponse(
            agent_name="claude", success=False, content="", latency=1.0
        ))

        metrics = adapter.get_metrics()
        assert metrics["total_requests"] == 2
        assert metrics["successful_requests"] == 1
        assert metrics["failed_requests"] == 1
        assert metrics["success_rate"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
