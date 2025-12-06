"""
AGI Trinity - Agent Adapters
에이전트 어댑터 모듈
"""
from .base import (
    BaseAgentAdapter,
    AgentConfig,
    AgentResponse,
    AgentStatus
)
from .claude_adapter import ClaudeAdapter
from .gemini_adapter import GeminiAdapter
from .openai_adapter import OpenAIAdapter

__all__ = [
    "BaseAgentAdapter",
    "AgentConfig",
    "AgentResponse",
    "AgentStatus",
    "ClaudeAdapter",
    "GeminiAdapter",
    "OpenAIAdapter"
]


def get_adapter(name: str) -> BaseAgentAdapter:
    """
    에이전트 이름으로 어댑터 인스턴스를 반환합니다.

    Args:
        name: 에이전트 이름 (claude, gemini, codex/openai)

    Returns:
        BaseAgentAdapter: 에이전트 어댑터 인스턴스
    """
    adapters = {
        "claude": ClaudeAdapter,
        "gemini": GeminiAdapter,
        "codex": OpenAIAdapter,
        "openai": OpenAIAdapter,
        "gpt4": OpenAIAdapter
    }

    adapter_class = adapters.get(name.lower())
    if adapter_class is None:
        raise ValueError(f"Unknown agent: {name}. Available: {list(adapters.keys())}")

    return adapter_class()
