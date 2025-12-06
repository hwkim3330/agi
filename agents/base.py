#!/usr/bin/env python3
"""
AGI Trinity - Base Agent Adapter
기본 에이전트 어댑터 클래스
"""
import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class AgentStatus(Enum):
    """에이전트 상태"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class AgentResponse:
    """에이전트 응답 데이터 클래스"""
    agent_name: str
    success: bool
    content: str
    latency: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    confidence: float = 1.0
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "agent_name": self.agent_name,
            "success": self.success,
            "content": self.content,
            "latency": self.latency,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "error": self.error,
            "confidence": self.confidence,
            "tokens_used": self.tokens_used
        }


@dataclass
class AgentConfig:
    """에이전트 설정"""
    name: str
    role: str
    specialty: str
    mode: str  # "batch" or "stream"
    cmd: List[str]
    timeout_s: int = 180
    personality: str = ""
    strengths: List[str] = field(default_factory=list)
    max_retries: int = 3
    retry_delay: float = 1.0


class BaseAgentAdapter(ABC):
    """
    기본 에이전트 어댑터 추상 클래스

    모든 AI 에이전트 어댑터는 이 클래스를 상속받아 구현해야 합니다.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.status = AgentStatus.IDLE
        self._process: Optional[asyncio.subprocess.Process] = None
        self._start_time: Optional[float] = None
        self._metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency": 0.0,
            "average_latency": 0.0
        }

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def role(self) -> str:
        return self.config.role

    @property
    def is_running(self) -> bool:
        return self.status == AgentStatus.RUNNING

    @abstractmethod
    async def execute(self, prompt: str) -> AgentResponse:
        """
        프롬프트를 실행하고 응답을 반환합니다.

        Args:
            prompt: 실행할 프롬프트

        Returns:
            AgentResponse: 에이전트 응답
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        에이전트 상태를 확인합니다.

        Returns:
            bool: 에이전트가 정상이면 True
        """
        pass

    async def execute_with_retry(self, prompt: str) -> AgentResponse:
        """
        재시도 로직이 포함된 실행
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = await self.execute(prompt)
                if response.success:
                    return response
                last_error = response.error
            except Exception as e:
                last_error = str(e)

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        return AgentResponse(
            agent_name=self.name,
            success=False,
            content="",
            latency=0,
            error=f"All {self.config.max_retries} retries failed: {last_error}"
        )

    def _update_metrics(self, response: AgentResponse):
        """메트릭 업데이트"""
        self._metrics["total_requests"] += 1
        self._metrics["total_latency"] += response.latency

        if response.success:
            self._metrics["successful_requests"] += 1
        else:
            self._metrics["failed_requests"] += 1

        self._metrics["average_latency"] = (
            self._metrics["total_latency"] / self._metrics["total_requests"]
        )

    def get_metrics(self) -> Dict[str, Any]:
        """현재 메트릭 반환"""
        return {
            **self._metrics,
            "success_rate": (
                self._metrics["successful_requests"] / self._metrics["total_requests"]
                if self._metrics["total_requests"] > 0 else 0
            )
        }

    async def cleanup(self):
        """리소스 정리"""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
        self.status = AgentStatus.IDLE

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, status={self.status.value})>"
