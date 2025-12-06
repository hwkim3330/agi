#!/usr/bin/env python3
"""
AGI Trinity - Real-time Monitor
실시간 모니터링 모듈

시스템 상태, 에이전트 성능, 메트릭 수집을 담당합니다.
"""
import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from collections import deque
import statistics


@dataclass
class SystemMetrics:
    """시스템 메트릭"""
    timestamp: datetime = field(default_factory=datetime.now)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    active_agents: int = 0
    total_tokens: int = 0


@dataclass
class AgentMetrics:
    """에이전트별 메트릭"""
    agent_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency: float = 0.0
    total_tokens: int = 0
    last_active: Optional[datetime] = None
    error_rate: float = 0.0
    uptime_ratio: float = 1.0


@dataclass
class RequestLog:
    """요청 로그"""
    request_id: str
    timestamp: datetime
    prompt_preview: str
    agents_used: List[str]
    strategy: str
    total_latency: float
    success: bool
    consensus_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Monitor:
    """
    실시간 모니터

    시스템 전체 상태와 개별 에이전트 성능을 추적합니다.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        log_dir: Optional[Path] = None
    ):
        self.config = config or {}
        self.log_dir = log_dir or Path.home() / ".trinity" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 메트릭 저장소
        self._system_metrics = SystemMetrics()
        self._agent_metrics: Dict[str, AgentMetrics] = {}
        self._request_logs: deque = deque(maxlen=10000)
        self._latency_history: deque = deque(maxlen=1000)

        # 알림 콜백
        self._alert_callbacks: List[Callable] = []

        # 시계열 데이터 (시각화용)
        self._timeseries: Dict[str, deque] = {
            "requests_per_minute": deque(maxlen=60),
            "latency_per_minute": deque(maxlen=60),
            "error_rate_per_minute": deque(maxlen=60)
        }

        # 마지막 분 집계
        self._last_minute_counts = {
            "requests": 0,
            "errors": 0,
            "total_latency": 0.0
        }
        self._last_minute_timestamp = datetime.now()

    def record_request(
        self,
        request_id: str,
        prompt: str,
        agents_used: List[str],
        strategy: str,
        responses: List[Dict[str, Any]],
        consensus_confidence: float
    ):
        """
        요청 결과를 기록합니다.
        """
        timestamp = datetime.now()

        # 전체 레이턴시 계산
        latencies = [r.get("latency", 0) for r in responses]
        total_latency = max(latencies) if latencies else 0
        success = any(r.get("success", False) for r in responses)

        # 요청 로그 저장
        log = RequestLog(
            request_id=request_id,
            timestamp=timestamp,
            prompt_preview=prompt[:100],
            agents_used=agents_used,
            strategy=strategy,
            total_latency=total_latency,
            success=success,
            consensus_confidence=consensus_confidence
        )
        self._request_logs.append(log)

        # 시스템 메트릭 업데이트
        self._system_metrics.total_requests += 1
        if success:
            self._system_metrics.successful_requests += 1
        else:
            self._system_metrics.failed_requests += 1

        self._latency_history.append(total_latency)
        self._update_latency_percentiles()

        # 에이전트별 메트릭 업데이트
        for response in responses:
            agent_name = response.get("agent_name", "unknown")
            self._update_agent_metrics(
                agent_name,
                response.get("success", False),
                response.get("latency", 0),
                response.get("tokens_used", 0)
            )

        # 분당 집계 업데이트
        self._update_per_minute_stats(success, total_latency)

        # 알림 확인
        self._check_alerts()

    def _update_agent_metrics(
        self,
        agent_name: str,
        success: bool,
        latency: float,
        tokens: int
    ):
        """에이전트 메트릭 업데이트"""
        if agent_name not in self._agent_metrics:
            self._agent_metrics[agent_name] = AgentMetrics(agent_name)

        metrics = self._agent_metrics[agent_name]
        metrics.total_requests += 1
        metrics.total_tokens += tokens
        metrics.last_active = datetime.now()

        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1

        # 평균 레이턴시 (지수 이동 평균)
        alpha = 0.2
        metrics.avg_latency = alpha * latency + (1 - alpha) * metrics.avg_latency

        # 에러율 계산
        metrics.error_rate = (
            metrics.failed_requests / metrics.total_requests
            if metrics.total_requests > 0 else 0
        )

    def _update_latency_percentiles(self):
        """레이턴시 백분위 업데이트"""
        if not self._latency_history:
            return

        sorted_latencies = sorted(self._latency_history)
        n = len(sorted_latencies)

        self._system_metrics.avg_latency = statistics.mean(sorted_latencies)
        self._system_metrics.p95_latency = sorted_latencies[int(n * 0.95)] if n >= 20 else 0
        self._system_metrics.p99_latency = sorted_latencies[int(n * 0.99)] if n >= 100 else 0

    def _update_per_minute_stats(self, success: bool, latency: float):
        """분당 통계 업데이트"""
        now = datetime.now()

        # 1분이 지났으면 시계열 데이터에 추가하고 리셋
        if now - self._last_minute_timestamp >= timedelta(minutes=1):
            rpm = self._last_minute_counts["requests"]
            avg_lat = (
                self._last_minute_counts["total_latency"] / rpm
                if rpm > 0 else 0
            )
            error_rate = (
                self._last_minute_counts["errors"] / rpm
                if rpm > 0 else 0
            )

            self._timeseries["requests_per_minute"].append({
                "timestamp": self._last_minute_timestamp.isoformat(),
                "value": rpm
            })
            self._timeseries["latency_per_minute"].append({
                "timestamp": self._last_minute_timestamp.isoformat(),
                "value": avg_lat
            })
            self._timeseries["error_rate_per_minute"].append({
                "timestamp": self._last_minute_timestamp.isoformat(),
                "value": error_rate
            })

            # 리셋
            self._last_minute_counts = {
                "requests": 0,
                "errors": 0,
                "total_latency": 0.0
            }
            self._last_minute_timestamp = now

        # 현재 분 집계
        self._last_minute_counts["requests"] += 1
        self._last_minute_counts["total_latency"] += latency
        if not success:
            self._last_minute_counts["errors"] += 1

    def _check_alerts(self):
        """알림 조건 확인"""
        alerts = []

        # 에러율 알림
        error_rate = (
            self._system_metrics.failed_requests / self._system_metrics.total_requests
            if self._system_metrics.total_requests > 0 else 0
        )
        if error_rate > 0.3:
            alerts.append({
                "type": "high_error_rate",
                "message": f"Error rate is {error_rate:.1%}",
                "severity": "warning"
            })

        # 레이턴시 알림
        if self._system_metrics.p95_latency > 60:
            alerts.append({
                "type": "high_latency",
                "message": f"P95 latency is {self._system_metrics.p95_latency:.1f}s",
                "severity": "warning"
            })

        # 에이전트 다운 알림
        for name, metrics in self._agent_metrics.items():
            if metrics.error_rate > 0.5:
                alerts.append({
                    "type": "agent_degraded",
                    "message": f"Agent {name} error rate: {metrics.error_rate:.1%}",
                    "severity": "critical"
                })

        # 콜백 실행
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception:
                    pass

    def add_alert_callback(self, callback: Callable):
        """알림 콜백 추가"""
        self._alert_callbacks.append(callback)

    def get_system_metrics(self) -> Dict[str, Any]:
        """시스템 메트릭 조회"""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_requests": self._system_metrics.total_requests,
            "successful_requests": self._system_metrics.successful_requests,
            "failed_requests": self._system_metrics.failed_requests,
            "success_rate": (
                self._system_metrics.successful_requests / self._system_metrics.total_requests
                if self._system_metrics.total_requests > 0 else 0
            ),
            "avg_latency": self._system_metrics.avg_latency,
            "p95_latency": self._system_metrics.p95_latency,
            "p99_latency": self._system_metrics.p99_latency,
            "active_agents": len(self._agent_metrics),
            "total_tokens": self._system_metrics.total_tokens
        }

    def get_agent_metrics(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """에이전트 메트릭 조회"""
        if agent_name:
            metrics = self._agent_metrics.get(agent_name)
            if metrics:
                return asdict(metrics)
            return {}

        return {name: asdict(m) for name, m in self._agent_metrics.items()}

    def get_timeseries(self, metric_name: str) -> List[Dict[str, Any]]:
        """시계열 데이터 조회"""
        return list(self._timeseries.get(metric_name, []))

    def get_recent_logs(self, count: int = 100) -> List[Dict[str, Any]]:
        """최근 요청 로그 조회"""
        logs = list(self._request_logs)[-count:]
        return [
            {
                "request_id": log.request_id,
                "timestamp": log.timestamp.isoformat(),
                "prompt_preview": log.prompt_preview,
                "agents_used": log.agents_used,
                "strategy": log.strategy,
                "total_latency": log.total_latency,
                "success": log.success,
                "consensus_confidence": log.consensus_confidence
            }
            for log in logs
        ]

    def get_dashboard_data(self) -> Dict[str, Any]:
        """대시보드용 종합 데이터"""
        return {
            "system": self.get_system_metrics(),
            "agents": self.get_agent_metrics(),
            "timeseries": {
                "rpm": list(self._timeseries["requests_per_minute"]),
                "latency": list(self._timeseries["latency_per_minute"]),
                "errors": list(self._timeseries["error_rate_per_minute"])
            },
            "recent_logs": self.get_recent_logs(20)
        }

    async def export_logs(self, filepath: Optional[Path] = None) -> Path:
        """로그 내보내기"""
        if filepath is None:
            filepath = self.log_dir / f"trinity_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            "exported_at": datetime.now().isoformat(),
            "system_metrics": self.get_system_metrics(),
            "agent_metrics": self.get_agent_metrics(),
            "request_logs": self.get_recent_logs(10000)
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        return filepath


class RealTimeMonitor(Monitor):
    """
    실시간 모니터링 확장

    WebSocket 또는 SSE를 통한 실시간 업데이트를 지원합니다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subscribers: List[Callable] = []
        self._running = False

    def subscribe(self, callback: Callable):
        """실시간 업데이트 구독"""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable):
        """구독 해제"""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    async def _notify_subscribers(self, data: Dict[str, Any]):
        """구독자에게 알림"""
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception:
                pass

    def record_request(self, *args, **kwargs):
        """요청 기록 및 실시간 알림"""
        super().record_request(*args, **kwargs)

        # 구독자에게 알림 (비동기로 실행)
        if self._subscribers:
            data = self.get_dashboard_data()
            asyncio.create_task(self._notify_subscribers(data))

    async def start_periodic_broadcast(self, interval: float = 5.0):
        """주기적 브로드캐스트 시작"""
        self._running = True
        while self._running:
            await asyncio.sleep(interval)
            data = self.get_dashboard_data()
            await self._notify_subscribers(data)

    def stop(self):
        """모니터링 중지"""
        self._running = False
