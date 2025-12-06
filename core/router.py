#!/usr/bin/env python3
"""
AGI Trinity - Request Router
요청 라우터 모듈

에이전트 선택, 로드 밸런싱, 요청 분배를 담당합니다.
"""
import asyncio
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from datetime import datetime, timedelta


class RoutingStrategy(Enum):
    """라우팅 전략"""
    ROUND_ROBIN = "round_robin"     # 순차 분배
    LEAST_LOADED = "least_loaded"    # 부하 기반
    SPECIALTY = "specialty"          # 전문성 기반
    RANDOM = "random"                # 무작위
    HYBRID = "hybrid"                # 복합 전략


@dataclass
class AgentHealth:
    """에이전트 건강 상태"""
    agent_name: str
    is_healthy: bool = True
    last_check: datetime = field(default_factory=datetime.now)
    success_rate: float = 1.0
    avg_latency: float = 0.0
    current_load: int = 0
    max_load: int = 10
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class RoutingDecision:
    """라우팅 결정"""
    selected_agents: List[str]
    strategy_used: str
    reasoning: str
    scores: Dict[str, float] = field(default_factory=dict)


class RequestRouter:
    """
    요청 라우터

    프롬프트 분석 및 에이전트 선택을 담당합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._agent_health: Dict[str, AgentHealth] = {}
        self._specialty_keywords: Dict[str, Set[str]] = {
            "claude": {
                "code", "debug", "review", "security", "architecture",
                "implementation", "bug", "error", "fix", "optimize",
                "refactor", "python", "javascript", "rust", "algorithm"
            },
            "gemini": {
                "research", "data", "analysis", "fact", "verify",
                "statistics", "trend", "market", "study", "evidence",
                "comparison", "report", "summary", "insight"
            },
            "codex": {
                "creative", "idea", "brainstorm", "innovative", "strategy",
                "design", "concept", "vision", "story", "narrative",
                "solution", "alternative", "think", "imagine"
            }
        }
        self._request_history: List[Dict[str, Any]] = []

    def route(
        self,
        prompt: str,
        available_agents: List[str],
        strategy: RoutingStrategy = RoutingStrategy.HYBRID,
        max_agents: Optional[int] = None
    ) -> RoutingDecision:
        """
        프롬프트를 분석하고 최적의 에이전트를 선택합니다.

        Args:
            prompt: 입력 프롬프트
            available_agents: 사용 가능한 에이전트 목록
            strategy: 라우팅 전략
            max_agents: 최대 에이전트 수

        Returns:
            RoutingDecision: 라우팅 결정
        """
        if not available_agents:
            return RoutingDecision(
                selected_agents=[],
                strategy_used=strategy.value,
                reasoning="No agents available"
            )

        # 건강한 에이전트만 필터링
        healthy_agents = [
            agent for agent in available_agents
            if self._is_agent_healthy(agent)
        ]

        if not healthy_agents:
            # 모든 에이전트가 비정상이면 그래도 시도
            healthy_agents = available_agents

        # 전략에 따른 선택
        strategy_handlers = {
            RoutingStrategy.ROUND_ROBIN: self._route_round_robin,
            RoutingStrategy.LEAST_LOADED: self._route_least_loaded,
            RoutingStrategy.SPECIALTY: self._route_specialty,
            RoutingStrategy.RANDOM: self._route_random,
            RoutingStrategy.HYBRID: self._route_hybrid
        }

        handler = strategy_handlers.get(strategy, self._route_hybrid)
        decision = handler(prompt, healthy_agents, max_agents)

        # 요청 기록
        self._request_history.append({
            "timestamp": datetime.now(),
            "prompt_preview": prompt[:100],
            "decision": decision
        })

        return decision

    def _route_round_robin(
        self,
        prompt: str,
        agents: List[str],
        max_agents: Optional[int]
    ) -> RoutingDecision:
        """순차 분배"""
        max_agents = max_agents or len(agents)
        selected = agents[:max_agents]

        return RoutingDecision(
            selected_agents=selected,
            strategy_used="round_robin",
            reasoning=f"Sequential selection of {len(selected)} agents"
        )

    def _route_least_loaded(
        self,
        prompt: str,
        agents: List[str],
        max_agents: Optional[int]
    ) -> RoutingDecision:
        """부하 기반 라우팅"""
        max_agents = max_agents or len(agents)

        # 부하 기준 정렬
        sorted_agents = sorted(
            agents,
            key=lambda a: self._agent_health.get(a, AgentHealth(a)).current_load
        )

        selected = sorted_agents[:max_agents]
        scores = {
            agent: 1.0 - (self._agent_health.get(agent, AgentHealth(agent)).current_load / 10)
            for agent in agents
        }

        return RoutingDecision(
            selected_agents=selected,
            strategy_used="least_loaded",
            reasoning=f"Selected {len(selected)} agents with lowest load",
            scores=scores
        )

    def _route_specialty(
        self,
        prompt: str,
        agents: List[str],
        max_agents: Optional[int]
    ) -> RoutingDecision:
        """전문성 기반 라우팅"""
        max_agents = max_agents or len(agents)

        # 프롬프트에서 키워드 추출
        prompt_words = set(prompt.lower().split())

        # 각 에이전트의 관련성 점수 계산
        scores = {}
        for agent in agents:
            keywords = self._specialty_keywords.get(agent, set())
            overlap = len(prompt_words & keywords)
            scores[agent] = overlap / (len(keywords) + 1)

        # 점수 기준 정렬
        sorted_agents = sorted(agents, key=lambda a: scores[a], reverse=True)
        selected = sorted_agents[:max_agents]

        reasoning = ", ".join([f"{a}:{scores[a]:.2f}" for a in selected])

        return RoutingDecision(
            selected_agents=selected,
            strategy_used="specialty",
            reasoning=f"Specialty match: {reasoning}",
            scores=scores
        )

    def _route_random(
        self,
        prompt: str,
        agents: List[str],
        max_agents: Optional[int]
    ) -> RoutingDecision:
        """무작위 라우팅"""
        max_agents = max_agents or len(agents)
        selected = random.sample(agents, min(max_agents, len(agents)))

        return RoutingDecision(
            selected_agents=selected,
            strategy_used="random",
            reasoning=f"Random selection of {len(selected)} agents"
        )

    def _route_hybrid(
        self,
        prompt: str,
        agents: List[str],
        max_agents: Optional[int]
    ) -> RoutingDecision:
        """복합 전략 라우팅"""
        max_agents = max_agents or len(agents)

        # 전문성 점수
        specialty_decision = self._route_specialty(prompt, agents, None)

        # 부하 점수
        load_scores = {
            agent: 1.0 - (self._agent_health.get(agent, AgentHealth(agent)).current_load / 10)
            for agent in agents
        }

        # 성공률 점수
        success_scores = {
            agent: self._agent_health.get(agent, AgentHealth(agent)).success_rate
            for agent in agents
        }

        # 복합 점수 계산
        combined_scores = {}
        for agent in agents:
            specialty = specialty_decision.scores.get(agent, 0.5)
            load = load_scores.get(agent, 0.5)
            success = success_scores.get(agent, 0.5)

            combined_scores[agent] = (
                specialty * 0.4 +
                load * 0.3 +
                success * 0.3
            )

        # 정렬 및 선택
        sorted_agents = sorted(agents, key=lambda a: combined_scores[a], reverse=True)
        selected = sorted_agents[:max_agents]

        return RoutingDecision(
            selected_agents=selected,
            strategy_used="hybrid",
            reasoning=f"Hybrid scoring: specialty(40%), load(30%), success(30%)",
            scores=combined_scores
        )

    def _is_agent_healthy(self, agent_name: str) -> bool:
        """에이전트 건강 상태 확인"""
        health = self._agent_health.get(agent_name)
        if health is None:
            return True  # 정보 없으면 건강하다고 가정

        # 최근 확인이 5분 이상 지났으면 건강하다고 가정
        if datetime.now() - health.last_check > timedelta(minutes=5):
            return True

        return health.is_healthy and health.error_count < 3

    def update_agent_health(
        self,
        agent_name: str,
        success: bool,
        latency: float,
        error: Optional[str] = None
    ):
        """에이전트 건강 상태 업데이트"""
        if agent_name not in self._agent_health:
            self._agent_health[agent_name] = AgentHealth(agent_name)

        health = self._agent_health[agent_name]
        health.last_check = datetime.now()

        if success:
            # 성공: 에러 카운트 감소, 성공률 증가
            health.error_count = max(0, health.error_count - 1)
            health.success_rate = min(1.0, health.success_rate * 0.9 + 0.1)
        else:
            # 실패: 에러 카운트 증가, 성공률 감소
            health.error_count += 1
            health.success_rate = max(0.0, health.success_rate * 0.9)
            health.last_error = error

        # 평균 레이턴시 업데이트 (지수 이동 평균)
        alpha = 0.2
        health.avg_latency = alpha * latency + (1 - alpha) * health.avg_latency

        # 건강 상태 결정
        health.is_healthy = (
            health.error_count < 5 and
            health.success_rate > 0.3 and
            health.avg_latency < 120
        )

    def get_agent_health(self, agent_name: str) -> Optional[AgentHealth]:
        """에이전트 건강 상태 조회"""
        return self._agent_health.get(agent_name)

    def get_all_health(self) -> Dict[str, AgentHealth]:
        """모든 에이전트 건강 상태 조회"""
        return self._agent_health.copy()

    def increment_load(self, agent_name: str):
        """에이전트 부하 증가"""
        if agent_name not in self._agent_health:
            self._agent_health[agent_name] = AgentHealth(agent_name)
        self._agent_health[agent_name].current_load += 1

    def decrement_load(self, agent_name: str):
        """에이전트 부하 감소"""
        if agent_name in self._agent_health:
            self._agent_health[agent_name].current_load = max(
                0, self._agent_health[agent_name].current_load - 1
            )


class AdaptiveRouter(RequestRouter):
    """
    적응형 라우터

    실시간 학습을 통해 라우팅 전략을 개선합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._feedback_history: List[Dict[str, Any]] = []
        self._learned_weights: Dict[str, Dict[str, float]] = {}

    def record_feedback(
        self,
        prompt: str,
        agent_name: str,
        quality_score: float,
        user_selected: bool = False
    ):
        """
        피드백 기록 (온라인 학습용)
        """
        self._feedback_history.append({
            "timestamp": datetime.now(),
            "prompt_keywords": set(prompt.lower().split()),
            "agent": agent_name,
            "quality": quality_score,
            "user_selected": user_selected
        })

        # 학습 업데이트
        self._update_learned_weights(prompt, agent_name, quality_score)

    def _update_learned_weights(
        self,
        prompt: str,
        agent_name: str,
        quality_score: float
    ):
        """
        온라인 학습: 가중치 업데이트
        """
        # 프롬프트에서 주요 키워드 추출
        keywords = set(prompt.lower().split())

        for keyword in keywords:
            if keyword not in self._learned_weights:
                self._learned_weights[keyword] = {}

            if agent_name not in self._learned_weights[keyword]:
                self._learned_weights[keyword][agent_name] = 0.5

            # 지수 이동 평균으로 가중치 업데이트
            current = self._learned_weights[keyword][agent_name]
            self._learned_weights[keyword][agent_name] = (
                current * 0.8 + quality_score * 0.2
            )

    def _route_hybrid(
        self,
        prompt: str,
        agents: List[str],
        max_agents: Optional[int]
    ) -> RoutingDecision:
        """
        학습된 가중치를 활용한 복합 라우팅
        """
        base_decision = super()._route_hybrid(prompt, agents, max_agents)

        # 학습된 가중치 적용
        prompt_keywords = set(prompt.lower().split())
        learned_scores = {agent: 0.0 for agent in agents}

        for keyword in prompt_keywords:
            if keyword in self._learned_weights:
                for agent in agents:
                    if agent in self._learned_weights[keyword]:
                        learned_scores[agent] += self._learned_weights[keyword][agent]

        # 정규화
        if learned_scores:
            max_score = max(learned_scores.values()) or 1
            learned_scores = {k: v / max_score for k, v in learned_scores.items()}

        # 기본 점수와 학습 점수 결합
        combined = {}
        for agent in agents:
            base = base_decision.scores.get(agent, 0.5)
            learned = learned_scores.get(agent, 0.5)
            combined[agent] = base * 0.6 + learned * 0.4

        sorted_agents = sorted(agents, key=lambda a: combined[a], reverse=True)
        selected = sorted_agents[:max_agents or len(agents)]

        return RoutingDecision(
            selected_agents=selected,
            strategy_used="adaptive_hybrid",
            reasoning="Hybrid with online learning adaptation",
            scores=combined
        )
