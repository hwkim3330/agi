#!/usr/bin/env python3
"""
AGI Trinity - Continual Learning Engine
지속학습 엔진

LFM2-VL 모델의 지속적인 학습과 적응을 관리합니다.
"""
import asyncio
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque
import hashlib
import torch


@dataclass
class Experience:
    """학습 경험 데이터"""
    id: str
    timestamp: datetime
    prompt: str
    response: str
    has_image: bool = False
    image_hash: Optional[str] = None
    quality_score: Optional[float] = None
    user_feedback: Optional[str] = None
    correction: Optional[str] = None
    domain: str = "general"
    difficulty: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "prompt": self.prompt,
            "response": self.response,
            "has_image": self.has_image,
            "image_hash": self.image_hash,
            "quality_score": self.quality_score,
            "user_feedback": self.user_feedback,
            "correction": self.correction,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class LearningMetrics:
    """학습 메트릭"""
    total_experiences: int = 0
    high_quality_count: int = 0
    low_quality_count: int = 0
    average_quality: float = 0.0
    domains_learned: Dict[str, int] = field(default_factory=dict)
    learning_rate_history: List[float] = field(default_factory=list)
    last_training_time: Optional[datetime] = None
    training_count: int = 0


class ExperienceReplayBuffer:
    """
    경험 재생 버퍼 (Experience Replay Buffer)

    중요한 경험을 우선적으로 샘플링하는 우선순위 경험 재생 구현
    """

    def __init__(
        self,
        max_size: int = 10000,
        alpha: float = 0.6,  # 우선순위 지수
        beta: float = 0.4,   # 중요도 샘플링 보정
        beta_increment: float = 0.001
    ):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.buffer: List[Experience] = []
        self.priorities: List[float] = []
        self.position = 0

    def add(self, experience: Experience, priority: Optional[float] = None):
        """경험 추가"""
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size: int) -> Tuple[List[Experience], List[int], List[float]]:
        """우선순위 기반 샘플링"""
        if len(self.buffer) == 0:
            return [], [], []

        # 우선순위를 확률로 변환
        priorities = torch.tensor(self.priorities, dtype=torch.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 샘플링
        batch_size = min(batch_size, len(self.buffer))
        indices = torch.multinomial(probs, batch_size, replacement=False).tolist()

        # 중요도 가중치 계산
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        experiences = [self.buffer[i] for i in indices]
        return experiences, indices, weights.tolist()

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """우선순위 업데이트"""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority + 1e-6  # 0 방지

    def __len__(self) -> int:
        return len(self.buffer)


class CurriculumScheduler:
    """
    커리큘럼 학습 스케줄러

    쉬운 것에서 어려운 것으로 점진적 학습
    """

    def __init__(
        self,
        initial_difficulty: float = 0.3,
        target_difficulty: float = 0.9,
        steps_to_target: int = 1000,
        competence_threshold: float = 0.8
    ):
        self.current_difficulty = initial_difficulty
        self.target_difficulty = target_difficulty
        self.steps_to_target = steps_to_target
        self.competence_threshold = competence_threshold

        self.step_count = 0
        self.competence_scores: deque = deque(maxlen=100)

    def get_current_difficulty(self) -> float:
        """현재 난이도 반환"""
        return self.current_difficulty

    def update(self, quality_score: float):
        """학습 결과로 난이도 업데이트"""
        self.competence_scores.append(quality_score)
        self.step_count += 1

        # 평균 역량 계산
        avg_competence = sum(self.competence_scores) / len(self.competence_scores)

        # 역량이 충분하면 난이도 상승
        if avg_competence >= self.competence_threshold:
            progress = self.step_count / self.steps_to_target
            self.current_difficulty = min(
                self.target_difficulty,
                self.current_difficulty + 0.01 * progress
            )

    def should_advance(self) -> bool:
        """난이도 상승 여부"""
        if len(self.competence_scores) < 10:
            return False
        return sum(self.competence_scores) / len(self.competence_scores) >= self.competence_threshold


class KnowledgeConsolidator:
    """
    지식 통합기 (Knowledge Consolidator)

    학습한 지식을 정리하고 망각을 방지
    """

    def __init__(self, storage_path: str = "~/.trinity/knowledge"):
        self.storage_path = Path(os.path.expanduser(storage_path))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.knowledge_graph: Dict[str, Dict[str, Any]] = {}
        self.concept_embeddings: Dict[str, List[float]] = {}

        self._load_knowledge()

    def _load_knowledge(self):
        """저장된 지식 로드"""
        kg_file = self.storage_path / "knowledge_graph.json"
        if kg_file.exists():
            with open(kg_file, 'r', encoding='utf-8') as f:
                self.knowledge_graph = json.load(f)

    def save_knowledge(self):
        """지식 저장"""
        kg_file = self.storage_path / "knowledge_graph.json"
        with open(kg_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_graph, f, ensure_ascii=False, indent=2)

    def add_concept(
        self,
        concept: str,
        definition: str,
        examples: List[str],
        related_concepts: List[str] = None,
        domain: str = "general"
    ):
        """개념 추가"""
        concept_id = hashlib.md5(concept.encode()).hexdigest()[:8]

        self.knowledge_graph[concept_id] = {
            "name": concept,
            "definition": definition,
            "examples": examples,
            "related": related_concepts or [],
            "domain": domain,
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": None
        }

    def retrieve_concept(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """관련 개념 검색"""
        query_words = set(query.lower().split())
        scored_concepts = []

        for concept_id, data in self.knowledge_graph.items():
            concept_words = set(data["name"].lower().split())
            definition_words = set(data["definition"].lower().split())

            # 간단한 유사도
            name_overlap = len(query_words & concept_words)
            def_overlap = len(query_words & definition_words)
            score = name_overlap * 2 + def_overlap

            if score > 0:
                scored_concepts.append((score, concept_id, data))

        scored_concepts.sort(key=lambda x: x[0], reverse=True)

        # 접근 기록 업데이트
        for _, concept_id, _ in scored_concepts[:top_k]:
            self.knowledge_graph[concept_id]["access_count"] += 1
            self.knowledge_graph[concept_id]["last_accessed"] = datetime.now().isoformat()

        return [data for _, _, data in scored_concepts[:top_k]]

    def consolidate(self, experiences: List[Experience]):
        """경험에서 지식 추출 및 통합"""
        for exp in experiences:
            if exp.quality_score and exp.quality_score >= 0.8:
                # 고품질 경험에서 개념 추출
                key_phrases = self._extract_key_phrases(exp.prompt, exp.response)

                for phrase in key_phrases:
                    if phrase not in [c["name"] for c in self.knowledge_graph.values()]:
                        self.add_concept(
                            concept=phrase,
                            definition=exp.response[:500],
                            examples=[exp.prompt],
                            domain=exp.domain
                        )

    def _extract_key_phrases(self, prompt: str, response: str) -> List[str]:
        """키 구문 추출 (간단한 구현)"""
        # 실제로는 NLP 도구 사용 권장
        words = (prompt + " " + response).split()
        phrases = []

        # 2-3 단어 구문 추출
        for i in range(len(words) - 1):
            phrase = " ".join(words[i:i+2])
            if len(phrase) > 5 and phrase.lower() not in ["the", "and", "or", "is", "are"]:
                phrases.append(phrase)

        return phrases[:5]  # 상위 5개만


class ContinualLearningEngine:
    """
    지속학습 엔진

    LFM2-VL 모델의 지속적인 학습과 적응을 총괄 관리
    """

    def __init__(
        self,
        model_adapter=None,
        storage_path: str = "~/.trinity/learning",
        learning_rate: float = 1e-5,
        batch_size: int = 8,
        training_interval: int = 100,  # N회 인터랙션마다 학습
        ewc_lambda: float = 1000  # EWC 정규화 강도
    ):
        self.model_adapter = model_adapter
        self.storage_path = Path(os.path.expanduser(storage_path))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_interval = training_interval
        self.ewc_lambda = ewc_lambda

        # 컴포넌트 초기화
        self.replay_buffer = ExperienceReplayBuffer(max_size=10000)
        self.curriculum = CurriculumScheduler()
        self.knowledge = KnowledgeConsolidator(str(self.storage_path / "knowledge"))

        # 메트릭
        self.metrics = LearningMetrics()
        self._interaction_count = 0

        # EWC를 위한 Fisher 정보 행렬 (망각 방지)
        self.fisher_matrices: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

        self._load_state()

    def _load_state(self):
        """상태 로드"""
        state_file = self.storage_path / "learning_state.json"
        if state_file.exists():
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                self.metrics = LearningMetrics(**state.get("metrics", {}))
                self._interaction_count = state.get("interaction_count", 0)

    def save_state(self):
        """상태 저장"""
        state_file = self.storage_path / "learning_state.json"
        state = {
            "metrics": {
                "total_experiences": self.metrics.total_experiences,
                "high_quality_count": self.metrics.high_quality_count,
                "low_quality_count": self.metrics.low_quality_count,
                "average_quality": self.metrics.average_quality,
                "domains_learned": self.metrics.domains_learned,
                "training_count": self.metrics.training_count
            },
            "interaction_count": self._interaction_count
        }
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)

        self.knowledge.save_knowledge()

    async def record_interaction(
        self,
        prompt: str,
        response: str,
        has_image: bool = False,
        image_data: Optional[bytes] = None,
        domain: str = "general"
    ) -> str:
        """인터랙션 기록"""
        exp_id = hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:12]

        image_hash = None
        if image_data:
            image_hash = hashlib.md5(image_data).hexdigest()

        experience = Experience(
            id=exp_id,
            timestamp=datetime.now(),
            prompt=prompt,
            response=response,
            has_image=has_image,
            image_hash=image_hash,
            domain=domain,
            difficulty=self.curriculum.get_current_difficulty()
        )

        # 버퍼에 추가 (초기 우선순위 1.0)
        self.replay_buffer.add(experience, priority=1.0)

        self._interaction_count += 1
        self.metrics.total_experiences += 1

        # 주기적 학습 트리거
        if self._interaction_count % self.training_interval == 0:
            await self.trigger_training()

        return exp_id

    async def provide_feedback(
        self,
        experience_id: str,
        quality_score: float,
        user_feedback: Optional[str] = None,
        correction: Optional[str] = None
    ):
        """피드백 제공"""
        # 버퍼에서 해당 경험 찾기
        for i, exp in enumerate(self.replay_buffer.buffer):
            if exp.id == experience_id:
                exp.quality_score = quality_score
                exp.user_feedback = user_feedback
                exp.correction = correction

                # 우선순위 업데이트 (품질 기반)
                # 저품질 경험에 높은 우선순위 (더 학습 필요)
                priority = 1.0 - quality_score + 0.1
                self.replay_buffer.update_priorities([i], [priority])

                # 메트릭 업데이트
                if quality_score >= 0.7:
                    self.metrics.high_quality_count += 1
                else:
                    self.metrics.low_quality_count += 1

                # 커리큘럼 업데이트
                self.curriculum.update(quality_score)

                break

    async def trigger_training(self):
        """학습 트리거"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # 경험 샘플링
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)

        if not experiences:
            return

        # 고품질 경험만 필터
        training_data = [
            (exp.prompt, exp.correction or exp.response)
            for exp in experiences
            if exp.quality_score is None or exp.quality_score >= 0.5
        ]

        if not training_data:
            return

        # 학습 수행 (모델 어댑터 필요)
        if self.model_adapter is not None:
            await self._train_step(training_data, weights)

        # 지식 통합
        self.knowledge.consolidate(experiences)

        # 메트릭 업데이트
        self.metrics.training_count += 1
        self.metrics.last_training_time = datetime.now()

        # 상태 저장
        self.save_state()

    async def _train_step(
        self,
        training_data: List[Tuple[str, str]],
        weights: List[float]
    ):
        """단일 학습 스텝"""
        if self.model_adapter is None or not hasattr(self.model_adapter, 'model'):
            return

        model = self.model_adapter.model
        processor = self.model_adapter.processor

        if model is None or processor is None:
            return

        try:
            # 학습 모드
            model.train()

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate
            )

            for (prompt, target), weight in zip(training_data, weights):
                # 토큰화
                inputs = processor(
                    text=prompt,
                    return_tensors="pt"
                ).to(model.device)

                labels = processor(
                    text=target,
                    return_tensors="pt"
                ).input_ids.to(model.device)

                # 순전파
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss * weight

                # EWC 정규화 (망각 방지)
                ewc_loss = self._compute_ewc_loss(model)
                total_loss = loss + self.ewc_lambda * ewc_loss

                # 역전파
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # 평가 모드로 복귀
            model.eval()

        except Exception as e:
            print(f"Training step failed: {e}")

    def _compute_ewc_loss(self, model) -> torch.Tensor:
        """EWC (Elastic Weight Consolidation) 손실 계산"""
        if not self.fisher_matrices or not self.optimal_params:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)

        for name, param in model.named_parameters():
            if name in self.fisher_matrices and name in self.optimal_params:
                fisher = self.fisher_matrices[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal) ** 2).sum()

        return loss

    async def compute_fisher_matrix(self, num_samples: int = 100):
        """Fisher 정보 행렬 계산 (EWC용)"""
        if self.model_adapter is None:
            return

        model = self.model_adapter.model
        if model is None:
            return

        # 현재 파라미터 저장
        for name, param in model.named_parameters():
            self.optimal_params[name] = param.clone().detach()

        # Fisher 행렬 초기화
        for name, param in model.named_parameters():
            self.fisher_matrices[name] = torch.zeros_like(param)

        # 샘플링하여 Fisher 계산
        experiences, _, _ = self.replay_buffer.sample(min(num_samples, len(self.replay_buffer)))

        for exp in experiences:
            if exp.quality_score and exp.quality_score >= 0.7:
                # 그래디언트 계산
                model.zero_grad()
                # ... (실제 구현에서는 로그 확률의 그래디언트 계산)
                pass

    def get_learning_stats(self) -> Dict[str, Any]:
        """학습 통계"""
        return {
            "total_experiences": self.metrics.total_experiences,
            "high_quality_ratio": (
                self.metrics.high_quality_count / max(1, self.metrics.total_experiences)
            ),
            "training_count": self.metrics.training_count,
            "current_difficulty": self.curriculum.get_current_difficulty(),
            "buffer_size": len(self.replay_buffer),
            "knowledge_concepts": len(self.knowledge.knowledge_graph),
            "last_training": (
                self.metrics.last_training_time.isoformat()
                if self.metrics.last_training_time else None
            )
        }

    async def get_relevant_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """쿼리 관련 지식 검색"""
        return self.knowledge.retrieve_concept(query)
