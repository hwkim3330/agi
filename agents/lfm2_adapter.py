#!/usr/bin/env python3
"""
AGI Trinity - LFM2-VL-1.6B Adapter
Liquid AI LFM2 Vision-Language Model 어댑터

지속학습(Continual Learning)을 지원하는 로컬 멀티모달 AI 에이전트
"""
import asyncio
import time
import torch
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import io
import base64

from .base import BaseAgentAdapter, AgentConfig, AgentResponse, AgentStatus


@dataclass
class LFM2Config:
    """LFM2-VL 모델 설정"""
    model_id: str = "LiquidAI/LFM2-VL-1.6B"
    device: str = "auto"
    dtype: str = "bfloat16"
    max_new_tokens: int = 512
    temperature: float = 0.1
    min_p: float = 0.15
    repetition_penalty: float = 1.05
    # Vision 설정
    min_image_tokens: int = 64
    max_image_tokens: int = 256
    do_image_splitting: bool = True
    # 메모리 설정
    memory_path: str = "~/.trinity/lfm2_memory"
    max_context_length: int = 32768
    # 지속학습 설정
    enable_continual_learning: bool = True
    learning_rate: float = 1e-5
    save_interval: int = 100  # 100 인터랙션마다 저장


class LFM2VLAdapter(BaseAgentAdapter):
    """
    LFM2-VL-1.6B Vision-Language 모델 어댑터

    Features:
    - 텍스트 + 이미지 멀티모달 처리
    - 지속학습 (Continual Learning)
    - 장기 메모리 시스템
    - 경량 로컬 추론
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        lfm2_config: Optional[LFM2Config] = None
    ):
        if config is None:
            config = AgentConfig(
                name="lfm2",
                role="Multimodal AGI Core",
                specialty="Vision-Language Understanding, Continual Learning",
                mode="local",
                cmd=[],
                timeout_s=120,
                personality="Adaptive, learning, multimodal intelligence"
            )

        super().__init__(config)
        self.lfm2_config = lfm2_config or LFM2Config()

        self.model = None
        self.processor = None
        self._is_loaded = False
        self._interaction_count = 0

        # 메모리 시스템
        self.memory_path = Path(os.path.expanduser(self.lfm2_config.memory_path))
        self.memory_path.mkdir(parents=True, exist_ok=True)

        # 경험 버퍼 (지속학습용)
        self.experience_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 1000

        # 장기 메모리
        self.long_term_memory: Dict[str, Any] = self._load_long_term_memory()

    def _load_long_term_memory(self) -> Dict[str, Any]:
        """장기 메모리 로드"""
        memory_file = self.memory_path / "long_term_memory.json"
        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "learned_concepts": {},
            "interaction_history": [],
            "skill_weights": {},
            "last_updated": None
        }

    def _save_long_term_memory(self):
        """장기 메모리 저장"""
        memory_file = self.memory_path / "long_term_memory.json"
        self.long_term_memory["last_updated"] = datetime.now().isoformat()
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)

    async def load_model(self):
        """모델 로드 (지연 로딩)"""
        if self._is_loaded:
            return

        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32
            }
            dtype = dtype_map.get(self.lfm2_config.dtype, torch.bfloat16)

            # 저장된 체크포인트가 있으면 로드
            checkpoint_path = self.memory_path / "model_checkpoint"
            if checkpoint_path.exists():
                model_path = str(checkpoint_path)
            else:
                model_path = self.lfm2_config.model_id

            self.processor = AutoProcessor.from_pretrained(
                self.lfm2_config.model_id,
                trust_remote_code=True
            )

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                device_map=self.lfm2_config.device,
                torch_dtype=dtype,
                trust_remote_code=True
            )

            self._is_loaded = True
            self.status = AgentStatus.IDLE

        except Exception as e:
            self.status = AgentStatus.ERROR
            raise RuntimeError(f"Failed to load LFM2-VL model: {e}")

    async def execute(
        self,
        prompt: str,
        images: Optional[List[Union[str, Image.Image, bytes]]] = None
    ) -> AgentResponse:
        """
        프롬프트 실행 (텍스트 + 이미지)

        Args:
            prompt: 텍스트 프롬프트
            images: 이미지 리스트 (URL, PIL Image, 또는 bytes)

        Returns:
            AgentResponse: 응답
        """
        start_time = time.time()
        self.status = AgentStatus.RUNNING

        try:
            # 모델 로드 (필요시)
            if not self._is_loaded:
                await self.load_model()

            # 이미지 처리
            processed_images = []
            if images:
                for img in images:
                    processed_images.append(await self._process_image(img))

            # 대화 구성
            conversation = self._build_conversation(prompt, processed_images)

            # 컨텍스트에 메모리 추가
            context_prompt = self._add_memory_context(prompt)

            # 토큰화 및 추론
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True
            ).to(self.model.device)

            # 생성 파라미터
            gen_kwargs = {
                "max_new_tokens": self.lfm2_config.max_new_tokens,
                "temperature": self.lfm2_config.temperature,
                "min_p": self.lfm2_config.min_p,
                "repetition_penalty": self.lfm2_config.repetition_penalty,
                "do_sample": True
            }

            # 추론
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            # 디코딩
            response_text = self.processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]

            latency = time.time() - start_time

            # 경험 기록 (지속학습용)
            self._record_experience(prompt, response_text, images is not None)

            self.status = AgentStatus.SUCCESS

            response = AgentResponse(
                agent_name=self.name,
                success=True,
                content=response_text,
                latency=latency,
                metadata={
                    "role": self.role,
                    "model": self.lfm2_config.model_id,
                    "has_images": images is not None,
                    "image_count": len(processed_images) if processed_images else 0,
                    "tokens_generated": outputs.shape[1] - inputs["input_ids"].shape[1]
                },
                confidence=0.85
            )

            self._update_metrics(response)
            return response

        except Exception as e:
            latency = time.time() - start_time
            self.status = AgentStatus.ERROR

            return AgentResponse(
                agent_name=self.name,
                success=False,
                content="",
                latency=latency,
                error=str(e),
                metadata={"role": self.role}
            )

    async def _process_image(
        self,
        image: Union[str, Image.Image, bytes]
    ) -> Image.Image:
        """이미지 전처리"""
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        elif isinstance(image, str):
            if image.startswith("http"):
                from transformers.image_utils import load_image
                return load_image(image)
            elif image.startswith("data:image"):
                # Base64 데이터 URL
                header, data = image.split(",", 1)
                image_data = base64.b64decode(data)
                return Image.open(io.BytesIO(image_data))
            else:
                # 파일 경로
                return Image.open(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _build_conversation(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None
    ) -> List[Dict[str, Any]]:
        """대화 형식 구성"""
        content = []

        if images:
            for img in images:
                content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def _add_memory_context(self, prompt: str) -> str:
        """메모리 컨텍스트 추가"""
        # 관련 장기 메모리 검색
        relevant_memories = self._retrieve_relevant_memories(prompt)

        if relevant_memories:
            context = "Previous relevant knowledge:\n"
            for mem in relevant_memories[:3]:
                context += f"- {mem}\n"
            context += f"\nCurrent query: {prompt}"
            return context

        return prompt

    def _retrieve_relevant_memories(self, query: str, top_k: int = 3) -> List[str]:
        """관련 메모리 검색 (간단한 키워드 매칭)"""
        query_words = set(query.lower().split())
        memories = []

        for concept, data in self.long_term_memory.get("learned_concepts", {}).items():
            concept_words = set(concept.lower().split())
            overlap = len(query_words & concept_words)
            if overlap > 0:
                memories.append((overlap, data.get("summary", concept)))

        memories.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in memories[:top_k]]

    def _record_experience(
        self,
        prompt: str,
        response: str,
        has_image: bool
    ):
        """경험 기록 (지속학습용)"""
        experience = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "has_image": has_image,
            "quality_score": None  # 추후 피드백으로 업데이트
        }

        self.experience_buffer.append(experience)
        self._interaction_count += 1

        # 버퍼 크기 관리
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer = self.experience_buffer[-self.max_buffer_size:]

        # 주기적 저장
        if self._interaction_count % self.lfm2_config.save_interval == 0:
            self._save_experiences()
            self._save_long_term_memory()

    def _save_experiences(self):
        """경험 버퍼 저장"""
        exp_file = self.memory_path / f"experiences_{datetime.now().strftime('%Y%m%d')}.json"
        with open(exp_file, 'w', encoding='utf-8') as f:
            json.dump(self.experience_buffer, f, ensure_ascii=False, indent=2)

    async def learn_from_feedback(
        self,
        experience_id: int,
        quality_score: float,
        correction: Optional[str] = None
    ):
        """
        피드백으로부터 학습

        Args:
            experience_id: 경험 인덱스
            quality_score: 품질 점수 (0-1)
            correction: 수정된 응답 (선택)
        """
        if not self.lfm2_config.enable_continual_learning:
            return

        if experience_id < len(self.experience_buffer):
            exp = self.experience_buffer[experience_id]
            exp["quality_score"] = quality_score

            if correction:
                exp["correction"] = correction

                # 개념 학습
                prompt_words = exp["prompt"].split()[:5]
                concept_key = " ".join(prompt_words)

                if concept_key not in self.long_term_memory["learned_concepts"]:
                    self.long_term_memory["learned_concepts"][concept_key] = {
                        "first_seen": datetime.now().isoformat(),
                        "examples": []
                    }

                self.long_term_memory["learned_concepts"][concept_key]["examples"].append({
                    "prompt": exp["prompt"],
                    "good_response": correction,
                    "score": quality_score
                })

                self.long_term_memory["learned_concepts"][concept_key]["summary"] = correction[:200]

    async def fine_tune_on_experiences(
        self,
        min_quality_score: float = 0.7
    ):
        """
        고품질 경험으로 파인튜닝

        주의: GPU 메모리와 시간이 필요함
        """
        if not self.lfm2_config.enable_continual_learning:
            return

        # 고품질 경험만 필터링
        good_experiences = [
            exp for exp in self.experience_buffer
            if exp.get("quality_score", 0) >= min_quality_score
        ]

        if len(good_experiences) < 10:
            return  # 충분한 데이터 없음

        try:
            from peft import LoraConfig, get_peft_model
            from transformers import TrainingArguments, Trainer

            # LoRA 설정 (효율적인 파인튜닝)
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none"
            )

            # TODO: 전체 파인튜닝 파이프라인 구현
            # 이는 별도 스크립트로 실행하는 것이 좋음

        except ImportError:
            pass  # PEFT 없으면 스킵

    async def health_check(self) -> bool:
        """건강 상태 확인"""
        try:
            if not self._is_loaded:
                return True  # 아직 로드 안됨 - 정상

            # 간단한 추론 테스트
            test_prompt = "Hello"
            inputs = self.processor(
                text=test_prompt,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.inference_mode():
                self.model.generate(**inputs, max_new_tokens=5)

            return True

        except Exception:
            return False

    async def cleanup(self):
        """리소스 정리"""
        # 메모리 저장
        self._save_experiences()
        self._save_long_term_memory()

        # 모델 언로드
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        self._is_loaded = False

        # GPU 메모리 해제
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        await super().cleanup()

    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계"""
        return {
            "experience_buffer_size": len(self.experience_buffer),
            "learned_concepts": len(self.long_term_memory.get("learned_concepts", {})),
            "interaction_count": self._interaction_count,
            "last_updated": self.long_term_memory.get("last_updated"),
            "memory_path": str(self.memory_path)
        }


class LFM2Ensemble:
    """
    LFM2 모델 앙상블

    여러 LFM2 인스턴스를 운영하여 다양성과 견고성 확보
    """

    def __init__(self, num_instances: int = 3):
        self.instances: List[LFM2VLAdapter] = []

        for i in range(num_instances):
            config = AgentConfig(
                name=f"lfm2_instance_{i}",
                role=f"AGI Core Instance {i}",
                specialty="Vision-Language Understanding",
                mode="local",
                cmd=[],
                timeout_s=120
            )

            lfm2_config = LFM2Config(
                temperature=0.1 + i * 0.05,  # 다양성을 위한 온도 변화
                memory_path=f"~/.trinity/lfm2_memory_{i}"
            )

            self.instances.append(LFM2VLAdapter(config, lfm2_config))

    async def execute_ensemble(
        self,
        prompt: str,
        images: Optional[List] = None
    ) -> List[AgentResponse]:
        """모든 인스턴스에서 실행"""
        tasks = [
            instance.execute(prompt, images)
            for instance in self.instances
        ]
        return await asyncio.gather(*tasks)

    async def vote_best_response(
        self,
        prompt: str,
        images: Optional[List] = None
    ) -> AgentResponse:
        """최선의 응답 투표"""
        responses = await self.execute_ensemble(prompt, images)

        # 성공한 응답만
        successful = [r for r in responses if r.success]

        if not successful:
            return responses[0]  # 모두 실패시 첫번째 반환

        # 가장 긴 응답 선택 (간단한 휴리스틱)
        return max(successful, key=lambda r: len(r.content))
