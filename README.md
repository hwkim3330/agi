# AGI Trinity - Continual Learning AGI

> "지속적으로 학습하고 진화하는 비전-언어 AGI 시스템"

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LFM2-VL](https://img.shields.io/badge/Model-LFM2--VL--1.6B-purple.svg)](https://huggingface.co/LiquidAI/LFM2-VL-1.6B)

## 🧠 개요

AGI Trinity는 **LiquidAI의 LFM2-VL-1.6B** 비전-언어 모델을 기반으로 한 **지속학습(Continual Learning) AGI 시스템**입니다.

### 핵심 특징

| 기능 | 설명 |
|------|------|
| 🖼️ **멀티모달** | 텍스트 + 이미지를 함께 이해 |
| 📚 **지속학습** | 상호작용을 통해 지속적으로 학습 |
| 🧠 **장기 메모리** | 학습한 개념과 지식을 저장 |
| ⚡ **로컬 실행** | 외부 API 없이 로컬에서 실행 |
| 🎯 **적응형** | 사용자 피드백으로 개선 |

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      AGI Trinity                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   User      │───▶│  Multimodal │───▶│   LFM2-VL   │     │
│  │   Input     │    │   Engine    │    │   Model     │     │
│  └─────────────┘    └─────────────┘    └──────┬──────┘     │
│        │                                       │            │
│        │            ┌─────────────────────────┘            │
│        │            ▼                                       │
│        │     ┌─────────────┐                               │
│        │     │  Response   │                               │
│        │     └──────┬──────┘                               │
│        │            │                                       │
│        ▼            ▼                                       │
│  ┌─────────────────────────────────┐                       │
│  │     Continual Learning Engine   │                       │
│  ├─────────────────────────────────┤                       │
│  │ • Experience Replay Buffer      │                       │
│  │ • Curriculum Scheduler          │                       │
│  │ • Knowledge Consolidator        │                       │
│  │ • EWC (망각 방지)                │                       │
│  └─────────────────────────────────┘                       │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────┐                       │
│  │       Long-Term Memory          │                       │
│  │  • Knowledge Graph              │                       │
│  │  • Learned Concepts             │                       │
│  │  • Model Checkpoints            │                       │
│  └─────────────────────────────────┘                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📦 설치

### 1. 저장소 클론

```bash
git clone https://github.com/hwkim3330/agi.git
cd agi
```

### 2. 의존성 설치

```bash
# Python 패키지
pip install -r requirements.txt

# PyTorch (CUDA 버전에 맞게 설치)
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. 모델 다운로드 (자동)

첫 실행 시 자동으로 LFM2-VL-1.6B 모델이 다운로드됩니다 (~3GB).

## 🚀 사용법

### CLI 명령어

```bash
# 질문하기
python agi.py ask "인공지능의 미래에 대해 설명해주세요"

# 이미지와 함께 질문
python agi.py ask "이 이미지에 무엇이 있나요?" --image ./photo.jpg

# 대화형 채팅
python agi.py chat

# 피드백 제공 (학습 개선)
python agi.py feedback abc123 --quality 0.9 --correction "더 나은 답변..."

# 특정 주제 자기학습
python agi.py learn "양자컴퓨팅" --depth 5

# 지식 그래프 조회
python agi.py knowledge "machine learning"
python agi.py knowledge --list

# 학습 트리거
python agi.py train --force

# 상태 확인
python agi.py status

# 내보내기
python agi.py export agi_backup.json
```

### Python API

```python
import asyncio
from agents.lfm2_adapter import LFM2VLAdapter, LFM2Config
from core.continual_learning import ContinualLearningEngine

async def main():
    # AGI 초기화
    config = LFM2Config(
        model_id="LiquidAI/LFM2-VL-1.6B",
        enable_continual_learning=True
    )
    agi = LFM2VLAdapter(lfm2_config=config)
    learning = ContinualLearningEngine(model_adapter=agi)

    # 모델 로드
    await agi.load_model()

    # 질문
    response = await agi.execute("What is machine learning?")
    print(response.content)

    # 이미지와 함께 질문
    response = await agi.execute(
        "Describe this image",
        images=["./photo.jpg"]
    )

    # 피드백 제공
    exp_id = await learning.record_interaction(
        prompt="What is ML?",
        response=response.content
    )
    await learning.provide_feedback(exp_id, quality_score=0.9)

    # 학습
    await learning.trigger_training()

asyncio.run(main())
```

## 🎓 지속학습 시스템

### 경험 재생 버퍼 (Experience Replay)

상호작용을 저장하고 우선순위 기반으로 학습합니다.

```python
# 고품질 경험에 높은 우선순위
await learning.provide_feedback(
    experience_id="abc123",
    quality_score=0.95,
    correction="개선된 응답..."
)
```

### 커리큘럼 학습

쉬운 것에서 어려운 것으로 점진적 학습:

```
Level 1 (0.3) → Level 2 (0.5) → Level 3 (0.7) → Level 4 (0.9)
```

### EWC (Elastic Weight Consolidation)

이전에 학습한 지식을 보존하면서 새로운 것을 학습:

```
Loss = Task_Loss + λ * Σ Fisher_i * (θ_i - θ*_i)²
```

### 지식 그래프

학습한 개념들을 구조화하여 저장:

```json
{
  "concept_id": {
    "name": "Machine Learning",
    "definition": "...",
    "examples": ["supervised", "unsupervised"],
    "related": ["AI", "Deep Learning"],
    "access_count": 42
  }
}
```

## 📁 프로젝트 구조

```
agi/
├── agi.py                      # 메인 CLI
├── trinity.py                  # 레거시 CLI
├── requirements.txt
├── config/
│   ├── lfm2_config.yaml       # LFM2 설정
│   └── agents.yaml            # 에이전트 설정
├── agents/
│   ├── base.py                # 기본 어댑터
│   ├── lfm2_adapter.py        # LFM2-VL 어댑터
│   └── ...
├── core/
│   ├── continual_learning.py  # 지속학습 엔진
│   ├── multimodal.py          # 멀티모달 처리
│   ├── consensus.py           # 합의 엔진
│   └── router.py              # 라우터
├── scripts/
│   └── train_continual.py     # 훈련 스크립트
└── tests/
```

## ⚙️ 설정

`config/lfm2_config.yaml`:

```yaml
model:
  id: "LiquidAI/LFM2-VL-1.6B"
  device: "auto"
  dtype: "bfloat16"

generation:
  max_new_tokens: 512
  temperature: 0.1

continual_learning:
  enabled: true
  learning_rate: 0.00001
  training_interval: 100

  lora:
    enabled: true
    r: 8
    alpha: 16
```

## 📊 모델 정보

### LFM2-VL-1.6B

| 속성 | 값 |
|------|-----|
| 파라미터 (LM) | 1.2B |
| 비전 인코더 | SigLIP2 NaFlex (400M) |
| 컨텍스트 길이 | 32,768 토큰 |
| 정밀도 | bfloat16 |
| 추론 속도 | 기존 VLM 대비 2배 |

### 시스템 요구사항

- **GPU**: 4GB+ VRAM (8GB+ 권장)
- **RAM**: 8GB+ (16GB 권장)
- **저장공간**: 10GB+
- **Python**: 3.10+

## 🔧 개발

```bash
# 테스트
pytest tests/ -v

# 린트
ruff check .
mypy .

# 훈련 스크립트 실행
python scripts/train_continual.py --help
```

## 📝 로드맵

- [x] LFM2-VL 모델 통합
- [x] 지속학습 엔진
- [x] 멀티모달 처리
- [x] 경험 재생 버퍼
- [x] 지식 그래프
- [ ] 음성 입출력
- [ ] 웹 UI
- [ ] 분산 학습
- [ ] RAG 통합

## 📜 라이선스

MIT License

## 🙏 감사

- [Liquid AI](https://liquid.ai) - LFM2-VL 모델
- [Hugging Face](https://huggingface.co) - Transformers 라이브러리

---

**"지속적인 학습을 통해 더 나은 AI로 진화합니다"** 🧠
