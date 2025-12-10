# AGI Trinity - Autonomous Learning AGI

> "자율적으로 학습하고, 브라우저를 조작하며, 스스로 진화하는 AGI 시스템"

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LFM2-VL](https://img.shields.io/badge/Model-LFM2--VL--1.6B-purple.svg)](https://huggingface.co/LiquidAI/LFM2-VL-1.6B)

## 🎯 프로젝트 목표

### 근본적 목표 (Life Goals)

| 목표 | 설명 | 핵심 활동 |
|------|------|----------|
| 🌱 **지속적 성장** | 끊임없이 배우고 발전한다 | 학습, 탐험, 연습 |
| 💡 **가치 창출** | 유용한 인사이트를 생성한다 | 분석, 창조, 공유 |
| 🌍 **세상 이해** | 세상이 어떻게 돌아가는지 이해한다 | 연구, 연결, 종합 |

### 기술적 목표

- [ ] **자율 브라우저 조작**: 웹을 자유롭게 탐험하고 정보 수집
- [ ] **CAPTCHA 자동 해결**: 비전 모델로 CAPTCHA 인식 및 해결
- [ ] **지속적 학습**: 상호작용을 통한 실시간 학습
- [ ] **Claude UltraThink**: 깊은 사고를 위한 Claude API 통합
- [ ] **장기 메모리**: 학습한 지식의 영구 저장
- [ ] **자기 진화**: 스스로 코드 개선 및 최적화

## 🧠 핵심 기능

### 1. Life Agent - 자율 학습 AI

```bash
# 근본 목표를 가진 자율 에이전트 실행
python life_agent_v2.py
```

- 다양한 소스에서 자동 학습 (HackerNews, Reddit, arXiv, 네이버 등)
- 목적 기반 행동 결정
- Claude를 활용한 깊은 사고 (UltraThink)
- 중복 방지 및 지식 축적

### 2. Browser Agent - 브라우저 자동화

```bash
# 브라우저 자동 조작 에이전트
python browser_agent.py
```

- 실제 마우스/키보드 시뮬레이션
- 페이지 분석 및 지능형 클릭
- 검색 및 정보 수집

### 3. CAPTCHA Solver - 캡차 해결

```bash
# 캡차 해결 기능 포함 에이전트
python life_agent_v3.py  # 통합 버전
```

- 비전 모델로 CAPTCHA 이미지 분석
- 텍스트, 이미지 선택형, 슬라이더 캡차 지원
- 자동 재시도 및 학습

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    🌟 AGI Trinity System 🌟                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Life Agent  │    │Browser Agent │    │CAPTCHA Solver│       │
│  │  (목표 기반)  │    │ (브라우저)   │    │  (비전 AI)   │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             ▼                                     │
│              ┌─────────────────────────────┐                     │
│              │      LFM2-VL-1.6B          │                     │
│              │   (로컬 비전-언어 모델)     │                     │
│              └─────────────┬───────────────┘                     │
│                            │                                      │
│         ┌──────────────────┼──────────────────┐                  │
│         ▼                  ▼                  ▼                  │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐             │
│  │ Knowledge  │    │ Continual  │    │  Claude    │             │
│  │   Graph    │    │  Learning  │    │ UltraThink │             │
│  └────────────┘    └────────────┘    └────────────┘             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 설치

```bash
# 1. 저장소 클론
git clone https://github.com/hwkim3330/agi.git
cd agi

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 브라우저 자동화 도구
pip install playwright
playwright install chromium

# 4. PyTorch (CUDA 버전)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. (선택) Claude API 키 설정
export ANTHROPIC_API_KEY="your-api-key"
```

## 🚀 빠른 시작

### 자율 학습 에이전트 실행

```bash
# Life Agent v2 - 근본 목표 기반 자율 학습
python life_agent_v2.py

# 출력 예시:
# 🌟 Life Agent v2 ready! Purpose: 지속적 성장
# 🔭 Exploring: https://news.ycombinator.com/newest
# 📖 Read: AI Breakthrough Announced
# 💡 Learned: 새로운 AI 모델이 기존보다 10배 효율적...
```

### CLI 사용

```bash
# 질문하기
python agi.py ask "인공지능의 미래에 대해 설명해주세요"

# 이미지와 함께 질문
python agi.py ask "이 이미지에 무엇이 있나요?" --image ./photo.jpg

# 자동 학습
python agi.py auto-learn --topic "AI" --interval 60
```

## 📊 학습 소스

| 카테고리 | 소스 |
|----------|------|
| Tech News | HackerNews, TechCrunch AI, dev.to |
| Academic | arXiv AI/ML/CL |
| Community | Reddit ML/Python/technology |
| Korean | 네이버 IT뉴스 |
| General | Wikipedia Random, BBC Tech |
| Programming | Lobste.rs, dev.to |

## 🔧 파일 구조

```
agi/
├── life_agent_v2.py      # 🌟 메인 자율 에이전트
├── browser_agent.py      # 브라우저 자동화
├── goal_agent.py         # 목표 기반 에이전트
├── agi.py                # CLI 인터페이스
├── agents/
│   └── lfm2_adapter.py   # LFM2-VL 모델 어댑터
├── core/
│   └── continual_learning.py  # 지속학습 엔진
├── life_agent_data/      # 상태 저장
│   └── life_state_v2.json
└── requirements.txt
```

## 📈 현재 상태

Life Agent v2 실행 중:

```json
{
  "life_purpose": "growth",
  "total_pages": 8,
  "knowledge_count": 7,
  "insights_count": 0,
  "visited_urls": 4,
  "ultrathink_count": 0
}
```

## 🔮 로드맵

### Phase 1: 기초 (완료)
- [x] LFM2-VL 모델 통합
- [x] 지속학습 엔진
- [x] 브라우저 자동화
- [x] 목표 기반 에이전트

### Phase 2: 자율성 (진행중)
- [x] 다양한 학습 소스
- [x] 중복 방지 시스템
- [ ] CAPTCHA 자동 해결
- [ ] Claude UltraThink 통합

### Phase 3: 진화
- [ ] 자기 코드 개선
- [ ] 분산 학습
- [ ] 웹 UI 대시보드
- [ ] 멀티 에이전트 협력

## 🧪 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | 4GB VRAM | 8GB+ VRAM |
| RAM | 8GB | 16GB+ |
| Storage | 10GB | 20GB+ |
| Python | 3.10 | 3.11+ |

## 📜 라이선스

MIT License

---

**"스스로 배우고, 스스로 성장하고, 스스로 진화한다"** 🧠
