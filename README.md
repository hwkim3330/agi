# AGI Trinity - Autonomous Learning AGI

> "자율적으로 학습하고, 브라우저를 조작하며, 스스로 진화하는 AGI 시스템"

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LFM2-VL](https://img.shields.io/badge/Model-LFM2--VL--1.6B-purple.svg)](https://huggingface.co/LiquidAI/LFM2-VL-1.6B)

---

## 핵심 시스템

### 1. Life Agent - 자율 학습 AI (비전 기반)

스크린샷을 보고 스스로 판단하고 행동하는 자율 에이전트

```bash
# 최신 버전 - 비전 기반 브라우저 조작
python life_agents/life_agent_v4.py
```

| 버전 | 특징 |
|------|------|
| v1 | 기본 브라우저 탐험 |
| v2 | 목적 기반 학습 (growth, value_creation, understanding) |
| v3 | CAPTCHA 해결 + Browser Use 학습 |
| **v4** | **비전 기반 제어** - 스크린샷 → 좌표 클릭 (VLA 스타일) |

### 2. Trinity - 3개 모델 합의 시스템

여러 AI 모델(Claude, Gemini, GPT)의 응답을 종합하여 최적의 결과 도출

```bash
# Trinity 사용
python trinity/trinity.py ask "질문"
python trinity/trinity.py --strategy synthesis "복잡한 문제"
```

**합의 전략:**
- `vote`: 가장 좋은 응답 선택
- `synthesis`: 모든 응답 종합
- `fanout`: 개별 응답 모두 반환

### 3. Browser Agent - 브라우저 자동화

```bash
python browser/browser_agent.py
```

## 프로젝트 구조

```
agi/
├── life_agents/           # 자율 학습 에이전트
│   ├── life_agent.py      # v1 - 기본
│   ├── life_agent_v2.py   # v2 - 목적 기반
│   ├── life_agent_v3.py   # v3 - CAPTCHA + Browser Use
│   └── life_agent_v4.py   # v4 - 비전 기반 (최신)
│
├── trinity/               # 3개 모델 합의 시스템
│   └── trinity.py         # Multi-Agent Orchestrator
│
├── browser/               # 브라우저 자동화
│   ├── browser_agent.py
│   ├── browser_learner.py
│   └── computer_use_learner.py
│
├── learners/              # 학습 에이전트
│   ├── fast_learner.py
│   └── trend_learner.py
│
├── agents/                # 모델 어댑터
│   └── lfm2_adapter.py    # LFM2-VL 모델
│
├── core/                  # 핵심 엔진
│   └── continual_learning.py
│
├── config/                # 설정
│   └── agents.yaml        # Trinity 에이전트 설정
│
├── agi.py                 # CLI 인터페이스
├── goal_agent.py          # 목표 기반 에이전트
└── eternal_agi.py         # 영구 실행 에이전트
```

## 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│                    AGI Trinity System                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐  │
│  │   Life Agent    │    │     Trinity     │    │   Browser    │  │
│  │   (비전 기반)    │    │ (3모델 합의)    │    │    Agent     │  │
│  └────────┬────────┘    └────────┬────────┘    └──────┬───────┘  │
│           │                      │                     │          │
│           └──────────────────────┼─────────────────────┘          │
│                                  ▼                                 │
│              ┌────────────────────────────────────┐               │
│              │         LFM2-VL-1.6B              │               │
│              │     (로컬 비전-언어 모델)          │               │
│              └────────────────┬───────────────────┘               │
│                               │                                    │
│         ┌─────────────────────┼─────────────────────┐             │
│         ▼                     ▼                     ▼             │
│  ┌────────────┐       ┌────────────┐       ┌────────────┐        │
│  │ Knowledge  │       │ Continual  │       │   Claude   │        │
│  │   Base     │       │  Learning  │       │ UltraThink │        │
│  └────────────┘       └────────────┘       └────────────┘        │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

## 근본적 목표 (Life Goals)

| 목표 | 설명 | 핵심 활동 |
|------|------|----------|
| **지속적 성장** | 끊임없이 배우고 발전한다 | 학습, 탐험, 연습 |
| **가치 창출** | 유용한 인사이트를 생성한다 | 분석, 창조, 공유 |
| **세상 이해** | 세상이 어떻게 돌아가는지 이해한다 | 연구, 연결, 종합 |

## 빠른 시작

```bash
# 1. 저장소 클론
git clone https://github.com/hwkim3330/agi.git
cd agi

# 2. 의존성 설치
pip install -r requirements.txt
pip install playwright && playwright install chromium

# 3. 비전 기반 자율 학습 에이전트 실행
python life_agents/life_agent_v4.py

# 4. Trinity 3모델 합의 시스템 사용
python trinity/trinity.py ask "인공지능의 미래"
```

## 핵심 기능

### 비전 기반 브라우저 제어 (v4)

```python
# 스크린샷 촬영 → 비전 모델 분석 → 좌표 클릭
class VisionBrowserAgent:
    async def analyze_screen(self, screenshot, goal):
        """비전 모델이 스크린샷을 보고 다음 행동 결정"""
        # ACTION: CLICK / CLICK_CAPTCHA / TYPE / SCROLL / WAIT / DONE
        # X, Y: 클릭 좌표

    async def execute_action(self, page, action):
        await page.mouse.click(action["x"], action["y"])
```

### CAPTCHA 자동 해결

- Cloudflare Turnstile
- reCAPTCHA
- hCaptcha
- 비전 모델이 체크박스 좌표를 직접 판단

### 행동 패턴 학습

```python
# 성공한 행동 시퀀스 저장
def record_success(self, goal, actions):
    pattern = {"goal": goal, "actions": actions}
    self.success_patterns.append(pattern)
```

## 학습 소스

| 카테고리 | 소스 |
|----------|------|
| Tech | HackerNews, Lobste.rs |
| Academic | arXiv AI/ML/CL |
| Community | Reddit ML/Python |
| Korean | 네이버 IT뉴스 |
| General | Wikipedia Random |

## 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | 4GB VRAM | 8GB+ VRAM |
| RAM | 8GB | 16GB+ |
| Python | 3.10 | 3.11+ |

## 로드맵

- [x] LFM2-VL 모델 통합
- [x] 지속학습 엔진
- [x] 브라우저 자동화
- [x] 비전 기반 CAPTCHA 해결
- [x] Trinity 3모델 합의
- [ ] 자기 코드 개선
- [ ] 분산 학습
- [ ] 웹 UI 대시보드

---

**"스스로 배우고, 스스로 성장하고, 스스로 진화한다"**
