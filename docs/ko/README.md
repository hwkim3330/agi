# AGI Trinity - 자율 학습 AGI

> 웹을 탐험하고, 캡차를 풀고, 스스로 진화하는 자율 학습 AGI

---

## 핵심 시스템

### 1. Life Agent - 비전 기반 자율 학습

스크린샷을 보고 스스로 판단하고 행동하는 에이전트 (VLA 스타일)

```bash
python life_agents/life_agent_v4.py
```

| 버전 | 특징 |
|------|------|
| v1 | 기본 브라우저 탐험 |
| v2 | 목적 기반 학습 (성장, 가치창출, 이해) |
| v3 | CAPTCHA 해결 + Browser Use 학습 |
| **v4** | **비전 기반 제어** - 스크린샷 → 좌표 → 클릭 |

### 2. Trinity - 3개 모델 합의 시스템

여러 AI 모델(Claude, Gemini, GPT)의 응답을 종합하여 최적의 결과 도출

```bash
python trinity/trinity.py ask "질문"
python trinity/trinity.py ask "복잡한 문제" --strategy synthesis
```

**합의 전략:**
- `vote`: 최고 점수 응답 선택
- `synthesis`: 모든 인사이트 통합
- `fanout`: 모든 응답 반환

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
├── learners/              # 학습 에이전트
├── agents/                # 모델 어댑터
└── core/                  # 핵심 엔진
```

## 근본적 목표 (Life Goals)

에이전트의 행동을 이끄는 근본적 목적:

| 목표 | 설명 | 활동 |
|------|------|------|
| **성장** | 끊임없이 배우고 발전한다 | 학습, 탐험, 연습 |
| **가치 창출** | 유용한 인사이트를 생성한다 | 분석, 창조, 공유 |
| **세상 이해** | 세상이 어떻게 돌아가는지 이해한다 | 연구, 연결, 종합 |

## 빠른 시작

```bash
# 1. 클론
git clone https://github.com/hwkim3330/agi.git
cd agi

# 2. 의존성 설치
pip install -r requirements.txt
pip install playwright && playwright install chromium

# 3. 비전 기반 자율 에이전트 실행
python life_agents/life_agent_v4.py

# 4. Trinity 합의 시스템 사용
python trinity/trinity.py ask "인공지능의 미래"
```

## 핵심 기능

### 비전 기반 브라우저 제어 (v4)

```python
class VisionBrowserAgent:
    async def analyze_screen(self, screenshot, goal):
        """비전 모델이 스크린샷을 보고 다음 행동 결정"""
        # 반환: ACTION, X, Y 좌표

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
def record_success(self, goal, actions):
    """성공한 행동 시퀀스 저장"""
    pattern = {"goal": goal, "actions": actions}
    self.success_patterns.append(pattern)
```

## 학습 소스

| 카테고리 | 소스 |
|----------|------|
| 기술 | HackerNews, Lobste.rs |
| 학술 | arXiv AI/ML/CL |
| 커뮤니티 | Reddit ML/Python |
| 한국어 | 네이버 IT뉴스 |
| 일반 | Wikipedia Random |

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
