# AGI Trinity - Multi-Agent AI Orchestrator

> "Three minds, one consciousness" - 차세대 멀티에이전트 AI 협업 프레임워크

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![WASM](https://img.shields.io/badge/WebAssembly-enabled-blueviolet.svg)](https://webassembly.org/)

## 개요

AGI Trinity는 **Claude Opus 4.5**, **Gemini 3 Pro**, **GPT-5.1**을 통합하여 집단 지능을 구현하는 멀티에이전트 오케스트레이터입니다.

### 왜 Trinity인가?

| 단일 AI의 한계 | Trinity의 해결책 |
|---------------|-----------------|
| 환각(Hallucination) | 교차 검증으로 신뢰도 향상 |
| 편향된 관점 | 다각도 분석으로 균형 있는 결과 |
| 특정 분야 약점 | 각 AI의 강점 조합 |
| 불확실한 답변 | 합의 기반 신뢰도 점수 |

## 아키텍처

```
                    ┌─────────────────────┐
                    │    USER REQUEST     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    SMART ROUTER     │
                    │  질문 분류 & 라우팅   │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
    ┌───────────┐        ┌───────────┐        ┌───────────┐
    │  CLAUDE   │        │  GEMINI   │        │    GPT    │
    │ Opus 4.5  │        │  3 Pro    │        │    5.1    │
    ├───────────┤        ├───────────┤        ├───────────┤
    │ • Coding  │        │ • Research│        │ • Creative│
    │ • Debug   │        │ • Analysis│        │ • Strategy│
    │ • Security│        │ • Multimod│        │ • Reasoning│
    └─────┬─────┘        └─────┬─────┘        └─────┬─────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  CONSENSUS ENGINE   │
                    │  (WASM Optimized)   │
                    ├─────────────────────┤
                    │ • Vote    • Debate  │
                    │ • Synth   • Expert  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   FINAL RESPONSE    │
                    │  Confidence: 94.2%  │
                    └─────────────────────┘
```

## 설치

```bash
# 클론
git clone https://github.com/hwkim3330/agi.git
cd agi

# 의존성 설치
pip install -r requirements.txt

# API 키 설정
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# (선택) WASM 빌드
cd wasm-consensus && wasm-pack build --target web && cd ..
```

## 사용법

### CLI

```bash
# 기본 질문
python trinity.py ask "마이크로서비스 아키텍처의 장단점"

# 전략 지정
python trinity.py ask "보안 취약점 분석" --strategy debate

# 특정 에이전트
python trinity.py ask "코드 리뷰" --agents claude,gpt
```

### Python API

```python
from trinity import Trinity

tri = Trinity()

# 기본 사용
response = tri.ask("REST vs GraphQL 비교")
print(response.content)
print(f"신뢰도: {response.confidence:.1%}")

# 전략 지정
response = tri.ask("신규 기능 아이디어", strategy="synthesis")
```

## 합의 전략

| 전략 | 설명 | 사용 사례 |
|------|------|----------|
| `vote` | 최고 점수 응답 선택 | 정답이 명확한 질문 |
| `synthesis` | 응답 통합 | 종합적 분석 필요 |
| `debate` | AI간 토론 후 결론 | 논쟁적 주제 |
| `specialist` | 전문가 자동 선택 | 특화된 질문 |

### Specialist 모드 자동 라우팅

```
코딩/디버깅 질문 ──────────▶ Claude (기술 전문가)
연구/데이터 분석 ──────────▶ Gemini (데이터 분석가)
창의적/전략적 질문 ────────▶ GPT (창의적 문제해결사)
복합적 질문 ───────────────▶ 3개 AI 동시 실행 + 합의
```

## 에이전트 역할

| 에이전트 | 모델 | 강점 |
|---------|------|------|
| **Claude** | Opus 4.5 | 코드 분석, 디버깅, 시스템 설계, 보안, 200K 컨텍스트 |
| **Gemini** | 3 Pro | 연구, 팩트체킹, 멀티모달, Deep Think 추론 |
| **GPT** | 5.1 | 창의적 솔루션, 브레인스토밍, 전략 수립, 통합 추론 |

## 프로젝트 구조

```
agi/
├── trinity.py              # CLI 엔트리포인트
├── core/
│   ├── orchestrator.py     # 메인 오케스트레이터
│   ├── consensus.py        # 합의 엔진
│   ├── router.py           # 스마트 라우터
│   └── session.py          # 세션 관리
├── agents/
│   ├── base.py             # 베이스 어댑터
│   ├── claude.py           # Claude API
│   ├── gemini.py           # Gemini API
│   └── openai.py           # OpenAI API
├── strategies/
│   ├── vote.py             # 투표 전략
│   ├── synthesis.py        # 통합 전략
│   ├── debate.py           # 토론 전략
│   └── specialist.py       # 전문가 선택
├── wasm-consensus/         # Rust WASM 고성능 합의
│   ├── src/
│   │   ├── lib.rs
│   │   ├── consensus.rs
│   │   └── similarity.rs
│   └── Cargo.toml
├── web/                    # 웹 인터페이스 (예정)
└── config/
    └── agents.yaml
```

## WASM 성능

| 작업 | Python | WASM | 개선 |
|------|--------|------|------|
| 텍스트 유사도 | 450ms | 12ms | 37x |
| 합의 계산 | 280ms | 8ms | 35x |
| 응답 파싱 | 120ms | 5ms | 24x |

## 개발

```bash
# 테스트
pytest tests/ -v
cd wasm-consensus && cargo test

# 린트
ruff check .
mypy .
```

## 라이선스

MIT License

---

**"The whole is greater than the sum of its parts"**
