# 🤖 AGI Trinity - Multi-Agent AI Orchestrator

> "Three minds, one consciousness" - Evangelion-inspired AI collaboration framework

## 🎯 Vision

AGI Trinity는 Claude Code, Gemini, 그리고 Codex(OpenAI)를 하나의 통합된 지능체처럼 운영하는 오케스트레이터입니다. 각 AI의 고유한 강점을 활용하여 복잡한 문제를 다각도로 분석하고 최적의 솔루션을 도출합니다.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     INPUT COMMAND                       │
└──────────────────┬──────────────────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│ CLAUDE  │    │ GEMINI  │    │ CODEX   │
│ Code    │    │ Pro     │    │ GPT-4   │
│ Expert  │    │ Analyst │    │ Creator │
└─────────┘    └─────────┘    └─────────┘
    │              │              │
    └──────────────┼──────────────┘
                   │
                   ▼
          ┌─────────────────┐
          │   CONSENSUS     │
          │   SYNTHESIS     │
          └─────────────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  FINAL ACTION   │
          └─────────────────┘
```

## 🚀 Key Features

- **🔄 Parallel Processing**: 3개 AI가 동시에 문제 분석
- **🗳️ Democratic Consensus**: 투표 및 합의 메커니즘으로 최적해 도출
- **⚡ Zero-Friction Integration**: 기존 CLI 도구들을 그대로 활용
- **🎭 Role Specialization**: 각 AI의 특성에 맞는 역할 분담
- **📊 Real-time Monitoring**: 각 AI의 처리 과정 실시간 관찰

## 🎯 Use Cases

### 💼 Business Applications
- 복잡한 기술적 의사결정 지원
- 다각도 시장 분석 및 전략 수립
- 창작과 분석이 결합된 콘텐츠 제작

### 🔬 Research Applications
- Multi-agent AI 협업 메커니즘 연구
- 집단지성 알고리즘 개발
- AI 앙상블 효과 분석

### 🛠️ Development Applications
- 코드 리뷰 및 최적화
- 아키텍처 설계 검증
- 문제 해결 방법론 개발

## 🏃‍♂️ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your AI CLI tools
cp config/agents.yaml.example config/agents.yaml
# Edit agents.yaml with your CLI commands

# 3. Basic usage
python trinity.py ask "How to optimize TSN network for industrial IoT?"

# 4. Advanced orchestration
echo "complex_task.log" | tee >(python trinity.py observe --agent claude) \
                                >(python trinity.py observe --agent gemini) \
                          | python trinity.py synthesize --strategy vote
```

## 📁 Project Structure

```
agi/
├── trinity.py              # Main orchestrator
├── agents/                 # Agent adapters
│   ├── claude_adapter.py
│   ├── gemini_adapter.py
│   └── codex_adapter.py
├── core/                   # Core orchestration logic
│   ├── consensus.py        # Voting and synthesis
│   ├── router.py          # Request routing
│   └── monitor.py         # Real-time monitoring
├── config/                 # Configuration
│   └── agents.yaml        # Agent definitions
├── scripts/               # Utility scripts
│   ├── nonint.sh         # Non-interactive wrapper
│   └── agent_wrap.expect # REPL wrapper
└── examples/              # Usage examples
```

## 🎭 Agent Roles

| Agent | Specialty | Best For |
|-------|-----------|----------|
| **Claude Code** | 🔧 Technical Implementation | Code analysis, debugging, system design |
| **Gemini Pro** | 📊 Data Analysis | Research, analysis, fact-checking |
| **Codex GPT-4** | 🎨 Creative Problem Solving | Innovation, creative solutions, brainstorming |

## 🌟 Goals

1. **Research**: Advance multi-agent AI collaboration
2. **Practical**: Solve complex real-world problems
3. **Commercial**: Enable AI-powered consulting solutions

---

*"The whole is greater than the sum of its parts"* - Building AGI through collaboration, not competition.