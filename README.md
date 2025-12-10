# AGI Trinity

> Autonomous learning AGI that browses the web, solves CAPTCHAs, and evolves itself

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LFM2-VL](https://img.shields.io/badge/Model-LFM2--VL--1.6B-purple.svg)](https://huggingface.co/LiquidAI/LFM2-VL-1.6B)

[Demo](https://hwkim3330.github.io/agi) | [Documentation (한국어)](docs/ko/README.md)

---

## Core Systems

### 1. Life Agent - Vision-Based Autonomous Learning

An agent that looks at screenshots and decides what to do next (VLA-style).

```bash
python life_agents/life_agent_v4.py
```

| Version | Features |
|---------|----------|
| v1 | Basic browser exploration |
| v2 | Purpose-driven learning (growth, value creation, understanding) |
| v3 | CAPTCHA solving + Browser Use learning |
| **v4** | **Vision-based control** - Screenshot → Coordinates → Click |

### 2. Trinity - Multi-Model Consensus

Orchestrates multiple AI models (Claude, Gemini, GPT) to reach consensus.

```bash
python trinity/trinity.py ask "your question"
python trinity/trinity.py ask "complex problem" --strategy synthesis
```

**Consensus Strategies:**
- `vote`: Select best response
- `synthesis`: Combine all insights
- `fanout`: Return all responses

### 3. Browser Agent

```bash
python browser/browser_agent.py
```

## Project Structure

```
agi/
├── life_agents/           # Autonomous learning agents
│   ├── life_agent.py      # v1 - Basic
│   ├── life_agent_v2.py   # v2 - Purpose-driven
│   ├── life_agent_v3.py   # v3 - CAPTCHA + Browser Use
│   └── life_agent_v4.py   # v4 - Vision-based (latest)
│
├── trinity/               # Multi-model consensus
│   └── trinity.py         # Multi-Agent Orchestrator
│
├── browser/               # Browser automation
│   ├── browser_agent.py
│   ├── browser_learner.py
│   └── computer_use_learner.py
│
├── learners/              # Learning agents
│   ├── fast_learner.py
│   └── trend_learner.py
│
├── agents/                # Model adapters
│   └── lfm2_adapter.py    # LFM2-VL model
│
├── core/                  # Core engines
│   └── continual_learning.py
│
└── config/                # Configuration
    └── agents.yaml        # Trinity agent config
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AGI Trinity System                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Life Agent   │  │   Trinity    │  │   Browser    │       │
│  │  (Vision)    │  │ (Consensus)  │  │    Agent     │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                  │               │
│         └─────────────────┼──────────────────┘               │
│                           ▼                                  │
│            ┌─────────────────────────────┐                  │
│            │       LFM2-VL-1.6B         │                  │
│            │   (Local Vision-Language)   │                  │
│            └─────────────┬───────────────┘                  │
│                          │                                   │
│        ┌─────────────────┼─────────────────┐                │
│        ▼                 ▼                 ▼                │
│  ┌──────────┐    ┌──────────────┐   ┌──────────────┐       │
│  │Knowledge │    │  Continual   │   │    Claude    │       │
│  │   Base   │    │   Learning   │   │  UltraThink  │       │
│  └──────────┘    └──────────────┘   └──────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Life Goals

The agent has fundamental purposes that drive its behavior:

| Goal | Description | Activities |
|------|-------------|------------|
| **Growth** | Continuously learn and improve | Learning, exploring, practicing |
| **Value Creation** | Generate useful insights | Analyzing, creating, sharing |
| **Understanding** | Comprehend how the world works | Researching, connecting, synthesizing |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/hwkim3330/agi.git
cd agi

# 2. Install dependencies
pip install -r requirements.txt
pip install playwright && playwright install chromium

# 3. Run vision-based autonomous agent
python life_agents/life_agent_v4.py

# 4. Use Trinity consensus
python trinity/trinity.py ask "future of AI"
```

## Key Features

### Vision-Based Browser Control (v4)

```python
class VisionBrowserAgent:
    async def analyze_screen(self, screenshot, goal):
        """Vision model decides next action from screenshot"""
        # Returns: ACTION, X, Y coordinates

    async def execute_action(self, page, action):
        await page.mouse.click(action["x"], action["y"])
```

### CAPTCHA Solving

- Cloudflare Turnstile
- reCAPTCHA
- hCaptcha
- Vision model directly identifies checkbox coordinates

### Action Pattern Learning

```python
def record_success(self, goal, actions):
    """Save successful action sequences for future reference"""
    pattern = {"goal": goal, "actions": actions}
    self.success_patterns.append(pattern)
```

## Learning Sources

| Category | Sources |
|----------|---------|
| Tech | HackerNews, Lobste.rs |
| Academic | arXiv AI/ML/CL |
| Community | Reddit ML/Python |
| Korean | Naver IT News |
| General | Wikipedia Random |

## Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| GPU | 4GB VRAM | 8GB+ VRAM |
| RAM | 8GB | 16GB+ |
| Python | 3.10 | 3.11+ |

## Roadmap

- [x] LFM2-VL model integration
- [x] Continual learning engine
- [x] Browser automation
- [x] Vision-based CAPTCHA solving
- [x] Trinity multi-model consensus
- [ ] Self-improvement capability
- [ ] Distributed learning
- [ ] Web UI dashboard

---

**"Learn, grow, evolve - autonomously"**
