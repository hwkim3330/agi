#!/usr/bin/env python3
"""
🌟 Life Agent - 인생 목표를 가진 자율 AI
근본적 목표: 성장, 학습, 가치 창출
로컬 모델(LFM2)로 계획, Claude로 울트라씽킹
"""
import asyncio
import json
import random
import sys
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("pip install playwright && playwright install chromium")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))


# 근본 목표들 (Life Goals)
LIFE_PURPOSES = {
    "growth": {
        "name": "지속적 성장",
        "description": "끊임없이 배우고 발전한다",
        "metrics": ["pages_read", "knowledge_gained", "skills_learned"],
        "activities": ["learn", "explore", "practice"]
    },
    "value_creation": {
        "name": "가치 창출",
        "description": "유용한 것을 만들고 공유한다",
        "metrics": ["ideas_generated", "insights_shared", "problems_solved"],
        "activities": ["create", "analyze", "share"]
    },
    "understanding": {
        "name": "세상 이해",
        "description": "세상이 어떻게 돌아가는지 이해한다",
        "metrics": ["topics_explored", "connections_made", "patterns_found"],
        "activities": ["research", "connect", "synthesize"]
    }
}


class LifeAgent:
    """인생 목표를 가진 자율 AI"""

    def __init__(self):
        self.browser = None
        self.page = None
        self.agi = None  # 로컬 모델 (LFM2)

        # 인생 상태
        self.birth_time = datetime.now()
        self.life_purpose = random.choice(list(LIFE_PURPOSES.keys()))

        # 지식 저장소
        self.knowledge_base = []
        self.insights = []
        self.ideas = []
        self.questions = []

        # 통계
        self.total_pages = 0
        self.total_actions = 0
        self.thinking_sessions = 0

        # 디렉토리
        self.data_dir = Path("/home/kim/agi/life_agent_data")
        self.data_dir.mkdir(exist_ok=True)

        # 상태 로드
        self._load_state()

    def _load_state(self):
        """이전 상태 로드"""
        state_file = self.data_dir / "life_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                    self.knowledge_base = state.get("knowledge", [])[-100:]
                    self.insights = state.get("insights", [])[-50:]
                    self.total_pages = state.get("total_pages", 0)
                    self.thinking_sessions = state.get("thinking_sessions", 0)
                print(f"📚 Loaded: {len(self.knowledge_base)} knowledge, {len(self.insights)} insights")
            except:
                pass

    def _save_state(self):
        """상태 저장"""
        state = {
            "life_purpose": self.life_purpose,
            "knowledge": self.knowledge_base[-100:],
            "insights": self.insights[-50:],
            "ideas": self.ideas[-30:],
            "questions": self.questions[-20:],
            "total_pages": self.total_pages,
            "thinking_sessions": self.thinking_sessions,
            "last_save": datetime.now().isoformat()
        }
        with open(self.data_dir / "life_state.json", 'w') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    async def setup(self):
        """초기화"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=['--no-sandbox'],
            slow_mo=30
        )
        self.context = await self.browser.new_context(viewport={'width': 1400, 'height': 900})
        self.page = await self.context.new_page()
        print("🌐 Browser ready")

        print("🧠 Loading local brain (LFM2)...")
        from agents.lfm2_adapter import LFM2VLAdapter, LFM2Config
        config = LFM2Config(model_id="LiquidAI/LFM2-VL-1.6B", enable_continual_learning=True)
        self.agi = LFM2VLAdapter(lfm2_config=config)
        await self.agi.load_model()
        print(f"✨ Life Agent ready! Purpose: {LIFE_PURPOSES[self.life_purpose]['name']}")

    async def local_think(self, prompt: str) -> str:
        """로컬 모델로 빠른 생각"""
        try:
            response = await self.agi.execute(prompt)
            return response.content[:300]
        except Exception as e:
            return f"생각 실패: {e}"

    def ultra_think(self, context: str) -> str:
        """Claude로 깊은 생각 (비동기 백그라운드)"""
        self.thinking_sessions += 1

        # 깊은 생각을 위한 프롬프트
        thinking_prompt = f"""나는 자율 학습 AI다. 현재 상태:
- 인생 목표: {LIFE_PURPOSES[self.life_purpose]['name']}
- 학습한 페이지: {self.total_pages}개
- 얻은 인사이트: {len(self.insights)}개
- 최근 학습: {context[:500]}

깊이 생각해보자:
1. 내가 배운 것 중 가장 중요한 패턴은?
2. 앞으로 무엇을 더 탐구해야 할까?
3. 이 지식으로 무엇을 만들 수 있을까?
4. 근본적인 질문은 무엇인가?

통찰을 50자 내로:"""

        # 파일에 저장해서 나중에 처리
        thinking_file = self.data_dir / f"thinking_{datetime.now().strftime('%H%M%S')}.txt"
        with open(thinking_file, 'w') as f:
            f.write(thinking_prompt)

        # TODO: Claude API 호출 (별도 프로세스로)
        return "🧠 Deep thinking queued..."

    async def decide_what_to_do(self) -> dict:
        """무엇을 할지 결정 (로컬 모델)"""
        purpose = LIFE_PURPOSES[self.life_purpose]

        # 최근 상황 요약
        recent_knowledge = " ".join([k[:50] for k in self.knowledge_base[-3:]])

        prompt = f"""나의 목표: {purpose['name']} - {purpose['description']}
최근 학습: {recent_knowledge[:200]}
학습량: {self.total_pages}페이지

다음 행동을 선택해:
1. LEARN [주제] - 새로운 것 학습
2. EXPLORE [URL] - 웹 탐험
3. SEARCH [검색어] - 정보 검색
4. REFLECT - 성찰하기

형식: ACTION: [행동] TARGET: [대상]
한 줄로:"""

        result = await self.local_think(prompt)

        action = "LEARN"
        target = "artificial intelligence"

        upper = result.upper()
        if "LEARN" in upper:
            action = "LEARN"
        elif "EXPLORE" in upper:
            action = "EXPLORE"
        elif "SEARCH" in upper:
            action = "SEARCH"
        elif "REFLECT" in upper:
            action = "REFLECT"

        if "TARGET:" in result:
            target = result.split("TARGET:")[-1].strip()[:50]

        return {"action": action, "target": target, "reason": result[:100]}

    async def execute_action(self, action: str, target: str):
        """행동 실행"""
        self.total_actions += 1

        if action == "LEARN":
            await self.learn_topic(target)
        elif action == "EXPLORE":
            await self.explore_url(target)
        elif action == "SEARCH":
            await self.search_and_learn(target)
        elif action == "REFLECT":
            await self.reflect()

    async def learn_topic(self, topic: str):
        """주제 학습"""
        print(f"\n📚 Learning: {topic}")

        # Google 검색
        search_url = f"https://www.google.com/search?q={topic.replace(' ', '+')}"
        try:
            await self.page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)

            # 첫 번째 결과 클릭
            links = await self.page.query_selector_all('a h3')
            if links:
                await links[0].click(timeout=5000)
                await asyncio.sleep(2)
                await self._read_and_learn()

        except Exception as e:
            print(f"   ❌ {e}")

    async def explore_url(self, url: str):
        """URL 탐험"""
        if not url.startswith("http"):
            url = f"https://{url}"

        print(f"\n🔭 Exploring: {url[:50]}")
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)
            await self._read_and_learn()

            # 흥미로운 링크 클릭
            links = await self.page.query_selector_all('a[href]')
            interesting = []
            for link in links[:20]:
                try:
                    text = await link.inner_text()
                    for kw in ["AI", "learn", "research", "data", "python", "tech"]:
                        if kw.lower() in text.lower():
                            interesting.append(link)
                            break
                except:
                    continue

            if interesting:
                await random.choice(interesting[:5]).click(timeout=5000)
                await asyncio.sleep(2)
                await self._read_and_learn()

        except Exception as e:
            print(f"   ❌ {e}")

    async def search_and_learn(self, query: str):
        """검색하고 학습"""
        print(f"\n🔍 Searching: {query}")
        try:
            await self.page.goto("https://duckduckgo.com", timeout=30000)
            await asyncio.sleep(1)

            await self.page.fill('input[name="q"]', query)
            await self.page.keyboard.press("Enter")
            await asyncio.sleep(2)

            # 결과 클릭
            links = await self.page.query_selector_all('a[data-testid="result-title-a"]')
            for link in links[:2]:
                try:
                    await link.click(timeout=5000)
                    await asyncio.sleep(2)
                    await self._read_and_learn()
                    await self.page.go_back(timeout=5000)
                except:
                    continue

        except Exception as e:
            print(f"   ❌ {e}")

    async def _read_and_learn(self):
        """현재 페이지 읽고 학습"""
        try:
            title = await self.page.title()
            text = await self.page.evaluate("""
                () => {
                    const main = document.querySelector('article, main, .content') || document.body;
                    return main.innerText.slice(0, 2000);
                }
            """)

            if len(text) < 100:
                return

            self.total_pages += 1

            # 요약
            summary = await self.local_think(f"핵심만 30자로: {text[:800]}")
            self.knowledge_base.append(summary)

            print(f"   📖 Read: {title[:40]}")
            print(f"   💡 Learned: {summary[:60]}")

            # 인사이트 생성 (가끔)
            if random.random() < 0.3:
                insight = await self.local_think(
                    f"이 내용에서 발견한 통찰 하나: {summary}"
                )
                self.insights.append(insight)
                print(f"   ✨ Insight: {insight[:50]}")

        except Exception as e:
            print(f"   ❌ Read failed: {e}")

    async def reflect(self):
        """성찰 - 깊은 생각"""
        print(f"\n🪞 Reflecting...")

        # 최근 지식 요약
        recent = " ".join(self.knowledge_base[-5:])

        # 로컬 모델로 빠른 성찰
        reflection = await self.local_think(
            f"지금까지 배운 것: {recent[:500]}\n\n가장 중요한 교훈은?"
        )
        print(f"   💭 {reflection[:80]}")

        # 울트라씽킹 (백그라운드)
        self.ultra_think(recent)

        # 상태 저장
        self._save_state()

        # 통계
        uptime = datetime.now() - self.birth_time
        print(f"\n📊 Life Stats:")
        print(f"   ⏱️ Uptime: {uptime}")
        print(f"   📚 Pages: {self.total_pages}")
        print(f"   💡 Insights: {len(self.insights)}")
        print(f"   🧠 Thinking sessions: {self.thinking_sessions}")

    async def live(self):
        """살아가기"""
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║          🌟 LIFE AGENT - Autonomous AI Life 🌟            ║
║                                                           ║
║   Purpose: {LIFE_PURPOSES[self.life_purpose]['name']:^43} ║
║   "I learn, I grow, I create"                             ║
╚═══════════════════════════════════════════════════════════╝
""")

        cycle = 0
        while True:
            cycle += 1
            print(f"\n{'='*60}")
            print(f"🔄 Cycle {cycle} | Pages: {self.total_pages} | Insights: {len(self.insights)}")

            try:
                # 무엇을 할지 결정
                decision = await self.decide_what_to_do()
                action = decision['action']
                target = decision['target']

                print(f"🎯 Decision: {action} - {target[:30]}")

                # 행동 실행
                await self.execute_action(action, target)

                # 주기적 성찰
                if cycle % 5 == 0:
                    await self.reflect()

                # 상태 저장
                if cycle % 3 == 0:
                    self._save_state()

            except Exception as e:
                print(f"❌ Error: {e}")
                try:
                    self.page = await self.context.new_page()
                except:
                    pass

            # 휴식
            wait = random.randint(5, 15)
            print(f"⏳ Next action in {wait}s...")
            await asyncio.sleep(wait)

    async def close(self):
        self._save_state()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


async def main():
    agent = LifeAgent()

    try:
        await agent.setup()
        await agent.live()
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Life paused.")
        print(f"📚 Knowledge gained: {len(agent.knowledge_base)}")
        print(f"💡 Insights: {len(agent.insights)}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
