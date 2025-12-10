#!/usr/bin/env python3
"""
🌟 Life Agent v2 - 더 다양한 소스, 실제 Claude 울트라씽킹
로컬 모델(LFM2)로 빠른 판단, Claude API로 깊은 성찰
"""
import asyncio
import json
import random
import sys
import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("pip install playwright && playwright install chromium")
    sys.exit(1)

# Claude API (optional)
try:
    import anthropic
    HAS_CLAUDE = True
except ImportError:
    HAS_CLAUDE = False
    print("⚠️ anthropic 없음 - ultrathink 비활성화")

sys.path.insert(0, str(Path(__file__).parent))

# 다양한 학습 소스
LEARNING_SOURCES = {
    "tech_news": [
        "https://news.ycombinator.com/newest",
        "https://www.reddit.com/r/MachineLearning/new/",
        "https://www.reddit.com/r/artificial/new/",
        "https://techcrunch.com/category/artificial-intelligence/",
    ],
    "academic": [
        "https://arxiv.org/list/cs.AI/recent",
        "https://arxiv.org/list/cs.LG/recent",
        "https://arxiv.org/list/cs.CL/recent",
    ],
    "korean": [
        "https://news.naver.com/section/105",  # IT/과학
        "https://www.reddit.com/r/korea/new/",
    ],
    "general": [
        "https://en.wikipedia.org/wiki/Special:Random",
        "https://www.bbc.com/news/technology",
    ],
    "programming": [
        "https://www.reddit.com/r/Python/new/",
        "https://dev.to/t/ai",
        "https://lobste.rs/",
    ]
}

# 근본 목표
LIFE_PURPOSES = {
    "growth": {
        "name": "지속적 성장",
        "description": "끊임없이 배우고 발전한다",
        "preferred_sources": ["tech_news", "academic", "programming"],
        "search_topics": ["machine learning tutorial", "AI research", "deep learning", "neural networks"]
    },
    "value_creation": {
        "name": "가치 창출",
        "description": "유용한 인사이트를 생성한다",
        "preferred_sources": ["tech_news", "general"],
        "search_topics": ["startup ideas", "innovation", "problem solving", "creative coding"]
    },
    "understanding": {
        "name": "세상 이해",
        "description": "세상이 어떻게 돌아가는지 이해한다",
        "preferred_sources": ["general", "korean", "academic"],
        "search_topics": ["philosophy", "science news", "economics", "society trends"]
    }
}


class LifeAgentV2:
    """인생 목표를 가진 자율 AI v2"""

    def __init__(self):
        self.browser = None
        self.page = None
        self.agi = None

        # 인생 상태
        self.birth_time = datetime.now()
        self.life_purpose = random.choice(list(LIFE_PURPOSES.keys()))

        # 지식 저장소
        self.knowledge_base = []
        self.insights = []
        self.questions = []
        self.visited_urls = set()

        # 통계
        self.total_pages = 0
        self.total_actions = 0
        self.thinking_sessions = 0
        self.ultrathink_count = 0

        # 디렉토리
        self.data_dir = Path("/home/kim/agi/life_agent_data")
        self.data_dir.mkdir(exist_ok=True)

        # Claude 클라이언트
        self.claude = None
        if HAS_CLAUDE:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self.claude = anthropic.Anthropic(api_key=api_key)
                print("✨ Claude API 연결됨 - ultrathink 활성화")

        self._load_state()

    def _load_state(self):
        """상태 로드"""
        state_file = self.data_dir / "life_state_v2.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                    self.knowledge_base = state.get("knowledge", [])[-100:]
                    self.insights = state.get("insights", [])[-50:]
                    self.questions = state.get("questions", [])[-30:]
                    self.visited_urls = set(state.get("visited_urls", [])[-500:])
                    self.total_pages = state.get("total_pages", 0)
                    self.thinking_sessions = state.get("thinking_sessions", 0)
                    self.ultrathink_count = state.get("ultrathink_count", 0)
                print(f"📚 Loaded: {len(self.knowledge_base)} knowledge, {len(self.insights)} insights")
            except Exception as e:
                print(f"⚠️ Load failed: {e}")

    def _save_state(self):
        """상태 저장"""
        state = {
            "life_purpose": self.life_purpose,
            "knowledge": self.knowledge_base[-100:],
            "insights": self.insights[-50:],
            "questions": self.questions[-30:],
            "visited_urls": list(self.visited_urls)[-500:],
            "total_pages": self.total_pages,
            "thinking_sessions": self.thinking_sessions,
            "ultrathink_count": self.ultrathink_count,
            "last_save": datetime.now().isoformat()
        }
        with open(self.data_dir / "life_state_v2.json", 'w') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    async def setup(self):
        """초기화"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=['--no-sandbox'],
            slow_mo=50
        )
        self.context = await self.browser.new_context(viewport={'width': 1400, 'height': 900})
        self.page = await self.context.new_page()
        print("🌐 Browser ready")

        print("🧠 Loading local brain (LFM2)...")
        from agents.lfm2_adapter import LFM2VLAdapter, LFM2Config
        config = LFM2Config(model_id="LiquidAI/LFM2-VL-1.6B", enable_continual_learning=True)
        self.agi = LFM2VLAdapter(lfm2_config=config)
        await self.agi.load_model()

        purpose = LIFE_PURPOSES[self.life_purpose]
        print(f"✨ Life Agent v2 ready! Purpose: {purpose['name']}")

    async def local_think(self, prompt: str) -> str:
        """로컬 모델로 빠른 생각"""
        try:
            response = await self.agi.execute(prompt)
            return response.content[:400]
        except Exception as e:
            return f"생각 실패: {e}"

    def ultra_think(self, context: str) -> str:
        """Claude로 깊은 생각 (실제 API 호출)"""
        if not self.claude:
            return "Claude API 없음"

        self.ultrathink_count += 1
        purpose = LIFE_PURPOSES[self.life_purpose]

        prompt = f"""나는 자율 학습 AI다. 현재 상태:
- 인생 목표: {purpose['name']} - {purpose['description']}
- 학습한 페이지: {self.total_pages}개
- 생성한 인사이트: {len(self.insights)}개

최근 학습 내용:
{context[:1500]}

최근 인사이트들:
{chr(10).join(self.insights[-5:])}

깊이 성찰해주세요:
1. 이 지식들에서 발견되는 패턴은?
2. 앞으로 탐구해야 할 새로운 방향은?
3. 이 지식을 어떻게 가치로 전환할 수 있을까?
4. 던져야 할 근본적인 질문은?

100자 내로 핵심 통찰을:"""

        try:
            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text

            # 통찰 저장
            self.insights.append(f"[UltraThink] {result[:200]}")

            # 파일로도 저장
            thinking_file = self.data_dir / f"ultrathink_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(thinking_file, 'w') as f:
                f.write(f"Context:\n{context}\n\nInsight:\n{result}")

            return result
        except Exception as e:
            return f"UltraThink 실패: {e}"

    def get_random_source(self) -> str:
        """목적에 맞는 랜덤 소스 선택"""
        purpose = LIFE_PURPOSES[self.life_purpose]
        preferred = purpose['preferred_sources']

        # 70% 확률로 선호 소스, 30% 전체 랜덤
        if random.random() < 0.7:
            category = random.choice(preferred)
        else:
            category = random.choice(list(LEARNING_SOURCES.keys()))

        return random.choice(LEARNING_SOURCES[category])

    async def decide_what_to_do(self) -> dict:
        """무엇을 할지 결정"""
        purpose = LIFE_PURPOSES[self.life_purpose]
        recent_knowledge = " ".join([k[:50] for k in self.knowledge_base[-3:]])

        # 가끔은 검색으로
        if random.random() < 0.3:
            topic = random.choice(purpose['search_topics'])
            return {"action": "SEARCH", "target": topic, "reason": "목표 관련 검색"}

        # 대부분은 소스 탐험
        source = self.get_random_source()

        prompt = f"""나의 목표: {purpose['name']}
최근 배운 것: {recent_knowledge[:150]}
방문할 곳: {source}

다음 행동:
1. EXPLORE - 이 소스 탐험
2. SEARCH [주제] - 관련 검색
3. REFLECT - 지금까지 성찰

한 단어로 (EXPLORE/SEARCH/REFLECT):"""

        result = await self.local_think(prompt)

        upper = result.upper()
        if "REFLECT" in upper:
            return {"action": "REFLECT", "target": "", "reason": result[:50]}
        elif "SEARCH" in upper:
            topic = random.choice(purpose['search_topics'])
            return {"action": "SEARCH", "target": topic, "reason": result[:50]}
        else:
            return {"action": "EXPLORE", "target": source, "reason": result[:50]}

    async def execute_action(self, action: str, target: str):
        """행동 실행"""
        self.total_actions += 1

        if action == "EXPLORE":
            await self.explore_source(target)
        elif action == "SEARCH":
            await self.search_and_learn(target)
        elif action == "REFLECT":
            await self.reflect()

    async def explore_source(self, url: str):
        """소스 탐험"""
        if url in self.visited_urls:
            # 이미 방문한 URL이면 다른 것 선택
            url = self.get_random_source()

        print(f"\n🔭 Exploring: {url[:60]}")
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)

            self.visited_urls.add(url)
            await self._read_and_learn()

            # 링크 클릭 (최대 2개)
            for _ in range(2):
                if random.random() < 0.6:
                    await self._click_interesting_link()
                    await asyncio.sleep(2)
                    await self._read_and_learn()

        except Exception as e:
            print(f"   ❌ {e}")
            self.page = await self.context.new_page()

    async def search_and_learn(self, query: str):
        """검색하고 학습"""
        print(f"\n🔍 Searching: {query}")
        try:
            # DuckDuckGo 사용
            await self.page.goto(f"https://duckduckgo.com/?q={query.replace(' ', '+')}", timeout=30000)
            await asyncio.sleep(2)

            # 결과 클릭
            links = await self.page.query_selector_all('a[data-testid="result-title-a"]')
            for link in links[:2]:
                try:
                    href = await link.get_attribute("href")
                    if href and href not in self.visited_urls:
                        await link.click(timeout=5000)
                        await asyncio.sleep(2)
                        self.visited_urls.add(self.page.url)
                        await self._read_and_learn()
                        await self.page.go_back(timeout=5000)
                except:
                    continue

        except Exception as e:
            print(f"   ❌ {e}")

    async def _click_interesting_link(self):
        """흥미로운 링크 클릭"""
        try:
            links = await self.page.query_selector_all('a[href]')
            interesting = []

            keywords = ["AI", "machine", "learn", "research", "study", "tech", "science",
                       "data", "python", "neural", "model", "algorithm", "news"]

            for link in links[:30]:
                try:
                    if not await link.is_visible():
                        continue
                    text = await link.inner_text()
                    href = await link.get_attribute("href")

                    if not href or href in self.visited_urls:
                        continue
                    if len(text.strip()) < 5:
                        continue

                    for kw in keywords:
                        if kw.lower() in text.lower():
                            interesting.append(link)
                            break
                except:
                    continue

            if interesting:
                link = random.choice(interesting[:5])
                text = await link.inner_text()
                await link.click(timeout=5000)
                print(f"   🔗 Clicked: {text[:40]}")
                self.visited_urls.add(self.page.url)

        except Exception as e:
            print(f"   ❌ Click failed: {e}")

    async def _read_and_learn(self):
        """페이지 읽고 학습"""
        try:
            title = await self.page.title()
            url = self.page.url

            text = await self.page.evaluate("""
                () => {
                    const main = document.querySelector('article, main, .content, .post-content') || document.body;
                    return main.innerText.slice(0, 3000);
                }
            """)

            if len(text) < 100:
                return

            self.total_pages += 1

            # 요약
            summary = await self.local_think(f"이 내용의 핵심을 50자로 요약: {text[:1000]}")

            # 중복 체크
            if summary[:50] not in [k[:50] for k in self.knowledge_base[-10:]]:
                self.knowledge_base.append(summary)
                print(f"   📖 Read: {title[:40]}")
                print(f"   💡 Learned: {summary[:80]}")

                # 질문 생성 (가끔)
                if random.random() < 0.2:
                    question = await self.local_think(f"이 내용에서 떠오르는 질문 하나: {summary}")
                    self.questions.append(question)
                    print(f"   ❓ Question: {question[:60]}")
            else:
                print(f"   ⏭️ Skip duplicate: {title[:30]}")

        except Exception as e:
            print(f"   ❌ Read failed: {e}")

    async def reflect(self):
        """성찰"""
        print(f"\n🪞 Reflecting...")
        self.thinking_sessions += 1

        recent = " ".join(self.knowledge_base[-10:])

        # 로컬 빠른 성찰
        reflection = await self.local_think(f"지금까지 배운 것: {recent[:800]}\n\n가장 중요한 교훈은?")
        print(f"   💭 Local: {reflection[:100]}")

        # 10회마다 Claude 울트라씽킹
        if self.thinking_sessions % 10 == 0 and self.claude:
            print(f"   🧠 UltraThinking with Claude...")
            ultra_result = self.ultra_think(recent)
            print(f"   ✨ Ultra: {ultra_result[:150]}")

        # 상태 저장
        self._save_state()

        # 통계
        uptime = datetime.now() - self.birth_time
        print(f"\n📊 Life Stats:")
        print(f"   ⏱️ Uptime: {uptime}")
        print(f"   📚 Pages: {self.total_pages}")
        print(f"   💡 Insights: {len(self.insights)}")
        print(f"   ❓ Questions: {len(self.questions)}")
        print(f"   🧠 UltraThinks: {self.ultrathink_count}")

    async def live(self):
        """살아가기"""
        purpose = LIFE_PURPOSES[self.life_purpose]
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║        🌟 LIFE AGENT v2 - Autonomous AI Life 🌟           ║
║                                                           ║
║   Purpose: {purpose['name']:^43} ║
║   "I learn, I grow, I create value"                       ║
║   Claude UltraThink: {'✅ Enabled' if self.claude else '❌ Disabled':^36} ║
╚═══════════════════════════════════════════════════════════╝
""")

        cycle = 0
        while True:
            cycle += 1
            print(f"\n{'='*60}")
            print(f"🔄 Cycle {cycle} | Pages: {self.total_pages} | Insights: {len(self.insights)}")

            try:
                decision = await self.decide_what_to_do()
                action = decision['action']
                target = decision['target']

                print(f"🎯 Decision: {action} - {target[:50] if target else 'N/A'}")

                await self.execute_action(action, target)

                # 주기적 성찰
                if cycle % 7 == 0:
                    await self.reflect()

                # 상태 저장
                if cycle % 5 == 0:
                    self._save_state()

            except Exception as e:
                print(f"❌ Error: {e}")
                try:
                    self.page = await self.context.new_page()
                except:
                    pass

            wait = random.randint(8, 20)
            print(f"⏳ Next action in {wait}s...")
            await asyncio.sleep(wait)

    async def close(self):
        self._save_state()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


async def main():
    agent = LifeAgentV2()

    try:
        await agent.setup()
        await agent.live()
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Life paused.")
        print(f"📚 Knowledge: {len(agent.knowledge_base)}")
        print(f"💡 Insights: {len(agent.insights)}")
        print(f"🧠 UltraThinks: {agent.ultrathink_count}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
