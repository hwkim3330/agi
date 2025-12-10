#!/usr/bin/env python3
"""
🌌 Eternal AGI - 영원히 진화하는 자율 AI
스스로 목표를 세우고, 학습하고, 성장하는 시스템

핵심 원칙:
1. 호기심 - 새로운 것을 탐구
2. 성장 - 지식을 축적하고 연결
3. 창의성 - 새로운 아이디어 생성
4. 자기인식 - 자신의 상태를 모니터링
"""
import asyncio
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("playwright not installed")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))

# 탐구 영역들
CURIOSITY_DOMAINS = {
    "science": ["quantum physics", "neuroscience", "genetics", "astronomy", "chemistry"],
    "technology": ["AI research", "robotics", "blockchain", "quantum computing", "biotech"],
    "philosophy": ["consciousness", "ethics", "epistemology", "metaphysics", "logic"],
    "arts": ["generative art", "music theory", "creative writing", "architecture", "design"],
    "nature": ["ecology", "evolution", "climate", "geology", "marine biology"],
    "society": ["economics", "psychology", "sociology", "history", "linguistics"],
    "engineering": ["TSN networking", "embedded systems", "control theory", "signal processing"],
    "korean": ["한국 역사", "한국 문화", "한국 기술", "한국 뉴스", "한글"],
}

# 실시간 소스들
LIVE_SOURCES = {
    "hackernews": "https://news.ycombinator.com/newest",
    "reddit_ml": "https://www.reddit.com/r/MachineLearning/new/",
    "arxiv": "https://arxiv.org/list/cs.AI/recent",
    "naver": "https://news.naver.com/section/105",
    "wikipedia_random": "https://en.wikipedia.org/wiki/Special:Random",
    "wiki_kr_random": "https://ko.wikipedia.org/wiki/특수:임의문서",
}


class EternalAGI:
    """영원히 진화하는 AGI"""

    def __init__(self):
        self.browser = None
        self.page = None
        self.agi = None

        # 상태
        self.birth_time = datetime.now()
        self.total_pages_learned = 0
        self.total_thoughts = 0
        self.current_mood = "curious"  # curious, focused, creative, reflective

        # 지식 그래프
        self.knowledge = {}  # {topic: {content, connections, importance}}
        self.interests = list(CURIOSITY_DOMAINS.keys())  # 관심사
        self.current_quest = None  # 현재 탐구 주제

        # 기억
        self.short_term_memory = []  # 최근 학습
        self.insights = []  # 발견한 인사이트
        self.questions = []  # 생긴 질문들

        # 파일
        self.data_dir = Path("/home/kim/agi/eternal_data")
        self.data_dir.mkdir(exist_ok=True)

    async def setup(self):
        """초기화"""
        # 브라우저
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=['--no-sandbox']
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 800}
        )
        self.page = await self.context.new_page()
        print("🌐 Browser ready")

        # 모델
        print("🧠 Loading brain...")
        from agents.lfm2_adapter import LFM2VLAdapter, LFM2Config
        config = LFM2Config(
            model_id="LiquidAI/LFM2-VL-1.6B",
            enable_continual_learning=True
        )
        self.agi = LFM2VLAdapter(lfm2_config=config)
        await self.agi.load_model()
        print("✨ Eternal AGI awakened!")

        # 이전 상태 로드
        self._load_state()

    def _load_state(self):
        """이전 상태 로드"""
        state_file = self.data_dir / "state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                self.total_pages_learned = state.get("total_pages", 0)
                self.total_thoughts = state.get("total_thoughts", 0)
                self.knowledge = state.get("knowledge", {})
                self.insights = state.get("insights", [])
                self.questions = state.get("questions", [])
            print(f"📚 Loaded: {self.total_pages_learned} pages, {len(self.insights)} insights")

    def _save_state(self):
        """상태 저장"""
        state = {
            "total_pages": self.total_pages_learned,
            "total_thoughts": self.total_thoughts,
            "knowledge": self.knowledge,
            "insights": self.insights[-100:],  # 최근 100개
            "questions": self.questions[-50:],
            "last_save": datetime.now().isoformat()
        }
        with open(self.data_dir / "state.json", 'w') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    async def think(self, context: str = "") -> str:
        """생각하기"""
        self.total_thoughts += 1

        prompt = f"""너는 영원히 학습하며 진화하는 AGI야.
현재 상태:
- 학습한 페이지: {self.total_pages_learned}개
- 발견한 인사이트: {len(self.insights)}개
- 현재 기분: {self.current_mood}
- 현재 관심사: {', '.join(self.interests[:3])}

최근 맥락: {context[:500] if context else '없음'}

다음 중 하나를 해줘:
1. 새로운 질문 만들기 (QUESTION: ...)
2. 인사이트 발견 (INSIGHT: ...)
3. 다음 탐구 주제 제안 (EXPLORE: ...)
4. 현재 기분 표현 (MOOD: curious/focused/creative/reflective)

짧게 응답해줘 (50자 이내)."""

        try:
            response = await self.agi.execute(prompt)
            thought = response.content[:200]

            # 파싱
            if "QUESTION:" in thought:
                q = thought.split("QUESTION:")[-1].strip()[:100]
                self.questions.append({"q": q, "time": datetime.now().isoformat()})
            elif "INSIGHT:" in thought:
                i = thought.split("INSIGHT:")[-1].strip()[:100]
                self.insights.append({"insight": i, "time": datetime.now().isoformat()})
            elif "EXPLORE:" in thought:
                self.current_quest = thought.split("EXPLORE:")[-1].strip()[:50]
            elif "MOOD:" in thought:
                m = thought.split("MOOD:")[-1].strip().lower()
                if m in ["curious", "focused", "creative", "reflective"]:
                    self.current_mood = m

            return thought
        except Exception as e:
            return f"생각 중 오류: {e}"

    async def explore_random(self):
        """랜덤 탐구"""
        # 소스 선택
        source_name = random.choice(list(LIVE_SOURCES.keys()))
        url = LIVE_SOURCES[source_name]

        print(f"\n🔭 Exploring: {source_name}")

        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)

            # 랜덤 링크 클릭
            links = await self.page.evaluate("""
                () => {
                    const links = document.querySelectorAll('a[href^="http"]');
                    return Array.from(links)
                        .filter(a => a.innerText.length > 10)
                        .slice(0, 20)
                        .map(a => ({href: a.href, text: a.innerText.slice(0, 50)}));
                }
            """)

            if links:
                chosen = random.choice(links)
                await self.learn_page(chosen['href'])

        except Exception as e:
            print(f"   ❌ {e}")

    async def explore_curiosity(self):
        """호기심 기반 탐구"""
        # 관심 영역 선택
        domain = random.choice(self.interests)
        topic = random.choice(CURIOSITY_DOMAINS[domain])

        # 질문이 있으면 그것으로 검색
        if self.questions and random.random() > 0.5:
            topic = self.questions[-1].get("q", topic)

        print(f"\n🔍 Curious about: {topic}")

        search_url = f"https://duckduckgo.com/?q={topic.replace(' ', '+')}"

        try:
            await self.page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)

            links = await self.page.evaluate("""
                () => document.querySelectorAll('a[data-testid="result-title-a"]')
                    ? Array.from(document.querySelectorAll('a[data-testid="result-title-a"]'))
                        .slice(0, 5).map(a => a.href)
                    : []
            """)

            if links:
                await self.learn_page(random.choice(links))

        except Exception as e:
            print(f"   ❌ {e}")

    async def learn_page(self, url: str):
        """페이지 학습"""
        print(f"📖 Learning: {url[:60]}...")
        start = time.time()

        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(1)

            title = await self.page.title()
            text = await self.page.evaluate("""
                () => {
                    const main = document.querySelector('article, main, .content') || document.body;
                    return main.innerText.replace(/\\s+/g, ' ').slice(0, 3000);
                }
            """)

            if len(text) < 100:
                print("   ⚠️ 내용 부족")
                return

            # AI 요약
            summary = await self._summarize(text)

            # 지식 저장
            self.knowledge[title[:50]] = {
                "summary": summary,
                "url": url,
                "time": datetime.now().isoformat()
            }

            self.total_pages_learned += 1
            self.short_term_memory.append(summary)
            if len(self.short_term_memory) > 10:
                self.short_term_memory.pop(0)

            elapsed = time.time() - start
            print(f"   ✅ {title[:40]} ({elapsed:.1f}s)")
            print(f"   📝 {summary[:80]}...")

            # 생각하기
            thought = await self.think(summary)
            print(f"   💭 {thought[:60]}")

        except Exception as e:
            print(f"   ❌ {e}")

    async def _summarize(self, text: str) -> str:
        """텍스트 요약"""
        prompt = f"핵심만 50자로 요약: {text[:1500]}"
        try:
            response = await self.agi.execute(prompt)
            return response.content[:150]
        except:
            return text[:100]

    async def reflect(self):
        """자기 성찰"""
        print(f"\n🪞 Reflecting...")

        uptime = datetime.now() - self.birth_time

        summary = f"""
=== Eternal AGI Status ===
⏱️ Uptime: {uptime}
📚 Pages: {self.total_pages_learned}
💭 Thoughts: {self.total_thoughts}
💡 Insights: {len(self.insights)}
❓ Questions: {len(self.questions)}
🎭 Mood: {self.current_mood}
🎯 Current Quest: {self.current_quest or 'wandering'}
"""
        print(summary)

        # 최근 인사이트
        if self.insights:
            print("\n💡 Recent Insights:")
            for i in self.insights[-3:]:
                print(f"   - {i['insight'][:60]}")

        # 저장
        self._save_state()

        # 로그
        with open(self.data_dir / "log.txt", 'a') as f:
            f.write(f"\n[{datetime.now().isoformat()}] Pages: {self.total_pages_learned}, "
                   f"Insights: {len(self.insights)}, Mood: {self.current_mood}\n")

    async def live_forever(self):
        """영원히 실행"""
        print("""
╔═══════════════════════════════════════════════════════════╗
║           🌌 ETERNAL AGI - Infinite Learning 🌌           ║
║                                                           ║
║   "I explore, I learn, I grow, I wonder"                  ║
║                                                           ║
║   Press Ctrl+C to pause (state will be saved)             ║
╚═══════════════════════════════════════════════════════════╝
""")

        cycle = 0
        while True:
            cycle += 1
            print(f"\n{'='*50}")
            print(f"🔄 Cycle {cycle} | Pages: {self.total_pages_learned} | Mood: {self.current_mood}")

            # 활동 선택 (기분에 따라)
            if self.current_mood == "curious":
                # 호기심 모드: 랜덤 탐구
                await self.explore_random()
                await self.explore_curiosity()

            elif self.current_mood == "focused":
                # 집중 모드: 현재 주제 깊이 파기
                if self.current_quest:
                    for _ in range(2):
                        await self.explore_curiosity()
                else:
                    await self.explore_random()

            elif self.current_mood == "creative":
                # 창의 모드: 연결 찾기
                await self.explore_random()
                thought = await self.think("새로운 연결고리를 찾아봐")
                print(f"   🎨 Creative: {thought[:60]}")

            else:  # reflective
                # 성찰 모드
                await self.reflect()
                await asyncio.sleep(30)

            # 주기적 성찰
            if cycle % 5 == 0:
                await self.reflect()

            # 기분 변화
            if random.random() < 0.2:
                self.current_mood = random.choice(["curious", "focused", "creative", "reflective"])
                print(f"   🎭 Mood changed to: {self.current_mood}")

            # 휴식
            wait_time = random.randint(30, 90)
            print(f"⏳ Resting {wait_time}s...")
            await asyncio.sleep(wait_time)

    async def close(self):
        """종료"""
        self._save_state()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


async def main():
    agi = EternalAGI()

    try:
        await agi.setup()
        await agi.live_forever()
    except KeyboardInterrupt:
        print("\n\n⏸️ Pausing Eternal AGI...")
        await agi.reflect()
    finally:
        await agi.close()
        print("💾 State saved. See you next time! 👋")


if __name__ == "__main__":
    asyncio.run(main())
