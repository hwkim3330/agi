#!/usr/bin/env python3
"""
🎯 Goal-Oriented Browser Agent
목표를 세우고, 계획하고, 판단해서 행동하는 AI
"""
import asyncio
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("pip install playwright && playwright install chromium")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))


# 가능한 목표들
GOALS = [
    {
        "name": "AI 최신 뉴스 수집",
        "description": "인공지능 관련 최신 뉴스와 연구를 찾아 학습한다",
        "start_url": "https://news.ycombinator.com",
        "keywords": ["AI", "machine learning", "neural", "GPT", "LLM", "model"],
        "success_criteria": "AI 관련 기사 3개 이상 읽기"
    },
    {
        "name": "TSN 네트워킹 학습",
        "description": "Time-Sensitive Networking 관련 정보를 검색하고 학습한다",
        "start_url": "https://www.google.com",
        "keywords": ["TSN", "IEEE 802.1", "time-sensitive", "deterministic", "ethernet"],
        "success_criteria": "TSN 관련 페이지 2개 이상 읽기"
    },
    {
        "name": "Python 프로그래밍 팁",
        "description": "Python 프로그래밍 관련 유용한 정보를 찾는다",
        "start_url": "https://www.reddit.com/r/Python/hot/",
        "keywords": ["python", "tutorial", "tip", "library", "async"],
        "success_criteria": "유용한 Python 정보 2개 이상 찾기"
    },
    {
        "name": "한국 IT 뉴스",
        "description": "한국 IT/기술 뉴스를 확인한다",
        "start_url": "https://news.naver.com/section/105",
        "keywords": ["AI", "반도체", "스타트업", "테크", "개발"],
        "success_criteria": "IT 뉴스 3개 이상 읽기"
    },
    {
        "name": "위키피디아 탐험",
        "description": "흥미로운 지식을 위키피디아에서 학습한다",
        "start_url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "keywords": ["history", "applications", "research", "future"],
        "success_criteria": "관련 문서 3개 이상 읽기"
    },
]


class GoalAgent:
    """목표 지향 에이전트"""

    def __init__(self):
        self.browser = None
        self.page = None
        self.agi = None

        # 상태
        self.current_goal = None
        self.plan = []
        self.completed_steps = []
        self.pages_read = []
        self.knowledge_gained = []

        self.total_goals_completed = 0
        self.total_actions = 0

        self.data_dir = Path("/home/kim/agi/goal_agent_data")
        self.data_dir.mkdir(exist_ok=True)

    async def setup(self):
        """초기화"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=['--no-sandbox'],
            slow_mo=50
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1400, 'height': 900}
        )
        self.page = await self.context.new_page()
        print("🌐 Browser ready")

        print("🧠 Loading brain...")
        from agents.lfm2_adapter import LFM2VLAdapter, LFM2Config
        config = LFM2Config(
            model_id="LiquidAI/LFM2-VL-1.6B",
            enable_continual_learning=True
        )
        self.agi = LFM2VLAdapter(lfm2_config=config)
        await self.agi.load_model()
        print("✨ Goal Agent ready!")

    async def set_goal(self, goal: dict):
        """목표 설정"""
        self.current_goal = goal
        self.plan = []
        self.completed_steps = []
        self.pages_read = []

        print(f"\n{'='*60}")
        print(f"🎯 Goal: {goal['name']}")
        print(f"📋 Description: {goal['description']}")
        print(f"✅ Success: {goal['success_criteria']}")
        print(f"🔑 Keywords: {', '.join(goal['keywords'][:5])}")

    async def create_plan(self) -> list:
        """AI가 목표 달성 계획 수립"""
        prompt = f"""목표: {self.current_goal['name']}
설명: {self.current_goal['description']}
시작 URL: {self.current_goal['start_url']}
키워드: {', '.join(self.current_goal['keywords'])}

이 목표를 달성하기 위한 3-5단계 계획을 세워줘.
각 단계는 구체적인 행동이어야 해.

형식:
1. [행동]: [설명]
2. [행동]: [설명]
...

예:
1. 검색: Google에서 "TSN networking"으로 검색
2. 클릭: 첫 번째 결과 클릭해서 읽기
3. 학습: 핵심 내용 파악

짧게 응답:"""

        try:
            response = await self.agi.execute(prompt)
            text = response.content

            # 계획 파싱
            lines = text.strip().split('\n')
            plan = []
            for line in lines:
                if line.strip() and (line[0].isdigit() or line.startswith('-')):
                    plan.append(line.strip())

            self.plan = plan[:5] if plan else ["검색하기", "읽기", "학습하기"]

            print(f"\n📝 Plan:")
            for i, step in enumerate(self.plan):
                print(f"   {i+1}. {step}")

            return self.plan
        except Exception as e:
            print(f"❌ Plan creation failed: {e}")
            self.plan = ["검색하기", "읽기", "학습하기"]
            return self.plan

    async def analyze_page(self) -> dict:
        """현재 페이지 분석"""
        try:
            title = await self.page.title()
            url = self.page.url

            # 페이지 텍스트
            text = await self.page.evaluate("""
                () => {
                    const main = document.querySelector('article, main, .content, #content') || document.body;
                    return main.innerText.slice(0, 2000);
                }
            """)

            # 클릭 가능한 요소들
            links = await self.page.evaluate("""
                () => {
                    const items = [];
                    document.querySelectorAll('a[href]').forEach((el) => {
                        if (el.offsetParent && el.innerText.trim().length > 3) {
                            items.push({
                                text: el.innerText.slice(0, 50).trim(),
                                href: el.href
                            });
                        }
                    });
                    return items.slice(0, 15);
                }
            """)

            # 입력 필드
            inputs = await self.page.evaluate("""
                () => document.querySelectorAll('input[type="text"], input[type="search"], textarea').length > 0
            """)

            return {
                "title": title,
                "url": url,
                "text": text[:500],
                "links": links,
                "has_search": inputs
            }
        except Exception as e:
            return {"title": "Error", "url": "", "text": "", "links": [], "has_search": False}

    async def decide_next_action(self, page_info: dict) -> dict:
        """AI가 다음 행동 결정 (목표 기반)"""
        # 관련 링크 찾기
        relevant_links = []
        for link in page_info['links']:
            for kw in self.current_goal['keywords']:
                if kw.lower() in link['text'].lower():
                    relevant_links.append(link['text'])
                    break

        links_str = '\n'.join([f"- {l['text'][:40]}" for l in page_info['links'][:8]])
        relevant_str = ', '.join(relevant_links[:3]) if relevant_links else '없음'

        prompt = f"""🎯 목표: {self.current_goal['name']}
📄 현재 페이지: {page_info['title'][:40]}
🔗 URL: {page_info['url'][:50]}
📝 읽은 페이지: {len(self.pages_read)}개
🔑 관련 키워드 있는 링크: {relevant_str}

사용 가능한 링크:
{links_str}

검색창 있음: {'예' if page_info['has_search'] else '아니오'}

목표 달성을 위해 다음 행동을 선택해:
1. CLICK [링크 텍스트] - 특정 링크 클릭
2. SEARCH [검색어] - 검색창에 입력
3. SCROLL - 아래로 스크롤
4. BACK - 뒤로가기
5. READ - 현재 페이지 읽고 학습
6. DONE - 목표 달성 완료

형식: ACTION: [행동] TARGET: [대상]
예: ACTION: CLICK TARGET: AI 연구

한 줄로 응답:"""

        try:
            response = await self.agi.execute(prompt)
            text = response.content.strip().upper()

            action = "READ"
            target = ""

            if "CLICK" in text:
                action = "CLICK"
                if "TARGET:" in text:
                    target = response.content.split("TARGET:")[-1].strip()
                elif relevant_links:
                    target = relevant_links[0]
            elif "SEARCH" in text:
                action = "SEARCH"
                if "TARGET:" in text:
                    target = response.content.split("TARGET:")[-1].strip()
                else:
                    target = self.current_goal['keywords'][0]
            elif "SCROLL" in text:
                action = "SCROLL"
            elif "BACK" in text:
                action = "BACK"
            elif "DONE" in text:
                action = "DONE"
            else:
                action = "READ"

            return {"action": action, "target": target, "raw": response.content[:100]}
        except Exception as e:
            return {"action": "READ", "target": "", "raw": str(e)}

    async def execute_action(self, action: str, target: str) -> bool:
        """행동 실행"""
        self.total_actions += 1

        try:
            if action == "CLICK":
                if target:
                    # 텍스트로 링크 찾기
                    try:
                        elem = self.page.get_by_text(target, exact=False).first
                        await elem.click(timeout=5000)
                        print(f"   🖱️ Clicked: '{target[:30]}'")
                        await asyncio.sleep(2)
                        return True
                    except:
                        # 모든 링크에서 찾기
                        links = await self.page.query_selector_all('a')
                        for link in links[:20]:
                            try:
                                text = await link.inner_text()
                                if target.lower() in text.lower():
                                    await link.click(timeout=3000)
                                    print(f"   🖱️ Clicked: '{text[:30]}'")
                                    await asyncio.sleep(2)
                                    return True
                            except:
                                continue
                print(f"   ❌ Could not find: '{target[:30]}'")
                return False

            elif action == "SEARCH":
                selectors = ['input[type="search"]', 'input[name="q"]', 'textarea[name="q"]',
                           'input[type="text"]', 'textarea']
                for sel in selectors:
                    try:
                        await self.page.fill(sel, target, timeout=3000)
                        await self.page.keyboard.press("Enter")
                        print(f"   🔍 Searched: '{target[:30]}'")
                        await asyncio.sleep(2)
                        return True
                    except:
                        continue
                return False

            elif action == "SCROLL":
                await self.page.mouse.wheel(0, 400)
                print("   📜 Scrolled down")
                return True

            elif action == "BACK":
                await self.page.go_back(timeout=5000)
                print("   ⬅️ Went back")
                await asyncio.sleep(1)
                return True

            elif action == "READ":
                page_info = await self.analyze_page()
                self.pages_read.append({
                    "title": page_info['title'],
                    "url": page_info['url'],
                    "text": page_info['text'][:200]
                })
                print(f"   📖 Read: {page_info['title'][:40]}")

                # 학습 내용 요약
                if page_info['text']:
                    summary = await self._summarize(page_info['text'])
                    self.knowledge_gained.append(summary)
                    print(f"   💡 Learned: {summary[:60]}...")
                return True

            elif action == "DONE":
                return True

        except Exception as e:
            print(f"   ❌ Action failed: {e}")
            return False

        return False

    async def _summarize(self, text: str) -> str:
        """텍스트 요약"""
        try:
            response = await self.agi.execute(f"핵심만 30자로 요약: {text[:800]}")
            return response.content[:100]
        except:
            return text[:50]

    async def pursue_goal(self, max_steps: int = 15):
        """목표 추구"""
        print(f"\n🚀 Starting goal pursuit...")

        # 시작 URL로 이동
        try:
            await self.page.goto(self.current_goal['start_url'],
                               wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            print(f"❌ Navigation failed: {e}")
            return False

        await asyncio.sleep(2)

        for step in range(max_steps):
            print(f"\n--- Step {step+1}/{max_steps} ---")

            try:
                # 페이지 분석
                page_info = await self.analyze_page()
                print(f"📄 Page: {page_info['title'][:40]}")

                # 다음 행동 결정
                decision = await self.decide_next_action(page_info)
                action = decision['action']
                target = decision['target']

                print(f"🎯 Decision: {action} | {target[:30] if target else 'N/A'}")

                # 행동 실행
                success = await self.execute_action(action, target)

                if action == "DONE":
                    print(f"\n✅ Goal completed!")
                    break

                # 목표 달성 체크
                if len(self.pages_read) >= 3:
                    print(f"\n✅ Read enough pages ({len(self.pages_read)}), goal likely completed!")
                    break

                await asyncio.sleep(1)

            except Exception as e:
                print(f"   ❌ Step error: {e}")
                try:
                    self.page = await self.context.new_page()
                    await self.page.goto(self.current_goal['start_url'], timeout=30000)
                except:
                    pass
                continue

        return True

    async def reflect(self):
        """목표 달성 후 성찰"""
        print(f"\n{'='*60}")
        print(f"📊 Goal Review: {self.current_goal['name']}")
        print(f"   Pages read: {len(self.pages_read)}")
        print(f"   Actions taken: {self.total_actions}")

        if self.knowledge_gained:
            print(f"\n💡 Knowledge gained:")
            for k in self.knowledge_gained[-5:]:
                print(f"   - {k[:60]}")

        # 저장
        self._save_session()

    def _save_session(self):
        """세션 저장"""
        session = {
            "goal": self.current_goal['name'],
            "pages_read": self.pages_read,
            "knowledge": self.knowledge_gained,
            "actions": self.total_actions,
            "timestamp": datetime.now().isoformat()
        }

        session_file = self.data_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session, f, ensure_ascii=False, indent=2)

    async def run_forever(self):
        """영원히 실행 - 목표를 계속 추구"""
        print("""
╔═══════════════════════════════════════════════════════════╗
║        🎯 Goal-Oriented Agent - Purposeful AI 🎯          ║
║                                                           ║
║   Setting goals, making plans, taking action              ║
║   Press Ctrl+C to stop                                    ║
╚═══════════════════════════════════════════════════════════╝
""")

        while True:
            # 목표 선택
            goal = random.choice(GOALS)
            await self.set_goal(goal)

            # 계획 수립
            await self.create_plan()

            # 목표 추구
            await self.pursue_goal(max_steps=12)

            # 성찰
            await self.reflect()

            self.total_goals_completed += 1

            # 휴식
            wait = random.randint(10, 30)
            print(f"\n⏳ Resting {wait}s before next goal...")
            print(f"📈 Goals completed: {self.total_goals_completed}")
            await asyncio.sleep(wait)

    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


async def main():
    agent = GoalAgent()

    try:
        await agent.setup()
        await agent.run_forever()
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Stopped.")
        print(f"📈 Total goals completed: {agent.total_goals_completed}")
        print(f"📚 Total pages read: {len(agent.pages_read)}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
