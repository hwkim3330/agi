#!/usr/bin/env python3
"""
🤖 Browser Agent - 진짜 브라우저 조작 AI
마우스 클릭, 키보드 입력, 스크롤 등 실제 인터랙션
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


class BrowserAgent:
    """브라우저를 직접 조작하는 AI 에이전트"""

    def __init__(self):
        self.browser = None
        self.page = None
        self.agi = None
        self.action_count = 0
        self.data_dir = Path("/home/kim/agi/agent_data")
        self.data_dir.mkdir(exist_ok=True)

    async def setup(self):
        """초기화"""
        # 브라우저 (보이게)
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=['--no-sandbox', '--start-maximized'],
            slow_mo=100  # 동작 보이게 약간 느리게
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1400, 'height': 900}
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
        print("✨ Browser Agent ready!")

    # ========== 브라우저 액션들 ==========

    async def click_element(self, selector: str = None, text: str = None):
        """요소 클릭"""
        try:
            if text:
                # 텍스트로 찾아서 클릭
                elem = self.page.get_by_text(text, exact=False).first
                await elem.click(timeout=5000)
                print(f"   🖱️ Clicked: '{text[:30]}'")
            elif selector:
                await self.page.click(selector, timeout=5000)
                print(f"   🖱️ Clicked: {selector}")
            self.action_count += 1
            return True
        except Exception as e:
            print(f"   ❌ Click failed: {e}")
            return False

    async def click_random_link(self):
        """랜덤 링크 클릭"""
        try:
            links = await self.page.query_selector_all('a[href]')
            visible_links = []
            for link in links[:30]:
                if await link.is_visible():
                    text = await link.inner_text()
                    if len(text.strip()) > 5:
                        visible_links.append(link)

            if visible_links:
                link = random.choice(visible_links[:10])
                text = await link.inner_text()
                await link.click(timeout=5000)
                print(f"   🔗 Clicked link: '{text[:40]}'")
                self.action_count += 1
                return True
        except Exception as e:
            print(f"   ❌ Random click failed: {e}")
        return False

    async def type_text(self, selector: str, text: str):
        """텍스트 입력"""
        try:
            await self.page.click(selector, timeout=5000)
            await self.page.fill(selector, text)
            print(f"   ⌨️ Typed: '{text[:30]}'")
            self.action_count += 1
            return True
        except Exception as e:
            print(f"   ❌ Type failed: {e}")
            return False

    async def press_key(self, key: str):
        """키 누르기"""
        try:
            await self.page.keyboard.press(key)
            print(f"   ⌨️ Pressed: {key}")
            self.action_count += 1
            return True
        except Exception as e:
            print(f"   ❌ Key failed: {e}")
            return False

    async def scroll(self, direction: str = "down", amount: int = 300):
        """스크롤"""
        try:
            if direction == "down":
                await self.page.mouse.wheel(0, amount)
            else:
                await self.page.mouse.wheel(0, -amount)
            print(f"   📜 Scrolled {direction}")
            self.action_count += 1
            return True
        except Exception as e:
            print(f"   ❌ Scroll failed: {e}")
            return False

    async def go_back(self):
        """뒤로가기"""
        try:
            await self.page.go_back(timeout=10000)
            print("   ⬅️ Went back")
            return True
        except:
            return False

    async def take_screenshot(self) -> str:
        """스크린샷"""
        path = self.data_dir / f"screen_{datetime.now().strftime('%H%M%S')}.png"
        await self.page.screenshot(path=str(path))
        return str(path)

    # ========== AI 기반 행동 ==========

    async def analyze_and_decide(self) -> dict:
        """현재 페이지 분석하고 다음 행동 결정"""
        # 페이지 정보 수집
        title = await self.page.title()
        url = self.page.url

        # 클릭 가능한 요소들
        clickables = await self.page.evaluate("""
            () => {
                const items = [];
                document.querySelectorAll('a, button, input[type="submit"]').forEach((el, i) => {
                    if (el.offsetParent && i < 20) {
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            items.push({
                                tag: el.tagName,
                                text: el.innerText?.slice(0, 50) || el.value || '',
                                href: el.href || ''
                            });
                        }
                    }
                });
                return items.slice(0, 10);
            }
        """)

        # 입력 필드
        inputs = await self.page.evaluate("""
            () => {
                const items = [];
                document.querySelectorAll('input[type="text"], input[type="search"], textarea').forEach((el) => {
                    if (el.offsetParent) {
                        items.push({
                            type: el.type || 'text',
                            placeholder: el.placeholder || '',
                            name: el.name || ''
                        });
                    }
                });
                return items.slice(0, 5);
            }
        """)

        return {
            "title": title,
            "url": url,
            "clickables": clickables,
            "inputs": inputs
        }

    async def decide_action(self, page_info: dict) -> dict:
        """AI가 다음 행동 결정"""
        clickables_str = "\n".join([f"- {c['text'][:30]}" for c in page_info['clickables'][:5]])
        inputs_str = "\n".join([f"- {i['placeholder'] or i['name']}" for i in page_info['inputs'][:3]])

        prompt = f"""현재 웹페이지: {page_info['title'][:50]}
URL: {page_info['url'][:50]}

클릭 가능:
{clickables_str or '없음'}

입력 필드:
{inputs_str or '없음'}

다음 행동을 선택해:
1. CLICK: 링크/버튼 클릭 (텍스트 지정)
2. TYPE: 검색어 입력 (검색창이 있으면)
3. SCROLL: 아래로 스크롤
4. BACK: 뒤로가기
5. RANDOM: 랜덤 링크 클릭

형식: ACTION: [행동] | TARGET: [대상/텍스트]
예: ACTION: CLICK | TARGET: 뉴스
예: ACTION: TYPE | TARGET: artificial intelligence

짧게 한 줄로 응답:"""

        try:
            response = await self.agi.execute(prompt)
            text = response.content.strip()

            # 파싱
            action = "RANDOM"
            target = ""

            if "ACTION:" in text.upper():
                parts = text.upper().split("|")
                for p in parts:
                    if "ACTION:" in p:
                        action = p.split("ACTION:")[-1].strip().split()[0]
                    if "TARGET:" in p:
                        target = text.split("TARGET:")[-1].strip() if "TARGET:" in text else ""

            return {"action": action, "target": target, "raw": text}

        except Exception as e:
            return {"action": "RANDOM", "target": "", "raw": str(e)}

    async def execute_action(self, decision: dict):
        """행동 실행"""
        action = decision['action'].upper()
        target = decision['target']

        print(f"   🎯 Decision: {action} | {target[:30] if target else 'N/A'}")

        if action == "CLICK" and target:
            await self.click_element(text=target)
        elif action == "TYPE" and target:
            # 검색창 찾기
            selectors = ['input[type="search"]', 'input[name="q"]', 'input[type="text"]', 'textarea']
            for sel in selectors:
                if await self.type_text(sel, target):
                    await self.press_key("Enter")
                    break
        elif action == "SCROLL":
            await self.scroll("down", random.randint(200, 500))
        elif action == "BACK":
            await self.go_back()
        else:  # RANDOM
            await self.click_random_link()

        await asyncio.sleep(1)

    # ========== 메인 루프 ==========

    async def explore_site(self, start_url: str, max_actions: int = 10):
        """사이트 탐험"""
        print(f"\n🌐 Exploring: {start_url}")
        try:
            await self.page.goto(start_url, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            print(f"   ❌ Navigation failed: {e}")
            # 페이지 재생성
            try:
                self.page = await self.context.new_page()
                await self.page.goto(start_url, wait_until="domcontentloaded", timeout=30000)
            except:
                return
        await asyncio.sleep(2)

        for i in range(max_actions):
            try:
                print(f"\n--- Action {i+1}/{max_actions} ---")

                # 분석
                page_info = await self.analyze_and_decide()
                print(f"📄 Page: {page_info['title'][:40]}")

                # 결정
                decision = await self.decide_action(page_info)

                # 실행
                await self.execute_action(decision)

                await asyncio.sleep(random.uniform(1, 3))
            except Exception as e:
                print(f"   ❌ Action error: {e}")
                # 페이지 복구 시도
                try:
                    self.page = await self.context.new_page()
                except:
                    pass
                break

    async def random_surf(self):
        """랜덤 서핑"""
        start_sites = [
            "https://news.ycombinator.com",
            "https://www.reddit.com/r/technology",
            "https://news.naver.com",
            "https://en.wikipedia.org/wiki/Special:Random",
            "https://www.google.com",
            "https://arxiv.org/list/cs.AI/recent",
        ]

        while True:
            site = random.choice(start_sites)
            actions = random.randint(5, 15)

            print(f"\n{'='*50}")
            print(f"🎲 Starting from: {site}")
            print(f"   Actions planned: {actions}")

            try:
                await self.explore_site(site, max_actions=actions)
            except Exception as e:
                print(f"❌ Error: {e}")

            print(f"\n📊 Total actions: {self.action_count}")
            wait = random.randint(10, 30)
            print(f"⏳ Waiting {wait}s...")
            await asyncio.sleep(wait)

    async def search_and_browse(self, query: str):
        """검색하고 결과 브라우징"""
        print(f"\n🔍 Searching: {query}")

        # Google 검색
        await self.page.goto("https://www.google.com", timeout=30000)
        await asyncio.sleep(1)

        # 검색창에 입력
        await self.type_text('textarea[name="q"]', query)
        await self.press_key("Enter")
        await asyncio.sleep(2)

        # 결과에서 랜덤 클릭
        for _ in range(3):
            await self.scroll("down", 200)
            await asyncio.sleep(1)
            if await self.click_random_link():
                await asyncio.sleep(3)
                # 페이지 탐험
                await self.explore_site(self.page.url, max_actions=5)
                await self.go_back()

    async def run_forever(self):
        """영원히 실행"""
        print("""
╔═══════════════════════════════════════════════════════════╗
║         🤖 Browser Agent - AI Web Explorer 🤖             ║
║                                                           ║
║   Clicking, typing, scrolling, exploring...               ║
║   Press Ctrl+C to stop                                    ║
╚═══════════════════════════════════════════════════════════╝
""")

        search_topics = [
            "latest AI news",
            "machine learning tutorial",
            "TSN time sensitive networking",
            "LiquidAI LFM2",
            "python programming",
            "robotics research",
            "quantum computing",
            "neural network",
        ]

        cycle = 0
        while True:
            cycle += 1
            print(f"\n{'='*60}")
            print(f"🔄 Cycle {cycle} | Total Actions: {self.action_count}")

            # 50% 확률로 검색, 50% 랜덤 서핑
            if random.random() < 0.5:
                query = random.choice(search_topics)
                await self.search_and_browse(query)
            else:
                site = random.choice([
                    "https://news.ycombinator.com",
                    "https://www.reddit.com/r/MachineLearning/new/",
                    "https://news.naver.com/section/105",
                    "https://en.wikipedia.org/wiki/Special:Random",
                ])
                await self.explore_site(site, max_actions=random.randint(5, 10))

            await asyncio.sleep(random.randint(5, 15))

    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


async def main():
    agent = BrowserAgent()

    try:
        await agent.setup()
        await agent.run_forever()
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Stopped. Total actions: {agent.action_count}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
