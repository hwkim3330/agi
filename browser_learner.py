#!/usr/bin/env python3
"""
AGI Trinity - Browser-based Learning
AI가 브라우저를 직접 제어하여 웹에서 학습하는 모듈
"""
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("playwright not installed. Run: pip install playwright && playwright install chromium")

# AGI 모듈 임포트
sys.path.insert(0, str(Path(__file__).parent))
from agents.lfm2_adapter import LFM2VLAdapter, LFM2Config


class BrowserLearner:
    """브라우저 기반 AI 학습기"""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.browser = None
        self.context = None
        self.page = None
        self.agi = None
        self.learning_history = []

    async def setup(self):
        """브라우저와 AGI 초기화"""
        # Playwright 설정
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=['--start-maximized']
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080}
        )
        self.page = await self.context.new_page()

        # AGI 초기화
        config = LFM2Config(
            model_id="LiquidAI/LFM2-VL-1.6B",
            enable_continual_learning=True
        )
        self.agi = LFM2VLAdapter(lfm2_config=config)
        await self.agi.load_model()
        print("🧠 AGI and Browser initialized")

    async def screenshot_and_analyze(self) -> str:
        """현재 페이지 스크린샷 찍고 AI로 분석"""
        # 스크린샷 저장
        screenshot_path = f"/tmp/agi_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        await self.page.screenshot(path=screenshot_path)

        # AGI로 분석
        response = await self.agi.execute(
            "이 웹페이지의 내용을 요약해주세요. 주요 정보와 학습할 만한 내용을 추출해주세요.",
            images=[screenshot_path]
        )

        return response.content

    async def navigate_and_learn(self, url: str) -> dict:
        """URL로 이동하고 내용 학습"""
        print(f"🌐 Navigating to: {url}")
        await self.page.goto(url, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)  # 페이지 로딩 대기

        # 페이지 제목
        title = await self.page.title()

        # 페이지 텍스트 추출
        text_content = await self.page.evaluate("""
            () => {
                const article = document.querySelector('article') || document.querySelector('main') || document.body;
                return article.innerText.slice(0, 10000);
            }
        """)

        # AI로 분석
        analysis = await self.screenshot_and_analyze()

        result = {
            "url": url,
            "title": title,
            "content_length": len(text_content),
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }

        self.learning_history.append(result)
        print(f"📚 Learned from: {title}")
        print(f"   Analysis: {analysis[:200]}...")

        return result

    async def search_and_learn(self, query: str, num_results: int = 3):
        """검색하고 결과에서 학습"""
        # Google 검색
        search_url = f"https://www.google.com/search?q={query}"
        print(f"🔍 Searching: {query}")
        await self.page.goto(search_url, wait_until="networkidle")
        await asyncio.sleep(2)

        # 검색 결과 링크 추출
        links = await self.page.evaluate("""
            () => {
                const results = document.querySelectorAll('div.g a[href^="http"]');
                return Array.from(results).slice(0, 10).map(a => a.href);
            }
        """)

        # 유효한 링크만 필터링
        valid_links = [l for l in links if 'google.com' not in l][:num_results]

        print(f"   Found {len(valid_links)} results")

        # 각 결과에서 학습
        for link in valid_links:
            try:
                await self.navigate_and_learn(link)
                await asyncio.sleep(1)
            except Exception as e:
                print(f"   Error learning from {link}: {e}")

    async def continuous_learn(self, topics: list, interval_minutes: int = 5):
        """지속적으로 학습"""
        print(f"🔄 Starting continuous learning")
        print(f"   Topics: {topics}")
        print(f"   Interval: {interval_minutes} minutes")

        cycle = 1
        while True:
            print(f"\n📖 Learning Cycle {cycle}")
            for topic in topics:
                await self.search_and_learn(topic, num_results=2)

            # 학습 통계
            print(f"\n📊 Statistics:")
            print(f"   Total pages learned: {len(self.learning_history)}")

            # 대기
            print(f"⏳ Waiting {interval_minutes} minutes...")
            await asyncio.sleep(interval_minutes * 60)
            cycle += 1

    async def close(self):
        """정리"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    def save_history(self, path: str = "/home/kim/agi/learning_history.json"):
        """학습 기록 저장"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.learning_history, f, ensure_ascii=False, indent=2)
        print(f"💾 Learning history saved to {path}")


async def main():
    """메인 실행"""
    import argparse
    parser = argparse.ArgumentParser(description="AI Browser Learner")
    parser.add_argument("--topics", nargs="+", default=["TSN networking", "artificial intelligence"])
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--interval", type=int, default=5, help="Learning interval in minutes")
    args = parser.parse_args()

    learner = BrowserLearner(headless=args.headless)

    try:
        await learner.setup()
        await learner.continuous_learn(args.topics, args.interval)
    except KeyboardInterrupt:
        print("\n⏹️ Stopping...")
    finally:
        learner.save_history()
        await learner.close()


if __name__ == "__main__":
    asyncio.run(main())
