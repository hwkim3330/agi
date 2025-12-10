#!/usr/bin/env python3
"""
AGI Trinity - Fast Browser Learning
최적화된 브라우저 기반 AI 학습기
- 모델 한번 로드 후 상주
- 간단한 텍스트 기반 학습 (스크린샷 없이)
- 빠른 페이지 처리
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("playwright not installed. Run: pip install playwright && playwright install chromium")
    sys.exit(1)

# AGI 모듈
sys.path.insert(0, str(Path(__file__).parent))


class FastLearner:
    """빠른 브라우저 학습기"""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.browser = None
        self.page = None
        self.model = None
        self.processor = None
        self.learning_history = []
        self.start_time = None

    async def setup_browser(self):
        """브라우저만 초기화"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0.0.0"
        )
        self.page = await self.context.new_page()
        print("🌐 Browser ready")

    async def setup_model(self):
        """모델 로드 (한번만) - LFM2-VL 어댑터 사용"""
        print("🧠 Loading model...")
        self.start_time = time.time()

        from agents.lfm2_adapter import LFM2VLAdapter, LFM2Config

        config = LFM2Config(
            model_id="LiquidAI/LFM2-VL-1.6B",
            enable_continual_learning=True
        )
        self.agi = LFM2VLAdapter(lfm2_config=config)
        await self.agi.load_model()

        load_time = time.time() - self.start_time
        print(f"✅ Model loaded in {load_time:.1f}s")

    async def analyze_text(self, text: str, max_length: int = 2000) -> str:
        """텍스트 분석 (AGI 어댑터 사용)"""
        prompt = f"다음 내용을 100자 이내로 핵심만 요약해주세요: {text[:max_length]}"

        try:
            response = await self.agi.execute(prompt)
            return response.content[:300]
        except Exception as e:
            return f"분석 실패: {e}"

    async def learn_from_url(self, url: str) -> dict:
        """URL에서 학습"""
        start = time.time()
        print(f"📖 Learning: {url}")

        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(1)

            # 페이지 정보 추출
            title = await self.page.title()
            text = await self.page.evaluate("""
                () => {
                    const main = document.querySelector('article, main, .content, #content') || document.body;
                    return main.innerText.replace(/\\s+/g, ' ').slice(0, 5000);
                }
            """)

            # AI 분석
            summary = await self.analyze_text(text) if len(text) > 100 else "내용 부족"

            result = {
                "url": url,
                "title": title,
                "text_length": len(text),
                "summary": summary,
                "time": time.time() - start,
                "timestamp": datetime.now().isoformat()
            }

            self.learning_history.append(result)
            print(f"   ✅ {title[:50]} ({result['time']:.1f}s)")
            print(f"   📝 {summary[:100]}...")

            return result

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {"url": url, "error": str(e)}

    async def search_and_learn(self, query: str, num_results: int = 3):
        """검색 후 학습"""
        print(f"\n🔍 Searching: {query}")

        # DuckDuckGo 검색 (Google보다 봇 친화적)
        search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}"
        await self.page.goto(search_url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(2)

        # 결과 링크 추출
        links = await self.page.evaluate("""
            () => {
                const results = document.querySelectorAll('a[data-testid="result-title-a"]');
                return Array.from(results).slice(0, 10).map(a => a.href);
            }
        """)

        if not links:
            # 대체 선택자
            links = await self.page.evaluate("""
                () => {
                    const results = document.querySelectorAll('.result__a, .result__url');
                    return Array.from(results).slice(0, 10).map(a => a.href);
                }
            """)

        valid_links = [l for l in links if l and l.startswith('http')][:num_results]
        print(f"   Found {len(valid_links)} results")

        for link in valid_links:
            await self.learn_from_url(link)
            await asyncio.sleep(0.5)

    async def continuous_learn(self, topics: list, interval_minutes: int = 3):
        """지속 학습"""
        print(f"\n🔄 Continuous Learning Started")
        print(f"   Topics: {', '.join(topics)}")
        print(f"   Interval: {interval_minutes} min")

        cycle = 1
        while True:
            print(f"\n{'='*50}")
            print(f"📚 Cycle {cycle}")

            for topic in topics:
                await self.search_and_learn(topic, num_results=2)

            # 통계
            print(f"\n📊 Stats: {len(self.learning_history)} pages learned")

            # 저장
            self.save_history()

            print(f"⏳ Waiting {interval_minutes} min...")
            await asyncio.sleep(interval_minutes * 60)
            cycle += 1

    def save_history(self, path: str = "/home/kim/agi/learning_history.json"):
        """학습 기록 저장"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.learning_history, f, ensure_ascii=False, indent=2)

    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", nargs="+", default=["TSN networking", "LiquidAI"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--interval", type=int, default=3)
    args = parser.parse_args()

    learner = FastLearner(headless=args.headless)

    try:
        await learner.setup_browser()
        await learner.setup_model()
        await learner.continuous_learn(args.topics, args.interval)
    except KeyboardInterrupt:
        print("\n⏹️ Stopping...")
    finally:
        learner.save_history()
        await learner.close()


if __name__ == "__main__":
    asyncio.run(main())
