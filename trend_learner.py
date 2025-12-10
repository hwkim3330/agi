#!/usr/bin/env python3
"""
AGI Trinity - Trend Learner
실시간 트렌드/뉴스 피드 학습기
- Reddit, Hacker News, Google Trends
- 네이버 실시간 검색어
- 항상 새로운 콘텐츠 학습
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
    print("playwright not installed. Run: pip install playwright && playwright install chromium")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))


# 실시간 피드 소스들
FEED_SOURCES = {
    "reddit_ml": {
        "url": "https://www.reddit.com/r/MachineLearning/new/",
        "selector": "a[data-click-id='body']",
        "name": "Reddit ML"
    },
    "reddit_tech": {
        "url": "https://www.reddit.com/r/technology/new/",
        "selector": "a[data-click-id='body']",
        "name": "Reddit Tech"
    },
    "hackernews": {
        "url": "https://news.ycombinator.com/newest",
        "selector": "a.titleline > a",
        "name": "Hacker News"
    },
    "arxiv_ai": {
        "url": "https://arxiv.org/list/cs.AI/recent",
        "selector": "a[title='Abstract']",
        "name": "arXiv AI"
    },
    "naver_news": {
        "url": "https://news.naver.com/section/105",  # IT/과학
        "selector": "a.sa_text_title",
        "name": "네이버 IT뉴스"
    },
    "google_trends": {
        "url": "https://trends.google.com/trending?geo=KR",
        "selector": "a[href*='/trending']",
        "name": "Google Trends KR"
    }
}


class TrendLearner:
    """실시간 트렌드 학습기"""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.browser = None
        self.page = None
        self.agi = None
        self.learning_history = []
        self.seen_urls = set()  # 중복 방지

    async def setup_browser(self):
        """브라우저 초기화"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 900},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0.0.0"
        )
        self.page = await self.context.new_page()
        print("🌐 Browser ready")

    async def setup_model(self):
        """모델 로드"""
        print("🧠 Loading model...")
        start = time.time()

        from agents.lfm2_adapter import LFM2VLAdapter, LFM2Config

        config = LFM2Config(
            model_id="LiquidAI/LFM2-VL-1.6B",
            enable_continual_learning=True
        )
        self.agi = LFM2VLAdapter(lfm2_config=config)
        await self.agi.load_model()

        print(f"✅ Model loaded in {time.time() - start:.1f}s")

    async def analyze_text(self, text: str, max_length: int = 2000) -> str:
        """텍스트 분석"""
        prompt = f"다음 내용을 100자 이내로 핵심만 요약해주세요: {text[:max_length]}"
        try:
            response = await self.agi.execute(prompt)
            return response.content[:300]
        except Exception as e:
            return f"분석 실패: {e}"

    async def fetch_feed_links(self, source_key: str) -> list:
        """피드에서 새 링크 가져오기"""
        source = FEED_SOURCES.get(source_key)
        if not source:
            return []

        print(f"\n📡 Fetching: {source['name']}")

        try:
            await self.page.goto(source['url'], wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)

            # 링크 추출
            links = await self.page.evaluate(f"""
                () => {{
                    const elements = document.querySelectorAll("{source['selector']}");
                    return Array.from(elements).slice(0, 10).map(a => a.href).filter(h => h && h.startsWith('http'));
                }}
            """)

            # 새 링크만 필터링
            new_links = [l for l in links if l not in self.seen_urls][:3]
            print(f"   Found {len(new_links)} new links")

            return new_links

        except Exception as e:
            print(f"   ❌ Error fetching {source['name']}: {e}")
            return []

    async def learn_from_url(self, url: str, source_name: str) -> dict:
        """URL에서 학습"""
        if url in self.seen_urls:
            return {"url": url, "skipped": True}

        self.seen_urls.add(url)
        start = time.time()
        print(f"📖 Learning: {url[:60]}...")

        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(1)

            title = await self.page.title()
            text = await self.page.evaluate("""
                () => {
                    const main = document.querySelector('article, main, .content, #content, .post-content') || document.body;
                    return main.innerText.replace(/\\s+/g, ' ').slice(0, 5000);
                }
            """)

            summary = await self.analyze_text(text) if len(text) > 100 else "내용 부족"

            result = {
                "url": url,
                "title": title,
                "source": source_name,
                "text_length": len(text),
                "summary": summary,
                "time": time.time() - start,
                "timestamp": datetime.now().isoformat()
            }

            self.learning_history.append(result)
            print(f"   ✅ {title[:45]} ({result['time']:.1f}s)")
            print(f"   📝 {summary[:80]}...")

            return result

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {"url": url, "error": str(e)}

    async def learn_from_feed(self, source_key: str):
        """특정 피드에서 학습"""
        source = FEED_SOURCES.get(source_key, {})
        links = await self.fetch_feed_links(source_key)

        for link in links:
            await self.learn_from_url(link, source.get('name', source_key))
            await asyncio.sleep(0.5)

    async def continuous_learn(self, sources: list = None, interval_minutes: int = 2):
        """지속 학습 - 라운드 로빈으로 소스 순환"""
        if sources is None:
            sources = list(FEED_SOURCES.keys())

        print(f"\n🔄 Trend Learning Started")
        print(f"   Sources: {', '.join(sources)}")
        print(f"   Interval: {interval_minutes} min")

        cycle = 1
        source_idx = 0

        while True:
            print(f"\n{'='*50}")
            print(f"📚 Cycle {cycle}")

            # 라운드 로빈으로 소스 2개씩 처리
            for _ in range(2):
                source = sources[source_idx % len(sources)]
                await self.learn_from_feed(source)
                source_idx += 1

            # 통계
            print(f"\n📊 Stats: {len(self.learning_history)} pages learned, {len(self.seen_urls)} unique URLs")

            # 저장
            self.save_history()

            print(f"⏳ Waiting {interval_minutes} min...")
            await asyncio.sleep(interval_minutes * 60)
            cycle += 1

    def save_history(self, path: str = "/home/kim/agi/trend_history.json"):
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
    parser = argparse.ArgumentParser(description="Trend Learner - 실시간 피드 학습")
    parser.add_argument("--sources", nargs="+",
                       default=["hackernews", "reddit_ml", "naver_news"],
                       help="Sources: " + ", ".join(FEED_SOURCES.keys()))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--interval", type=int, default=2, help="Interval in minutes")
    args = parser.parse_args()

    learner = TrendLearner(headless=args.headless)

    try:
        await learner.setup_browser()
        await learner.setup_model()
        await learner.continuous_learn(args.sources, args.interval)
    except KeyboardInterrupt:
        print("\n⏹️ Stopping...")
    finally:
        learner.save_history()
        await learner.close()


if __name__ == "__main__":
    asyncio.run(main())
