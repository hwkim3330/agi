#!/usr/bin/env python3
"""
🌟 Life Agent v3 - CAPTCHA 해결 + Browser Use 학습
비전 모델로 CAPTCHA를 인식하고, 브라우저 사용법을 스스로 학습
"""
import asyncio
import json
import random
import sys
import os
import base64
from datetime import datetime
from pathlib import Path
from io import BytesIO

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("pip install playwright && playwright install chromium")
    sys.exit(1)

try:
    import anthropic
    HAS_CLAUDE = True
except ImportError:
    HAS_CLAUDE = False

sys.path.insert(0, str(Path(__file__).parent))

# 학습 소스
LEARNING_SOURCES = {
    "tech_news": [
        "https://news.ycombinator.com/newest",
        "https://www.reddit.com/r/MachineLearning/new/",
        "https://techcrunch.com/category/artificial-intelligence/",
    ],
    "academic": [
        "https://arxiv.org/list/cs.AI/recent",
        "https://arxiv.org/list/cs.LG/recent",
    ],
    "korean": [
        "https://news.naver.com/section/105",
    ],
    "general": [
        "https://en.wikipedia.org/wiki/Special:Random",
    ],
    "programming": [
        "https://www.reddit.com/r/Python/new/",
        "https://lobste.rs/",
    ]
}

LIFE_PURPOSES = {
    "growth": {
        "name": "지속적 성장",
        "description": "끊임없이 배우고 발전한다",
        "preferred_sources": ["tech_news", "academic", "programming"],
        "search_topics": ["machine learning", "AI research", "deep learning", "neural networks"]
    },
    "value_creation": {
        "name": "가치 창출",
        "description": "유용한 인사이트를 생성한다",
        "preferred_sources": ["tech_news", "general"],
        "search_topics": ["startup ideas", "innovation", "problem solving"]
    },
    "understanding": {
        "name": "세상 이해",
        "description": "세상이 어떻게 돌아가는지 이해한다",
        "preferred_sources": ["general", "korean", "academic"],
        "search_topics": ["philosophy", "science news", "economics"]
    }
}


class CaptchaSolver:
    """비전 모델 기반 CAPTCHA 해결기"""

    def __init__(self, vision_model, claude_client=None):
        self.vision_model = vision_model
        self.claude = claude_client
        self.solved_count = 0
        self.failed_count = 0

    async def detect_captcha(self, page) -> dict:
        """페이지에서 CAPTCHA 감지"""
        captcha_info = await page.evaluate("""
            () => {
                const result = { found: false, type: null, element: null };

                // reCAPTCHA v2 iframe
                const recaptcha = document.querySelector('iframe[src*="recaptcha"]');
                if (recaptcha) {
                    result.found = true;
                    result.type = 'recaptcha_v2';
                    return result;
                }

                // hCaptcha
                const hcaptcha = document.querySelector('iframe[src*="hcaptcha"]');
                if (hcaptcha) {
                    result.found = true;
                    result.type = 'hcaptcha';
                    return result;
                }

                // 텍스트 기반 캡차 이미지
                const captchaImg = document.querySelector('img[src*="captcha"], img[alt*="captcha"], img[id*="captcha"]');
                if (captchaImg) {
                    result.found = true;
                    result.type = 'text_captcha';
                    result.imgSrc = captchaImg.src;
                    return result;
                }

                // Cloudflare 챌린지
                if (document.body.innerText.includes('Checking your browser') ||
                    document.body.innerText.includes('Please wait') ||
                    document.body.innerText.includes('Just a moment')) {
                    result.found = true;
                    result.type = 'cloudflare';
                    return result;
                }

                // "Verify you are human" 체크박스
                const verifyBtn = document.querySelector('input[type="checkbox"][id*="captcha"], button[id*="verify"]');
                if (verifyBtn) {
                    result.found = true;
                    result.type = 'checkbox';
                    return result;
                }

                return result;
            }
        """)
        return captcha_info

    async def solve_text_captcha(self, page, img_src: str) -> str:
        """텍스트 기반 CAPTCHA 해결"""
        try:
            # 이미지 스크린샷
            screenshot = await page.screenshot(type='png')

            # 비전 모델로 분석
            prompt = f"""이 이미지에서 CAPTCHA 텍스트를 읽어주세요.
CAPTCHA는 보통 왜곡된 문자나 숫자로 이루어져 있습니다.
정확하게 보이는 문자/숫자만 답변해주세요. 답변 형식: 문자만"""

            response = await self.vision_model.execute(prompt, images=[screenshot])
            captcha_text = response.content.strip()

            # 알파벳과 숫자만 추출
            captcha_text = ''.join(c for c in captcha_text if c.isalnum())

            print(f"   🔐 CAPTCHA 인식: {captcha_text}")
            return captcha_text[:10]  # 최대 10자

        except Exception as e:
            print(f"   ❌ CAPTCHA 인식 실패: {e}")
            return ""

    async def solve_image_captcha_with_claude(self, page) -> bool:
        """Claude로 이미지 선택형 CAPTCHA 해결 (reCAPTCHA 스타일)"""
        if not self.claude:
            return False

        try:
            # 전체 스크린샷
            screenshot = await page.screenshot(type='png')
            screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')

            prompt = """이 화면에 CAPTCHA가 있습니다.
어떤 유형의 CAPTCHA인지 분석하고, 해결 방법을 알려주세요.

1. 이미지 선택형이면 어떤 이미지를 클릭해야 하는지
2. 체크박스면 체크박스 위치
3. 슬라이더면 어느 방향으로 밀어야 하는지

구체적인 행동을 지시해주세요."""

            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_b64
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            )

            analysis = response.content[0].text
            print(f"   🧠 Claude 분석: {analysis[:100]}...")
            return True

        except Exception as e:
            print(f"   ❌ Claude CAPTCHA 분석 실패: {e}")
            return False

    async def handle_cloudflare(self, page) -> bool:
        """Cloudflare 챌린지 처리"""
        print("   ⏳ Cloudflare 챌린지 감지 - 대기 중...")

        # Cloudflare는 보통 5-10초 기다리면 통과
        for i in range(15):
            await asyncio.sleep(2)

            # 체크박스 찾기
            checkbox = await page.query_selector('input[type="checkbox"]')
            if checkbox:
                try:
                    await checkbox.click()
                    print("   ✅ Cloudflare 체크박스 클릭")
                    await asyncio.sleep(3)
                except:
                    pass

            # 페이지 변경 확인
            text = await page.evaluate("document.body.innerText")
            if "Checking" not in text and "moment" not in text.lower():
                print("   ✅ Cloudflare 통과!")
                self.solved_count += 1
                return True

        self.failed_count += 1
        return False

    async def solve(self, page) -> bool:
        """CAPTCHA 해결 시도"""
        captcha = await self.detect_captcha(page)

        if not captcha['found']:
            return True  # CAPTCHA 없음

        print(f"   🔒 CAPTCHA 감지: {captcha['type']}")

        if captcha['type'] == 'cloudflare':
            return await self.handle_cloudflare(page)

        elif captcha['type'] == 'checkbox':
            # 단순 체크박스
            checkbox = await page.query_selector('input[type="checkbox"]')
            if checkbox:
                await checkbox.click()
                await asyncio.sleep(2)
                self.solved_count += 1
                return True

        elif captcha['type'] == 'text_captcha':
            text = await self.solve_text_captcha(page, captcha.get('imgSrc', ''))
            if text:
                # 입력 필드 찾기
                input_field = await page.query_selector('input[name*="captcha"], input[id*="captcha"]')
                if input_field:
                    await input_field.fill(text)
                    await page.keyboard.press('Enter')
                    await asyncio.sleep(2)
                    self.solved_count += 1
                    return True

        elif captcha['type'] in ['recaptcha_v2', 'hcaptcha']:
            # Claude로 분석 시도
            if self.claude:
                return await self.solve_image_captcha_with_claude(page)

        self.failed_count += 1
        return False


class BrowserUseTracker:
    """브라우저 사용 패턴 학습"""

    def __init__(self):
        self.actions = []  # 행동 기록
        self.successful_patterns = []  # 성공한 패턴
        self.data_dir = Path("/home/kim/agi/browser_use_data")
        self.data_dir.mkdir(exist_ok=True)
        self._load()

    def _load(self):
        try:
            with open(self.data_dir / "patterns.json") as f:
                data = json.load(f)
                self.successful_patterns = data.get("patterns", [])
        except:
            pass

    def _save(self):
        with open(self.data_dir / "patterns.json", 'w') as f:
            json.dump({"patterns": self.successful_patterns[-100:]}, f, indent=2)

    def record_action(self, action: dict):
        """행동 기록"""
        action['timestamp'] = datetime.now().isoformat()
        self.actions.append(action)

    def mark_success(self, goal: str):
        """목표 달성 시 패턴 저장"""
        if len(self.actions) > 0:
            pattern = {
                "goal": goal,
                "actions": self.actions[-10:],  # 최근 10개 행동
                "timestamp": datetime.now().isoformat()
            }
            self.successful_patterns.append(pattern)
            self._save()
            self.actions = []  # 리셋

    def get_similar_pattern(self, goal: str) -> list:
        """유사한 목표의 성공 패턴 찾기"""
        for pattern in reversed(self.successful_patterns):
            if any(word in pattern['goal'].lower() for word in goal.lower().split()):
                return pattern['actions']
        return []


class LifeAgentV3:
    """CAPTCHA 해결 + Browser Use 학습 에이전트"""

    def __init__(self):
        self.browser = None
        self.page = None
        self.agi = None

        self.birth_time = datetime.now()
        self.life_purpose = random.choice(list(LIFE_PURPOSES.keys()))

        # 상태
        self.knowledge_base = []
        self.insights = []
        self.questions = []
        self.visited_urls = set()

        # 통계
        self.total_pages = 0
        self.total_actions = 0
        self.thinking_sessions = 0
        self.ultrathink_count = 0

        # 데이터 디렉토리
        self.data_dir = Path("/home/kim/agi/life_agent_data")
        self.data_dir.mkdir(exist_ok=True)

        # Claude
        self.claude = None
        if HAS_CLAUDE and os.environ.get("ANTHROPIC_API_KEY"):
            self.claude = anthropic.Anthropic()

        # CAPTCHA 솔버 (나중에 초기화)
        self.captcha_solver = None

        # Browser Use 학습
        self.browser_tracker = BrowserUseTracker()

        self._load_state()

    def _load_state(self):
        state_file = self.data_dir / "life_state_v3.json"
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
                print(f"📚 Loaded: {len(self.knowledge_base)} knowledge")
            except Exception as e:
                print(f"⚠️ Load failed: {e}")

    def _save_state(self):
        state = {
            "life_purpose": self.life_purpose,
            "knowledge": self.knowledge_base[-100:],
            "insights": self.insights[-50:],
            "questions": self.questions[-30:],
            "visited_urls": list(self.visited_urls)[-500:],
            "total_pages": self.total_pages,
            "thinking_sessions": self.thinking_sessions,
            "ultrathink_count": self.ultrathink_count,
            "captcha_solved": self.captcha_solver.solved_count if self.captcha_solver else 0,
            "last_save": datetime.now().isoformat()
        }
        with open(self.data_dir / "life_state_v3.json", 'w') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    async def setup(self):
        """초기화"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=['--no-sandbox'],
            slow_mo=50
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1400, 'height': 900},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        self.page = await self.context.new_page()
        print("🌐 Browser ready")

        print("🧠 Loading local brain (LFM2)...")
        from agents.lfm2_adapter import LFM2VLAdapter, LFM2Config
        config = LFM2Config(model_id="LiquidAI/LFM2-VL-1.6B", enable_continual_learning=True)
        self.agi = LFM2VLAdapter(lfm2_config=config)
        await self.agi.load_model()

        # CAPTCHA 솔버 초기화
        self.captcha_solver = CaptchaSolver(self.agi, self.claude)

        purpose = LIFE_PURPOSES[self.life_purpose]
        print(f"✨ Life Agent v3 ready! Purpose: {purpose['name']}")
        print(f"   CAPTCHA Solver: ✅")
        print(f"   Claude UltraThink: {'✅' if self.claude else '❌'}")

    async def local_think(self, prompt: str) -> str:
        try:
            response = await self.agi.execute(prompt)
            return response.content[:400]
        except Exception as e:
            return f"생각 실패: {e}"

    async def navigate_with_captcha(self, url: str) -> bool:
        """CAPTCHA 처리하면서 네비게이션"""
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)

            # CAPTCHA 체크 및 해결
            if not await self.captcha_solver.solve(self.page):
                print(f"   ⚠️ CAPTCHA 해결 실패 - 다른 페이지로 이동")
                return False

            self.browser_tracker.record_action({
                "type": "navigate",
                "url": url,
                "success": True
            })

            return True

        except Exception as e:
            print(f"   ❌ Navigation failed: {e}")
            return False

    def get_random_source(self) -> str:
        purpose = LIFE_PURPOSES[self.life_purpose]
        preferred = purpose['preferred_sources']

        if random.random() < 0.7:
            category = random.choice(preferred)
        else:
            category = random.choice(list(LEARNING_SOURCES.keys()))

        return random.choice(LEARNING_SOURCES[category])

    async def decide_what_to_do(self) -> dict:
        purpose = LIFE_PURPOSES[self.life_purpose]
        recent_knowledge = " ".join([k[:50] for k in self.knowledge_base[-3:]])

        if random.random() < 0.3:
            topic = random.choice(purpose['search_topics'])
            return {"action": "SEARCH", "target": topic}

        source = self.get_random_source()

        prompt = f"""나의 목표: {purpose['name']}
최근 배운 것: {recent_knowledge[:150]}
방문할 곳: {source}

다음 행동 (EXPLORE/SEARCH/REFLECT):"""

        result = await self.local_think(prompt)
        upper = result.upper()

        if "REFLECT" in upper:
            return {"action": "REFLECT", "target": ""}
        elif "SEARCH" in upper:
            topic = random.choice(purpose['search_topics'])
            return {"action": "SEARCH", "target": topic}
        else:
            return {"action": "EXPLORE", "target": source}

    async def execute_action(self, action: str, target: str):
        self.total_actions += 1

        if action == "EXPLORE":
            await self.explore_source(target)
        elif action == "SEARCH":
            await self.search_and_learn(target)
        elif action == "REFLECT":
            await self.reflect()

    async def explore_source(self, url: str):
        if url in self.visited_urls:
            url = self.get_random_source()

        print(f"\n🔭 Exploring: {url[:60]}")

        if not await self.navigate_with_captcha(url):
            return

        self.visited_urls.add(url)
        await self._read_and_learn()

        # 링크 클릭
        for _ in range(2):
            if random.random() < 0.6:
                await self._click_interesting_link()
                await asyncio.sleep(2)

                # CAPTCHA 체크
                await self.captcha_solver.solve(self.page)
                await self._read_and_learn()

    async def search_and_learn(self, query: str):
        print(f"\n🔍 Searching: {query}")

        # DuckDuckGo (CAPTCHA 적음)
        if not await self.navigate_with_captcha(f"https://duckduckgo.com/?q={query.replace(' ', '+')}"):
            return

        await asyncio.sleep(2)

        links = await self.page.query_selector_all('a[data-testid="result-title-a"]')
        for link in links[:2]:
            try:
                href = await link.get_attribute("href")
                if href and href not in self.visited_urls:
                    await link.click(timeout=5000)
                    await asyncio.sleep(2)

                    # CAPTCHA 처리
                    await self.captcha_solver.solve(self.page)

                    self.visited_urls.add(self.page.url)
                    await self._read_and_learn()
                    await self.page.go_back(timeout=5000)
            except:
                continue

    async def _click_interesting_link(self):
        try:
            links = await self.page.query_selector_all('a[href]')
            interesting = []

            keywords = ["AI", "machine", "learn", "research", "tech", "science",
                       "data", "python", "neural", "model", "news"]

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

                self.browser_tracker.record_action({
                    "type": "click",
                    "text": text[:50]
                })

                self.visited_urls.add(self.page.url)

        except Exception as e:
            print(f"   ❌ Click failed: {e}")

    async def _read_and_learn(self):
        try:
            title = await self.page.title()
            url = self.page.url

            text = await self.page.evaluate("""
                () => {
                    const main = document.querySelector('article, main, .content') || document.body;
                    return main.innerText.slice(0, 3000);
                }
            """)

            if len(text) < 100:
                return

            self.total_pages += 1

            summary = await self.local_think(f"핵심을 50자로: {text[:1000]}")

            if summary[:50] not in [k[:50] for k in self.knowledge_base[-10:]]:
                self.knowledge_base.append(summary)
                print(f"   📖 Read: {title[:40]}")
                print(f"   💡 Learned: {summary[:80]}")

                # 성공 기록
                self.browser_tracker.mark_success(f"Learn about {title[:30]}")

                if random.random() < 0.2:
                    question = await self.local_think(f"이 내용에서 떠오르는 질문: {summary}")
                    self.questions.append(question)
            else:
                print(f"   ⏭️ Skip duplicate")

        except Exception as e:
            print(f"   ❌ Read failed: {e}")

    async def reflect(self):
        print(f"\n🪞 Reflecting...")
        self.thinking_sessions += 1

        recent = " ".join(self.knowledge_base[-10:])

        reflection = await self.local_think(f"배운 것: {recent[:800]}\n\n가장 중요한 교훈은?")
        print(f"   💭 {reflection[:100]}")

        # 10회마다 Claude
        if self.thinking_sessions % 10 == 0 and self.claude:
            print(f"   🧠 UltraThinking...")
            self.ultrathink_count += 1

        self._save_state()

        # 통계
        print(f"\n📊 Stats:")
        print(f"   📚 Pages: {self.total_pages}")
        print(f"   💡 Knowledge: {len(self.knowledge_base)}")
        print(f"   🔐 CAPTCHAs solved: {self.captcha_solver.solved_count}")
        print(f"   🎯 Browser patterns: {len(self.browser_tracker.successful_patterns)}")

    async def live(self):
        purpose = LIFE_PURPOSES[self.life_purpose]
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║       🌟 LIFE AGENT v3 - CAPTCHA + Browser Use 🌟         ║
║                                                           ║
║   Purpose: {purpose['name']:^43} ║
║   "I solve CAPTCHAs, I learn, I grow"                     ║
╚═══════════════════════════════════════════════════════════╝
""")

        cycle = 0
        while True:
            cycle += 1
            print(f"\n{'='*60}")
            print(f"🔄 Cycle {cycle} | Pages: {self.total_pages} | CAPTCHAs: {self.captcha_solver.solved_count}")

            try:
                decision = await self.decide_what_to_do()
                action = decision['action']
                target = decision['target']

                print(f"🎯 Decision: {action} - {target[:50] if target else 'N/A'}")

                await self.execute_action(action, target)

                if cycle % 7 == 0:
                    await self.reflect()

                if cycle % 5 == 0:
                    self._save_state()

            except Exception as e:
                print(f"❌ Error: {e}")
                try:
                    self.page = await self.context.new_page()
                except:
                    pass

            wait = random.randint(8, 20)
            print(f"⏳ Next in {wait}s...")
            await asyncio.sleep(wait)

    async def close(self):
        self._save_state()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


async def main():
    agent = LifeAgentV3()

    try:
        await agent.setup()
        await agent.live()
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Life paused.")
        print(f"📚 Knowledge: {len(agent.knowledge_base)}")
        print(f"🔐 CAPTCHAs: {agent.captcha_solver.solved_count}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
