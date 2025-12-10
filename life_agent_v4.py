#!/usr/bin/env python3
"""
🌟 Life Agent v4 - 비전 기반 브라우저 조작 + 실시간 지속학습
스크린샷을 보고 어디를 클릭할지 판단, 행동 패턴 학습
"""
import asyncio
import json
import random
import sys
import os
from datetime import datetime
from pathlib import Path
from io import BytesIO

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("pip install playwright && playwright install chromium")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))

# 학습 소스
LEARNING_SOURCES = [
    "https://news.ycombinator.com/newest",
    "https://www.reddit.com/r/MachineLearning/new/",
    "https://arxiv.org/list/cs.AI/recent",
    "https://news.naver.com/section/105",
    "https://en.wikipedia.org/wiki/Special:Random",
    "https://lobste.rs/",
]

LIFE_PURPOSES = {
    "growth": {"name": "지속적 성장", "description": "끊임없이 배우고 발전한다"},
    "value_creation": {"name": "가치 창출", "description": "유용한 인사이트를 생성한다"},
    "understanding": {"name": "세상 이해", "description": "세상이 어떻게 돌아가는지 이해한다"}
}


class VisionBrowserAgent:
    """스크린샷 기반 브라우저 조작 - 비전 모델이 모든 것을 판단"""

    def __init__(self, vision_model):
        self.vision_model = vision_model
        self.action_history = []  # 행동 기록
        self.success_patterns = []  # 성공 패턴
        self.data_dir = Path("/home/kim/agi/vision_agent_data")
        self.data_dir.mkdir(exist_ok=True)
        self._load_patterns()

    def _load_patterns(self):
        try:
            with open(self.data_dir / "patterns.json") as f:
                self.success_patterns = json.load(f).get("patterns", [])
        except:
            pass

    def _save_patterns(self):
        with open(self.data_dir / "patterns.json", 'w') as f:
            json.dump({"patterns": self.success_patterns[-200:]}, f, indent=2)

    async def analyze_screen(self, screenshot: bytes, goal: str) -> dict:
        """스크린샷을 분석해서 다음 행동 결정"""
        prompt = f"""이 브라우저 스크린샷을 보고 다음 행동을 결정해주세요.

현재 목표: {goal}

화면을 분석하고 다음 중 하나를 선택해서 정확히 답해주세요:

1. CAPTCHA가 보이면:
   ACTION: CLICK_CAPTCHA
   X: [체크박스 중심 x좌표 (0-1400)]
   Y: [체크박스 중심 y좌표 (0-900)]

2. 클릭할 링크/버튼이 보이면:
   ACTION: CLICK
   X: [클릭할 x좌표]
   Y: [클릭할 y좌표]
   TARGET: [클릭 대상 설명]

3. 검색창이 보이면:
   ACTION: TYPE
   X: [입력창 x좌표]
   Y: [입력창 y좌표]
   TEXT: [입력할 텍스트]

4. 스크롤이 필요하면:
   ACTION: SCROLL
   DIRECTION: down 또는 up

5. 페이지 로딩 대기가 필요하면:
   ACTION: WAIT
   REASON: [이유]

6. 목표 달성됨:
   ACTION: DONE
   RESULT: [결과 요약]

형식을 정확히 지켜서 답해주세요."""

        try:
            response = await self.vision_model.execute(prompt, images=[screenshot])
            result = self._parse_action(response.content)
            return result
        except Exception as e:
            print(f"   ❌ 분석 실패: {e}")
            return {"action": "WAIT", "reason": str(e)}

    def _parse_action(self, text: str) -> dict:
        """응답에서 행동 파싱"""
        result = {"action": "WAIT", "raw": text}

        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("ACTION:"):
                result["action"] = line.split("ACTION:")[-1].strip().upper()
            elif line.startswith("X:"):
                try:
                    result["x"] = int(line.split("X:")[-1].strip().split()[0])
                except:
                    pass
            elif line.startswith("Y:"):
                try:
                    result["y"] = int(line.split("Y:")[-1].strip().split()[0])
                except:
                    pass
            elif line.startswith("TARGET:"):
                result["target"] = line.split("TARGET:")[-1].strip()
            elif line.startswith("TEXT:"):
                result["text"] = line.split("TEXT:")[-1].strip()
            elif line.startswith("DIRECTION:"):
                result["direction"] = line.split("DIRECTION:")[-1].strip().lower()
            elif line.startswith("RESULT:"):
                result["result"] = line.split("RESULT:")[-1].strip()
            elif line.startswith("REASON:"):
                result["reason"] = line.split("REASON:")[-1].strip()

        return result

    async def execute_action(self, page, action: dict) -> bool:
        """행동 실행"""
        action_type = action.get("action", "WAIT")
        print(f"   🎯 Action: {action_type}")

        try:
            if action_type in ["CLICK", "CLICK_CAPTCHA"]:
                x = action.get("x", 700)
                y = action.get("y", 450)
                print(f"   🖱️ Click at ({x}, {y})")
                await page.mouse.click(x, y)
                await asyncio.sleep(2)
                return True

            elif action_type == "TYPE":
                x = action.get("x", 700)
                y = action.get("y", 450)
                text = action.get("text", "AI")
                await page.mouse.click(x, y)
                await asyncio.sleep(0.5)
                await page.keyboard.type(text, delay=50)
                await page.keyboard.press("Enter")
                await asyncio.sleep(2)
                return True

            elif action_type == "SCROLL":
                direction = action.get("direction", "down")
                amount = 400 if direction == "down" else -400
                await page.mouse.wheel(0, amount)
                await asyncio.sleep(1)
                return True

            elif action_type == "WAIT":
                await asyncio.sleep(3)
                return True

            elif action_type == "DONE":
                return True

        except Exception as e:
            print(f"   ❌ Action failed: {e}")
            return False

        return True

    def record_success(self, goal: str, actions: list):
        """성공 패턴 저장"""
        pattern = {
            "goal": goal,
            "actions": actions[-10:],
            "timestamp": datetime.now().isoformat()
        }
        self.success_patterns.append(pattern)
        self._save_patterns()
        print(f"   ✅ 패턴 저장됨 (총 {len(self.success_patterns)}개)")


class LifeAgentV4:
    """비전 기반 자율 학습 에이전트"""

    def __init__(self):
        self.browser = None
        self.page = None
        self.agi = None

        self.birth_time = datetime.now()
        self.life_purpose = random.choice(list(LIFE_PURPOSES.keys()))

        # 상태
        self.knowledge_base = []
        self.visited_urls = set()
        self.total_pages = 0
        self.total_actions = 0

        # 데이터 디렉토리
        self.data_dir = Path("/home/kim/agi/life_agent_data")
        self.data_dir.mkdir(exist_ok=True)

        # 비전 브라우저 에이전트 (나중에 초기화)
        self.vision_agent = None

        self._load_state()

    def _load_state(self):
        state_file = self.data_dir / "life_state_v4.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                    self.knowledge_base = state.get("knowledge", [])[-100:]
                    self.visited_urls = set(state.get("visited_urls", [])[-500:])
                    self.total_pages = state.get("total_pages", 0)
                print(f"📚 Loaded: {len(self.knowledge_base)} knowledge")
            except:
                pass

    def _save_state(self):
        state = {
            "life_purpose": self.life_purpose,
            "knowledge": self.knowledge_base[-100:],
            "visited_urls": list(self.visited_urls)[-500:],
            "total_pages": self.total_pages,
            "total_actions": self.total_actions,
            "last_save": datetime.now().isoformat()
        }
        with open(self.data_dir / "life_state_v4.json", 'w') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    async def setup(self):
        """초기화"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=['--no-sandbox'],
            slow_mo=100  # 동작 보이게
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1400, 'height': 900},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        self.page = await self.context.new_page()
        print("🌐 Browser ready")

        print("🧠 Loading vision model (LFM2-VL)...")
        from agents.lfm2_adapter import LFM2VLAdapter, LFM2Config
        config = LFM2Config(model_id="LiquidAI/LFM2-VL-1.6B", enable_continual_learning=True)
        self.agi = LFM2VLAdapter(lfm2_config=config)
        await self.agi.load_model()

        # 비전 에이전트 초기화
        self.vision_agent = VisionBrowserAgent(self.agi)

        purpose = LIFE_PURPOSES[self.life_purpose]
        print(f"✨ Life Agent v4 ready! Purpose: {purpose['name']}")
        print(f"   📸 Vision-based browser control enabled")

    async def take_screenshot(self) -> bytes:
        """스크린샷 촬영"""
        return await self.page.screenshot(type='png')

    async def explore_with_vision(self, url: str, goal: str, max_actions: int = 10):
        """비전 기반 탐험"""
        print(f"\n🔭 Exploring: {url[:50]}")
        print(f"   🎯 Goal: {goal}")

        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            print(f"   ❌ Navigation failed: {e}")
            return

        await asyncio.sleep(2)
        actions_taken = []

        for i in range(max_actions):
            print(f"\n--- Step {i+1}/{max_actions} ---")

            # 스크린샷 촬영
            screenshot = await self.take_screenshot()

            # 비전 모델로 분석
            action = await self.vision_agent.analyze_screen(screenshot, goal)
            actions_taken.append(action)

            # 행동 실행
            success = await self.vision_agent.execute_action(self.page, action)
            self.total_actions += 1

            # DONE이면 종료
            if action.get("action") == "DONE":
                print(f"   ✅ Goal achieved: {action.get('result', 'success')}")
                self.vision_agent.record_success(goal, actions_taken)
                break

            # 페이지 내용 학습
            if action.get("action") in ["CLICK", "DONE"]:
                await self._learn_from_page()

            await asyncio.sleep(1)

        self.visited_urls.add(url)

    async def _learn_from_page(self):
        """현재 페이지에서 학습"""
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

            # 요약 생성
            summary = await self.agi.execute(f"핵심을 50자로: {text[:800]}")
            summary_text = summary.content[:200]

            # 중복 체크
            if summary_text[:50] not in [k[:50] for k in self.knowledge_base[-10:]]:
                self.knowledge_base.append(summary_text)
                print(f"   📖 Read: {title[:40]}")
                print(f"   💡 Learned: {summary_text[:80]}")

        except Exception as e:
            print(f"   ❌ Learn failed: {e}")

    async def live(self):
        """살아가기"""
        purpose = LIFE_PURPOSES[self.life_purpose]
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║     🌟 LIFE AGENT v4 - Vision-Based Browser Control 🌟    ║
║                                                           ║
║   Purpose: {purpose['name']:^43} ║
║   "I see, I think, I act, I learn"                        ║
╚═══════════════════════════════════════════════════════════╝
""")

        goals = [
            "최신 AI 뉴스 읽기",
            "흥미로운 기사 찾아서 읽기",
            "새로운 정보 학습하기",
            "검색해서 정보 찾기",
        ]

        cycle = 0
        while True:
            cycle += 1
            print(f"\n{'='*60}")
            print(f"🔄 Cycle {cycle} | Pages: {self.total_pages} | Actions: {self.total_actions}")

            try:
                # 랜덤 소스 선택
                url = random.choice(LEARNING_SOURCES)
                goal = random.choice(goals)

                # 비전 기반 탐험
                await self.explore_with_vision(url, goal, max_actions=8)

                # 상태 저장
                if cycle % 3 == 0:
                    self._save_state()

            except Exception as e:
                print(f"❌ Error: {e}")
                try:
                    self.page = await self.context.new_page()
                except:
                    pass

            wait = random.randint(10, 20)
            print(f"⏳ Next in {wait}s...")
            await asyncio.sleep(wait)

    async def close(self):
        self._save_state()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


async def main():
    agent = LifeAgentV4()

    try:
        await agent.setup()
        await agent.live()
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Life paused.")
        print(f"📚 Knowledge: {len(agent.knowledge_base)}")
        print(f"🎯 Actions: {agent.total_actions}")
        print(f"📸 Patterns: {len(agent.vision_agent.success_patterns)}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
