#!/usr/bin/env python3
"""
AGI Trinity - Computer Use Learner (VLA Style)
마우스/키보드 조작 + 스크린샷으로 컴퓨터 사용법 학습
OpenAI VLA / RT-2 스타일 데이터 수집

데이터 형식:
{
    "screenshot_before": image,
    "action": {"type": "click/type/scroll", "x": 100, "y": 200, "text": "..."},
    "screenshot_after": image,
    "task_description": "..."
}
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import base64

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("playwright not installed. Run: pip install playwright && playwright install chromium")
    sys.exit(1)

try:
    import pyautogui
    from pynput import mouse, keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("pyautogui/pynput not installed. Run: pip install pyautogui pynput")

sys.path.insert(0, str(Path(__file__).parent))


class ComputerUseLearner:
    """VLA 스타일 컴퓨터 사용 학습기"""

    def __init__(self, data_dir: str = "/home/kim/agi/computer_use_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.agi = None
        self.recording = False
        self.episode_data = []
        self.current_episode = 0
        self.last_screenshot = None
        self.action_buffer = []

    async def setup_model(self):
        """모델 로드"""
        print("🧠 Loading VL model...")
        start = time.time()

        from agents.lfm2_adapter import LFM2VLAdapter, LFM2Config

        config = LFM2Config(
            model_id="LiquidAI/LFM2-VL-1.6B",
            enable_continual_learning=True
        )
        self.agi = LFM2VLAdapter(lfm2_config=config)
        await self.agi.load_model()
        print(f"✅ Model loaded in {time.time() - start:.1f}s")

    def take_screenshot(self) -> str:
        """스크린샷 찍고 base64로 반환"""
        screenshot_path = self.data_dir / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        pyautogui.screenshot(str(screenshot_path))
        return str(screenshot_path)

    def start_recording(self, task_description: str = ""):
        """조작 기록 시작"""
        self.recording = True
        self.current_episode += 1
        self.episode_data = []
        self.task_description = task_description

        print(f"\n🔴 Recording Episode {self.current_episode}")
        if task_description:
            print(f"   Task: {task_description}")

        # 초기 스크린샷
        self.last_screenshot = self.take_screenshot()

        # 마우스/키보드 리스너 시작
        self._start_listeners()

    def stop_recording(self):
        """기록 중지"""
        self.recording = False
        self._stop_listeners()

        # 에피소드 저장
        episode_file = self.data_dir / f"episode_{self.current_episode:04d}.json"
        with open(episode_file, 'w', encoding='utf-8') as f:
            json.dump({
                "episode_id": self.current_episode,
                "task_description": self.task_description,
                "actions": self.episode_data,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

        print(f"⏹️ Stopped. Saved {len(self.episode_data)} actions to {episode_file}")

    def _start_listeners(self):
        """마우스/키보드 이벤트 리스너 시작"""
        def on_click(x, y, button, pressed):
            if not self.recording or not pressed:
                return
            self._record_action({
                "type": "click",
                "x": x,
                "y": y,
                "button": str(button)
            })

        def on_key(key):
            if not self.recording:
                return
            try:
                char = key.char
            except AttributeError:
                char = str(key)
            self._record_action({
                "type": "keypress",
                "key": char
            })

        self.mouse_listener = mouse.Listener(on_click=on_click)
        self.keyboard_listener = keyboard.Listener(on_press=on_key)

        self.mouse_listener.start()
        self.keyboard_listener.start()

    def _stop_listeners(self):
        """리스너 중지"""
        if hasattr(self, 'mouse_listener'):
            self.mouse_listener.stop()
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()

    def _record_action(self, action: dict):
        """액션 기록"""
        # 현재 스크린샷
        screenshot_after = self.take_screenshot()

        action_data = {
            "screenshot_before": self.last_screenshot,
            "action": action,
            "screenshot_after": screenshot_after,
            "timestamp": datetime.now().isoformat()
        }

        self.episode_data.append(action_data)
        self.last_screenshot = screenshot_after

        action_str = f"{action['type']}"
        if action['type'] == 'click':
            action_str += f" ({action['x']}, {action['y']})"
        elif action['type'] == 'keypress':
            action_str += f" [{action['key']}]"

        print(f"   📝 {action_str}")

    async def analyze_screenshot(self, screenshot_path: str) -> str:
        """스크린샷 분석"""
        if not self.agi:
            return "모델 미로드"

        try:
            response = await self.agi.execute(
                "이 화면에서 무엇을 할 수 있는지 간단히 설명해주세요.",
                images=[screenshot_path]
            )
            return response.content[:200]
        except Exception as e:
            return f"분석 실패: {e}"

    async def autonomous_task(self, task: str, max_steps: int = 10):
        """자율 태스크 수행 (AI가 직접 조작)"""
        print(f"\n🤖 Autonomous Task: {task}")

        for step in range(max_steps):
            # 현재 화면 캡처
            screenshot = self.take_screenshot()

            # AI에게 다음 행동 질문
            prompt = f"""현재 화면을 보고 다음 작업을 수행해주세요: {task}

다음 형식으로 응답해주세요:
ACTION: click/type/scroll/done
X: (클릭 x좌표, 0-1920)
Y: (클릭 y좌표, 0-1080)
TEXT: (입력할 텍스트, type인 경우)
REASON: (이유)"""

            try:
                response = await self.agi.execute(prompt, images=[screenshot])
                print(f"   Step {step+1}: {response.content[:100]}...")

                # 응답 파싱
                action = self._parse_action(response.content)
                if action['type'] == 'done':
                    print("   ✅ Task completed!")
                    break

                # 액션 수행
                await self._execute_action(action)
                await asyncio.sleep(1)

            except Exception as e:
                print(f"   ❌ Error: {e}")
                break

    def _parse_action(self, text: str) -> dict:
        """응답에서 액션 파싱"""
        action = {"type": "done"}

        lines = text.upper().split('\n')
        for line in lines:
            if line.startswith('ACTION:'):
                action['type'] = line.split(':')[1].strip().lower()
            elif line.startswith('X:'):
                try:
                    action['x'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('Y:'):
                try:
                    action['y'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('TEXT:'):
                action['text'] = line.split(':')[1].strip()

        return action

    async def _execute_action(self, action: dict):
        """액션 수행"""
        action_type = action.get('type', 'done')

        if action_type == 'click':
            x, y = action.get('x', 500), action.get('y', 500)
            print(f"   🖱️ Clicking ({x}, {y})")
            pyautogui.click(x, y)

        elif action_type == 'type':
            text = action.get('text', '')
            print(f"   ⌨️ Typing: {text[:30]}...")
            pyautogui.write(text, interval=0.05)

        elif action_type == 'scroll':
            print("   🔄 Scrolling")
            pyautogui.scroll(-3)

    async def demo_mode(self):
        """데모 모드 - 실시간 화면 분석"""
        print("\n🎮 Demo Mode - 실시간 화면 분석")
        print("   Ctrl+C로 종료")

        while True:
            screenshot = self.take_screenshot()
            analysis = await self.analyze_screenshot(screenshot)
            print(f"\n📸 화면 분석: {analysis}")
            await asyncio.sleep(5)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Computer Use Learner - VLA 스타일")
    parser.add_argument("--mode", choices=["record", "demo", "auto"], default="demo")
    parser.add_argument("--task", type=str, default="", help="Task description")
    args = parser.parse_args()

    if not PYNPUT_AVAILABLE:
        print("❌ pynput이 필요합니다: pip install pynput pyautogui")
        return

    learner = ComputerUseLearner()

    try:
        await learner.setup_model()

        if args.mode == "demo":
            await learner.demo_mode()
        elif args.mode == "record":
            learner.start_recording(args.task)
            print("Press Ctrl+C to stop recording...")
            while True:
                await asyncio.sleep(1)
        elif args.mode == "auto":
            if not args.task:
                args.task = "웹 브라우저를 열고 구글에서 'AI news' 검색"
            await learner.autonomous_task(args.task)

    except KeyboardInterrupt:
        print("\n⏹️ Stopping...")
        if learner.recording:
            learner.stop_recording()


if __name__ == "__main__":
    asyncio.run(main())
