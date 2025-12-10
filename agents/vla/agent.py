"""
VLA Agent - Main Vision-Language-Action Agent

Dual-System Architecture:
- System 2 (Slow): Vision-Language Model for perception & planning
- System 1 (Fast): Action generation from learned patterns

Based on Helix (Figure AI) and GR00T N1 (NVIDIA) architectures.
"""

import asyncio
import base64
import io
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .system1 import System1ActionGenerator
from .system2 import System2VisionPlanner

try:
    from playwright.async_api import async_playwright, Page, Browser
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


class ActionType(Enum):
    """Supported browser actions"""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    CAPTCHA = "captcha"
    DONE = "done"
    NAVIGATE = "navigate"


@dataclass
class Action:
    """A browser action to execute"""
    type: ActionType
    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = None
    direction: Optional[str] = None  # up/down for scroll
    url: Optional[str] = None
    target: Optional[str] = None  # Description of what was clicked
    confidence: float = 0.0


class VLAAgent:
    """
    Vision-Language-Action Agent for autonomous browser control.

    Uses dual-system architecture:
    - System 2: Slow, deliberate reasoning with VLM
    - System 1: Fast, pattern-based action generation

    Flow:
    1. Take screenshot
    2. System 2 analyzes screenshot + goal → high-level plan
    3. System 1 generates precise action from plan + patterns
    4. Execute action
    5. Loop until goal achieved
    """

    def __init__(self, vlm_model: str = "lfm2", max_steps: int = 20,
                 patterns_file: str = "data/patterns/browser_patterns.json"):
        self.vlm_model = vlm_model
        self.max_steps = max_steps
        self.patterns_file = patterns_file

        # Dual-system components
        self.system2 = System2VisionPlanner(model=vlm_model)
        self.system1 = System1ActionGenerator(patterns_file=patterns_file)

        # State
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.current_goal: str = ""
        self.action_history: List[Action] = []
        self.screenshot_history: List[bytes] = []

    async def initialize(self, headless: bool = True):
        """Initialize browser"""
        if not HAS_PLAYWRIGHT:
            raise ImportError("Playwright not installed. Run: pip install playwright && playwright install chromium")

        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=headless,
            args=['--disable-blink-features=AutomationControlled']
        )
        context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        self.page = await context.new_page()
        print("[VLA] Browser initialized")

    async def execute(self, task: str, context: Optional[Dict] = None) -> Dict:
        """
        Execute a browser task.

        Args:
            task: Goal to achieve (e.g., "Navigate to HackerNews and read top story")
            context: Additional context (e.g., starting URL)

        Returns:
            Dict with success status, actions taken, and result
        """
        self.current_goal = task
        self.action_history = []
        self.screenshot_history = []

        # Initialize browser if needed
        if not self.page:
            await self.initialize()

        # Navigate to starting URL if provided
        if context and context.get("url"):
            await self.page.goto(context["url"])
            await asyncio.sleep(1)

        print(f"\n{'='*60}")
        print(f"[VLA] Goal: {task}")
        print(f"{'='*60}")

        # Main execution loop
        for step in range(self.max_steps):
            print(f"\n--- Step {step + 1}/{self.max_steps} ---")

            # 1. Take screenshot
            screenshot = await self._take_screenshot()
            self.screenshot_history.append(screenshot)

            # 2. System 2: Analyze and plan (slow thinking)
            plan = await self.system2.analyze(
                screenshot=screenshot,
                goal=self.current_goal,
                history=self.action_history
            )

            print(f"[System 2] Plan: {plan.get('reasoning', 'No reasoning')[:100]}...")

            # Check if goal achieved
            if plan.get("goal_achieved"):
                print(f"[VLA] Goal achieved!")
                return {
                    "success": True,
                    "goal": task,
                    "steps": step + 1,
                    "actions": [self._action_to_dict(a) for a in self.action_history],
                    "result": plan.get("result", "Goal completed")
                }

            # 3. System 1: Generate precise action (fast action)
            action = self.system1.generate(
                plan=plan,
                screenshot_size=(1280, 720),
                history=self.action_history
            )

            print(f"[System 1] Action: {action.type.value} at ({action.x}, {action.y}) - {action.target}")

            # 4. Execute action
            success = await self._execute_action(action)
            self.action_history.append(action)

            if not success:
                print(f"[VLA] Action failed, trying recovery...")
                # Try recovery
                recovery_action = self.system1.recover(action, self.action_history)
                if recovery_action:
                    await self._execute_action(recovery_action)
                    self.action_history.append(recovery_action)

            # Small delay between actions
            await asyncio.sleep(0.5)

        # Max steps reached
        return {
            "success": False,
            "goal": task,
            "steps": self.max_steps,
            "actions": [self._action_to_dict(a) for a in self.action_history],
            "result": "Max steps reached without achieving goal"
        }

    async def _take_screenshot(self) -> bytes:
        """Take screenshot of current page"""
        return await self.page.screenshot(type='png')

    async def _execute_action(self, action: Action) -> bool:
        """Execute a browser action"""
        try:
            if action.type == ActionType.CLICK:
                await self.page.mouse.click(action.x, action.y)

            elif action.type == ActionType.TYPE:
                await self.page.mouse.click(action.x, action.y)
                await asyncio.sleep(0.2)
                await self.page.keyboard.type(action.text, delay=50)
                await self.page.keyboard.press('Enter')

            elif action.type == ActionType.SCROLL:
                delta = -300 if action.direction == "down" else 300
                await self.page.mouse.wheel(0, delta)

            elif action.type == ActionType.WAIT:
                await asyncio.sleep(2)

            elif action.type == ActionType.NAVIGATE:
                await self.page.goto(action.url)
                await asyncio.sleep(1)

            elif action.type == ActionType.CAPTCHA:
                # Special handling for CAPTCHA
                return await self._solve_captcha(action)

            elif action.type == ActionType.DONE:
                return True

            return True

        except Exception as e:
            print(f"[VLA] Action execution error: {e}")
            return False

    async def _solve_captcha(self, action: Action) -> bool:
        """Attempt to solve CAPTCHA"""
        print("[VLA] Attempting CAPTCHA solve...")

        # Click the CAPTCHA checkbox
        if action.x and action.y:
            await self.page.mouse.click(action.x, action.y)
            await asyncio.sleep(2)

            # Check if challenge appeared
            screenshot = await self._take_screenshot()
            plan = await self.system2.analyze(
                screenshot=screenshot,
                goal="Check if CAPTCHA was solved",
                history=self.action_history
            )

            return plan.get("captcha_solved", False)

        return False

    def _action_to_dict(self, action: Action) -> Dict:
        """Convert Action to dictionary"""
        return {
            "type": action.type.value,
            "x": action.x,
            "y": action.y,
            "text": action.text,
            "target": action.target,
            "confidence": action.confidence
        }

    async def close(self):
        """Close browser"""
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.page = None

    def save_patterns(self):
        """Save learned patterns to file"""
        if self.action_history and len(self.action_history) >= 2:
            self.system1.save_pattern(
                goal=self.current_goal,
                actions=self.action_history,
                success=True
            )

    def get_statistics(self) -> Dict:
        """Get agent statistics"""
        return {
            "total_actions": len(self.action_history),
            "patterns_learned": self.system1.get_pattern_count(),
            "current_goal": self.current_goal
        }


# Factory function
def create_vla_agent(vlm_model: str = "lfm2", max_steps: int = 20) -> VLAAgent:
    """Create a configured VLA agent"""
    return VLAAgent(vlm_model=vlm_model, max_steps=max_steps)
