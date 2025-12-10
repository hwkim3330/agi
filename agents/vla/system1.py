"""
System 1 - Fast Action Generator

The "motor cortex" that:
- Converts high-level plans to precise actions
- Uses learned patterns for fast generation
- Handles coordinate calculation
- Provides recovery strategies

Based on the action generation in Helix/GR00T architectures.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .agent import Action, ActionType


class System1ActionGenerator:
    """
    System 1: Fast, pattern-based action generation.

    Takes high-level plan from System 2 and produces:
    - Precise X, Y coordinates
    - Exact action type
    - Confidence score

    Uses learned patterns for common actions.
    """

    # Screen region to coordinate mapping (1280x720)
    REGION_COORDS = {
        "top-left": (150, 100),
        "top-center": (640, 100),
        "top-right": (1130, 100),
        "middle-left": (150, 360),
        "center": (640, 360),
        "middle-right": (1130, 360),
        "bottom-left": (150, 620),
        "bottom-center": (640, 620),
        "bottom-right": (1130, 620),
    }

    # Common target patterns
    TARGET_PATTERNS = {
        "search": {"region": "top-center", "offset": (0, 30)},
        "logo": {"region": "top-left", "offset": (50, 0)},
        "navigation": {"region": "top-center", "offset": (0, 0)},
        "main content": {"region": "center", "offset": (0, 0)},
        "article": {"region": "center", "offset": (0, -50)},
        "link": {"region": "center", "offset": (0, 0)},
        "button": {"region": "center", "offset": (0, 50)},
        "submit": {"region": "center", "offset": (0, 100)},
        "captcha": {"region": "center", "offset": (-200, 100)},
        "checkbox": {"region": "center", "offset": (-250, 0)},
    }

    def __init__(self, patterns_file: str = "data/patterns/browser_patterns.json"):
        self.patterns_file = patterns_file
        self.learned_patterns: List[Dict] = []
        self._load_patterns()

    def generate(self, plan: Dict, screenshot_size: Tuple[int, int] = (1280, 720),
                 history: Optional[List] = None) -> Action:
        """
        Generate precise action from high-level plan.

        Args:
            plan: Output from System 2 with action, target, target_location
            screenshot_size: (width, height) of screenshot
            history: Previous actions for context

        Returns:
            Action with precise coordinates
        """

        action_type = self._parse_action_type(plan.get("action", "SCROLL"))
        target = plan.get("target", "").lower()
        location = plan.get("target_location", "center").lower()

        # Calculate coordinates
        x, y = self._calculate_coordinates(target, location, screenshot_size)

        # Check learned patterns for refinement
        pattern_match = self._find_matching_pattern(target, history)
        if pattern_match:
            x = pattern_match.get("x", x)
            y = pattern_match.get("y", y)

        # Avoid clicking same spot repeatedly
        if history and len(history) >= 2:
            last_actions = history[-2:]
            if all(a.type == ActionType.CLICK for a in last_actions):
                if all(abs(a.x - x) < 50 and abs(a.y - y) < 50 for a in last_actions):
                    # Stuck in same spot, try different location
                    x += 100
                    y += 50

        return Action(
            type=action_type,
            x=x,
            y=y,
            text=plan.get("text_to_type", ""),
            direction="down" if action_type == ActionType.SCROLL else None,
            target=plan.get("target", ""),
            confidence=self._calculate_confidence(plan, pattern_match)
        )

    def recover(self, failed_action: Action, history: List) -> Optional[Action]:
        """
        Generate recovery action after failure.

        Strategies:
        1. Try nearby coordinates
        2. Try scroll then retry
        3. Try wait then retry
        """

        # Strategy 1: Offset coordinates
        if failed_action.type == ActionType.CLICK:
            return Action(
                type=ActionType.CLICK,
                x=failed_action.x + 50,
                y=failed_action.y + 30,
                target=f"[Recovery] {failed_action.target}",
                confidence=failed_action.confidence * 0.7
            )

        # Strategy 2: Scroll
        if failed_action.type != ActionType.SCROLL:
            return Action(
                type=ActionType.SCROLL,
                direction="down",
                target="Recovery scroll",
                confidence=0.5
            )

        # Strategy 3: Wait
        return Action(
            type=ActionType.WAIT,
            target="Recovery wait",
            confidence=0.3
        )

    def save_pattern(self, goal: str, actions: List[Action], success: bool):
        """Save successful action sequence as pattern"""

        if not success or len(actions) < 2:
            return

        pattern = {
            "goal": goal,
            "actions": [
                {
                    "type": a.type.value,
                    "x": a.x,
                    "y": a.y,
                    "target": a.target
                }
                for a in actions
            ],
            "success_count": 1
        }

        # Check for similar existing pattern
        for existing in self.learned_patterns:
            if self._patterns_similar(existing, pattern):
                existing["success_count"] += 1
                self._save_patterns()
                return

        self.learned_patterns.append(pattern)
        self._save_patterns()

    def get_pattern_count(self) -> int:
        """Get number of learned patterns"""
        return len(self.learned_patterns)

    def _parse_action_type(self, action_str: str) -> ActionType:
        """Parse action string to ActionType"""
        mapping = {
            "click": ActionType.CLICK,
            "type": ActionType.TYPE,
            "scroll": ActionType.SCROLL,
            "wait": ActionType.WAIT,
            "captcha": ActionType.CAPTCHA,
            "done": ActionType.DONE,
            "navigate": ActionType.NAVIGATE,
        }
        return mapping.get(action_str.lower(), ActionType.SCROLL)

    def _calculate_coordinates(self, target: str, location: str,
                               size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate coordinates from target and location"""

        width, height = size

        # Get base coordinates from region
        base_coords = self.REGION_COORDS.get(location, self.REGION_COORDS["center"])
        x, y = base_coords

        # Apply target-specific offset
        for pattern_key, pattern_data in self.TARGET_PATTERNS.items():
            if pattern_key in target:
                offset = pattern_data["offset"]
                x += offset[0]
                y += offset[1]
                break

        # Ensure within bounds
        x = max(10, min(width - 10, x))
        y = max(10, min(height - 10, y))

        return x, y

    def _find_matching_pattern(self, target: str,
                               history: Optional[List]) -> Optional[Dict]:
        """Find learned pattern matching current target"""

        target_lower = target.lower()

        for pattern in self.learned_patterns:
            for action in pattern.get("actions", []):
                if action.get("target", "").lower() in target_lower:
                    return action

        return None

    def _patterns_similar(self, p1: Dict, p2: Dict) -> bool:
        """Check if two patterns are similar"""
        if p1.get("goal", "").lower() != p2.get("goal", "").lower():
            return False

        a1 = p1.get("actions", [])
        a2 = p2.get("actions", [])

        if len(a1) != len(a2):
            return False

        for i in range(len(a1)):
            if a1[i].get("type") != a2[i].get("type"):
                return False

        return True

    def _calculate_confidence(self, plan: Dict, pattern_match: Optional[Dict]) -> float:
        """Calculate action confidence score"""
        base_confidence = 0.5

        # Boost if pattern match found
        if pattern_match:
            base_confidence += 0.3

        # Boost if clear target
        if plan.get("target"):
            base_confidence += 0.1

        # Boost if location specified
        if plan.get("target_location") != "center":
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def _load_patterns(self):
        """Load patterns from file"""
        if os.path.exists(self.patterns_file):
            try:
                with open(self.patterns_file, "r") as f:
                    self.learned_patterns = json.load(f)
            except Exception:
                self.learned_patterns = []

    def _save_patterns(self):
        """Save patterns to file"""
        os.makedirs(os.path.dirname(self.patterns_file), exist_ok=True)
        with open(self.patterns_file, "w") as f:
            json.dump(self.learned_patterns, f, indent=2)
