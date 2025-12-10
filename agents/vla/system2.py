"""
System 2 - Slow Thinking (Vision-Language Model)

The "brain" that:
- Analyzes screenshots
- Understands page context
- Plans high-level actions
- Determines if goal is achieved

Based on the VLM component in Helix/GR00T architectures.
"""

import base64
import io
import re
from typing import Any, Dict, List, Optional

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class System2VisionPlanner:
    """
    System 2: Slow, deliberate reasoning with Vision-Language Model.

    Takes screenshot + goal and produces:
    - Analysis of what's visible
    - High-level action plan
    - Target element description
    - Goal achievement status
    """

    ANALYSIS_PROMPT = """Analyze this browser screenshot.

Current Goal: {goal}

Previous Actions:
{history}

Based on what you see in the screenshot:

1. What is currently visible on the page?
2. Is the goal achieved? (yes/no)
3. If not achieved, what action should be taken next?
4. What element should be interacted with?
5. Is there a CAPTCHA visible? (yes/no)

Respond in this exact format:
PAGE_CONTENT: [brief description of visible content]
GOAL_ACHIEVED: [yes/no]
NEXT_ACTION: [CLICK/TYPE/SCROLL/WAIT/CAPTCHA/DONE]
TARGET: [description of element to interact with]
TARGET_LOCATION: [top-left/top-center/top-right/middle-left/center/middle-right/bottom-left/bottom-center/bottom-right]
REASONING: [why this action]
CAPTCHA_VISIBLE: [yes/no]
TEXT_TO_TYPE: [if action is TYPE, what text to enter]
"""

    def __init__(self, model: str = "lfm2"):
        self.model = model
        self._vlm = None

    async def analyze(self, screenshot: bytes, goal: str,
                      history: Optional[List] = None) -> Dict:
        """
        Analyze screenshot and produce action plan.

        Returns dict with:
        - page_content: Description of what's visible
        - goal_achieved: Boolean
        - action: Recommended action type
        - target: What to interact with
        - target_location: Approximate region
        - reasoning: Why this action
        - captcha_visible: Boolean
        """

        # Format history
        history_text = self._format_history(history) if history else "None"

        # Create prompt
        prompt = self.ANALYSIS_PROMPT.format(
            goal=goal,
            history=history_text
        )

        # Get VLM response
        response = await self._query_vlm(screenshot, prompt)

        # Parse response
        parsed = self._parse_response(response)

        return parsed

    async def _query_vlm(self, screenshot: bytes, prompt: str) -> str:
        """Query the Vision-Language Model"""

        # Try local LFM2 model
        if self.model == "lfm2":
            return await self._query_lfm2(screenshot, prompt)

        # Fallback to heuristic analysis
        return self._heuristic_analysis(screenshot, prompt)

    async def _query_lfm2(self, screenshot: bytes, prompt: str) -> str:
        """Query local LFM2-VL model"""
        try:
            # Try to load model
            if self._vlm is None:
                from transformers import AutoModelForVision2Seq, AutoProcessor
                import torch

                model_id = "LiquidAI/LFM2-VL-1.6B"
                self._vlm = {
                    "processor": AutoProcessor.from_pretrained(model_id, trust_remote_code=True),
                    "model": AutoModelForVision2Seq.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        device_map="auto"
                    )
                }

            # Process image
            if HAS_PIL:
                image = Image.open(io.BytesIO(screenshot))
            else:
                raise ImportError("PIL not available")

            # Generate response
            inputs = self._vlm["processor"](
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self._vlm["model"].device)

            outputs = self._vlm["model"].generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False
            )

            response = self._vlm["processor"].decode(outputs[0], skip_special_tokens=True)
            return response

        except Exception as e:
            print(f"[System2] LFM2 error: {e}, using heuristic")
            return self._heuristic_analysis(screenshot, prompt)

    def _heuristic_analysis(self, screenshot: bytes, prompt: str) -> str:
        """Fallback heuristic analysis when VLM not available"""

        # Extract goal from prompt
        goal_match = re.search(r"Current Goal: (.+)", prompt)
        goal = goal_match.group(1) if goal_match else ""

        # Basic response based on goal keywords
        goal_lower = goal.lower()

        if "news" in goal_lower or "article" in goal_lower:
            return """PAGE_CONTENT: News website with article headlines
GOAL_ACHIEVED: no
NEXT_ACTION: SCROLL
TARGET: page content
TARGET_LOCATION: center
REASONING: Need to scroll to find relevant content
CAPTCHA_VISIBLE: no
TEXT_TO_TYPE: """

        elif "search" in goal_lower:
            return """PAGE_CONTENT: Page with search functionality
GOAL_ACHIEVED: no
NEXT_ACTION: CLICK
TARGET: search input field
TARGET_LOCATION: top-center
REASONING: Need to click search box to enter query
CAPTCHA_VISIBLE: no
TEXT_TO_TYPE: """

        elif "click" in goal_lower or "navigate" in goal_lower:
            return """PAGE_CONTENT: Web page with clickable elements
GOAL_ACHIEVED: no
NEXT_ACTION: CLICK
TARGET: main content link
TARGET_LOCATION: center
REASONING: Need to click to navigate
CAPTCHA_VISIBLE: no
TEXT_TO_TYPE: """

        else:
            return """PAGE_CONTENT: Web page
GOAL_ACHIEVED: no
NEXT_ACTION: SCROLL
TARGET: page content
TARGET_LOCATION: center
REASONING: Exploring page content
CAPTCHA_VISIBLE: no
TEXT_TO_TYPE: """

    def _parse_response(self, response: str) -> Dict:
        """Parse VLM response into structured dict"""

        result = {
            "page_content": "",
            "goal_achieved": False,
            "action": "SCROLL",
            "target": "",
            "target_location": "center",
            "reasoning": "",
            "captcha_visible": False,
            "text_to_type": ""
        }

        # Parse each field
        patterns = {
            "page_content": r"PAGE_CONTENT:\s*(.+)",
            "goal_achieved": r"GOAL_ACHIEVED:\s*(yes|no)",
            "action": r"NEXT_ACTION:\s*(\w+)",
            "target": r"TARGET:\s*(.+)",
            "target_location": r"TARGET_LOCATION:\s*(.+)",
            "reasoning": r"REASONING:\s*(.+)",
            "captcha_visible": r"CAPTCHA_VISIBLE:\s*(yes|no)",
            "text_to_type": r"TEXT_TO_TYPE:\s*(.+)"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if key in ["goal_achieved", "captcha_visible"]:
                    result[key] = value.lower() == "yes"
                else:
                    result[key] = value

        return result

    def _format_history(self, history: List) -> str:
        """Format action history for prompt"""
        if not history:
            return "None"

        lines = []
        for i, action in enumerate(history[-5:]):  # Last 5 actions
            action_type = getattr(action, 'type', action.get('type', 'unknown'))
            if hasattr(action_type, 'value'):
                action_type = action_type.value
            target = getattr(action, 'target', action.get('target', ''))
            lines.append(f"{i+1}. {action_type}: {target}")

        return "\n".join(lines)
