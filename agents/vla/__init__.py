"""
VLA Agent - Vision-Language-Action for Browser Control

Based on 2025 VLA research:
- Dual-System Architecture (Helix, GR00T N1)
- Screenshot → VLM → Coordinates pipeline
- CAPTCHA solving with vision
"""

from .agent import VLAAgent
from .system1 import System1ActionGenerator
from .system2 import System2VisionPlanner

__all__ = ["VLAAgent", "System1ActionGenerator", "System2VisionPlanner"]
