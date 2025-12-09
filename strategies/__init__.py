"""
AGI Trinity - Consensus Strategies
합의 전략 모듈
"""
from .vote import VoteStrategy
from .synthesis import SynthesisStrategy
from .debate import DebateStrategy
from .specialist import SpecialistStrategy

__all__ = [
    "VoteStrategy",
    "SynthesisStrategy",
    "DebateStrategy",
    "SpecialistStrategy",
]
