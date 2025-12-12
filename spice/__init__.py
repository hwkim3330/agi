"""
SPICE - Self-Play In Corpus Environments

Based on Meta's SPICE paper (arXiv:2510.24684)
- Challenger: Generates reasoning tasks from web corpus
- Reasoner: Solves tasks without seeing source document
- Online Learning: LoRA-based weight updates with EWC

Features:
- Browser-based corpus grounding (real web documents)
- Adversarial self-play for automatic curriculum
- Continual learning with catastrophic forgetting prevention

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    SPICE Self-Play                       │
    │                                                          │
    │  ┌──────────┐     ┌─────────────┐     ┌──────────────┐  │
    │  │  Corpus  │────▶│  Challenger │────▶│   Reasoner   │  │
    │  │(Browser) │     │(Task Gen)   │     │(Task Solve)  │  │
    │  └──────────┘     └─────────────┘     └──────────────┘  │
    │       ▲                                      │          │
    │       │                                      ▼          │
    │       │           ┌─────────────────────────────┐       │
    │       │           │        SPICETrainer          │       │
    │       │           │  ┌─────────┐  ┌─────────┐   │       │
    │       └───────────│  │  LoRA   │  │   EWC   │   │       │
    │                   │  │(Update) │  │(Memory) │   │       │
    │                   │  └─────────┘  └─────────┘   │       │
    │                   └─────────────────────────────┘       │
    └─────────────────────────────────────────────────────────┘

Usage:
    from spice import SPICETrainer

    trainer = SPICETrainer()
    await trainer.initialize()
    await trainer.continuous_train(rounds=100)
"""

from .corpus import BrowserCorpus, Document
from .challenger import Challenger, Task
from .reasoner import Reasoner, Attempt
from .trainer import SPICETrainer, TrainingConfig, EWC, LoRALayer

__all__ = [
    "BrowserCorpus",
    "Document",
    "Challenger",
    "Task",
    "Reasoner",
    "Attempt",
    "SPICETrainer",
    "TrainingConfig",
    "EWC",
    "LoRALayer"
]

__version__ = "0.1.0"
