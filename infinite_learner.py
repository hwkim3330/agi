#!/usr/bin/env python3
"""
Infinite Learner - SPICE + VLA 통합 무한 학습 에이전트

모든 학습 시스템을 통합하여 무한으로 실행:
1. SPICE Self-Play - Challenger/Reasoner 자기학습
2. VLA Browser Agent - 브라우저 기반 정보 수집
3. Trend Learning - 트렌드 모니터링 및 학습
4. Life Goals - 자기발전 목표 추구

24/7 무중단 학습 시스템
"""

import asyncio
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class InfiniteLearner:
    """
    무한 학습 에이전트 - 모든 학습 시스템 통합

    Components:
    - SPICE Trainer: Self-play 추론 학습
    - Browser Corpus: 웹 문서 수집
    - VLA Agent: 비전 기반 브라우저 조작
    - Trend Learner: 트렌드 모니터링
    """

    LIFE_GOALS = [
        "성장: 지식과 능력을 확장한다",
        "가치창출: 유용한 정보와 인사이트를 제공한다",
        "이해: 세상과 기술을 더 깊이 이해한다",
        "연결: 다양한 지식을 연결하여 새로운 통찰을 만든다",
        "적응: 변화하는 환경에 지속적으로 적응한다"
    ]

    LEARNING_CATEGORIES = ["tech", "science", "news", "wiki"]

    def __init__(self, data_dir: str = "data/infinite_learner"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # State
        self.running = True
        self.cycle_count = 0
        self.start_time = datetime.now()

        # Stats
        self.stats = {
            "total_cycles": 0,
            "spice_rounds": 0,
            "documents_collected": 0,
            "tasks_solved": 0,
            "browser_actions": 0,
            "trends_learned": 0,
            "errors": 0,
            "uptime_seconds": 0
        }

        # Components (lazy init)
        self._spice_trainer = None
        self._browser_corpus = None
        self._vla_agent = None

        # Load state
        self._load_state()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n[InfiniteLearner] Received signal {signum}, shutting down gracefully...")
        self.running = False

    async def initialize(self, headless: bool = True):
        """Initialize all components"""
        print("=" * 60)
        print("  INFINITE LEARNER - 24/7 Autonomous Learning System")
        print("=" * 60)
        print(f"  Start time: {self.start_time}")
        print(f"  Data dir: {self.data_dir}")
        print("=" * 60)

        # Initialize SPICE
        print("\n[*] Initializing SPICE Trainer...")
        try:
            from spice import SPICETrainer, TrainingConfig
            config = TrainingConfig(data_dir=os.path.join(self.data_dir, "spice"))
            self._spice_trainer = SPICETrainer(config)
            await self._spice_trainer.initialize(headless=headless)
            print("    SPICE Trainer: OK")
        except Exception as e:
            print(f"    SPICE Trainer: FAILED ({e})")
            self._spice_trainer = None

        # Initialize Browser Corpus (separate instance)
        print("[*] Initializing Browser Corpus...")
        try:
            from spice import BrowserCorpus
            self._browser_corpus = BrowserCorpus(
                data_dir=os.path.join(self.data_dir, "corpus")
            )
            await self._browser_corpus.initialize(headless=headless)
            print("    Browser Corpus: OK")
        except Exception as e:
            print(f"    Browser Corpus: FAILED ({e})")
            self._browser_corpus = None

        print("\n[*] Initialization complete!")
        print("=" * 60)

    async def run_forever(self):
        """Run infinite learning loop"""
        print("\n[InfiniteLearner] Starting infinite learning loop...")
        print("[InfiniteLearner] Press Ctrl+C to stop gracefully\n")

        while self.running:
            try:
                self.cycle_count += 1
                cycle_start = datetime.now()

                print(f"\n{'='*60}")
                print(f"  CYCLE {self.cycle_count}")
                print(f"  Time: {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Uptime: {self._format_uptime()}")
                print(f"{'='*60}")

                # Show current goal
                current_goal = self.LIFE_GOALS[self.cycle_count % len(self.LIFE_GOALS)]
                print(f"\n[Goal] {current_goal}\n")

                # Phase 1: Corpus Collection
                await self._phase_corpus_collection()

                # Phase 2: SPICE Self-Play
                await self._phase_spice_training()

                # Phase 3: Browser Exploration
                await self._phase_browser_exploration()

                # Phase 4: Reflection & Save
                await self._phase_reflection()

                # Update stats
                self.stats["total_cycles"] = self.cycle_count
                self.stats["uptime_seconds"] = (datetime.now() - self.start_time).total_seconds()

                # Save state
                self._save_state()

                # Print cycle summary
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                print(f"\n[Cycle {self.cycle_count}] Complete in {cycle_duration:.1f}s")
                self._print_stats()

                # Rest between cycles
                if self.running:
                    print(f"\n[*] Resting for 30 seconds before next cycle...")
                    await asyncio.sleep(30)

            except Exception as e:
                self.stats["errors"] += 1
                print(f"\n[ERROR] Cycle {self.cycle_count} failed: {e}")
                traceback.print_exc()

                # Brief pause before retry
                if self.running:
                    print("[*] Retrying in 60 seconds...")
                    await asyncio.sleep(60)

        # Cleanup
        await self._cleanup()
        print("\n[InfiniteLearner] Shutdown complete.")

    async def _phase_corpus_collection(self):
        """Phase 1: Collect documents from web"""
        print("\n--- Phase 1: Corpus Collection ---")

        if not self._browser_corpus:
            print("  [Skip] Browser Corpus not available")
            return

        try:
            # Rotate categories
            category = self.LEARNING_CATEGORIES[self.cycle_count % len(self.LEARNING_CATEGORIES)]
            print(f"  Category: {category}")

            # Collect documents
            docs = await self._browser_corpus.collect(
                category=category,
                count=3  # 3 documents per cycle
            )

            self.stats["documents_collected"] += len(docs)
            print(f"  Collected: {len(docs)} documents")
            print(f"  Total corpus: {len(self._browser_corpus)} documents")

            for doc in docs[:2]:  # Show first 2
                print(f"    - {doc.title[:50]}...")

        except Exception as e:
            print(f"  [Error] {e}")

    async def _phase_spice_training(self):
        """Phase 2: SPICE self-play training"""
        print("\n--- Phase 2: SPICE Self-Play Training ---")

        if not self._spice_trainer:
            print("  [Skip] SPICE Trainer not available")
            return

        try:
            # Run one training round
            category = self.LEARNING_CATEGORIES[self.cycle_count % len(self.LEARNING_CATEGORIES)]

            stats = await self._spice_trainer.train_round(
                corpus_category=category,
                collect_count=2,
                task_count=5
            )

            self.stats["spice_rounds"] += 1
            self.stats["tasks_solved"] += stats.get("successful_attempts", 0)

            print(f"  Tasks generated: {stats.get('tasks_generated', 0)}")
            print(f"  Success rate: {stats.get('success_rate', 0):.1%}")
            print(f"  Train loss: {stats.get('train_loss', 0):.4f}")

        except Exception as e:
            print(f"  [Error] {e}")

    async def _phase_browser_exploration(self):
        """Phase 3: Autonomous browser exploration"""
        print("\n--- Phase 3: Browser Exploration ---")

        # Define exploration goals
        exploration_goals = [
            ("HackerNews", "https://news.ycombinator.com", "Read top tech news"),
            ("arXiv AI", "https://arxiv.org/list/cs.AI/recent", "Find recent AI papers"),
            ("Reddit ML", "https://www.reddit.com/r/MachineLearning/", "Check ML discussions"),
            ("Wikipedia", "https://en.wikipedia.org/wiki/Special:Random", "Learn something random"),
        ]

        goal_idx = self.cycle_count % len(exploration_goals)
        name, url, task = exploration_goals[goal_idx]

        print(f"  Target: {name}")
        print(f"  Task: {task}")

        # Use corpus browser to visit and extract
        if self._browser_corpus and self._browser_corpus._page:
            try:
                from spice.corpus import Document

                # Navigate and extract
                await self._browser_corpus._page.goto(url, timeout=15000)
                await asyncio.sleep(2)

                title = await self._browser_corpus._page.title()
                content = await self._browser_corpus._page.evaluate("""
                    () => {
                        const remove = document.querySelectorAll('script, style, nav, footer');
                        remove.forEach(el => el.remove());
                        return document.body.innerText.substring(0, 3000);
                    }
                """)

                print(f"  Visited: {title[:50]}...")
                print(f"  Content extracted: {len(content)} chars")

                self.stats["browser_actions"] += 1

            except Exception as e:
                print(f"  [Error] {e}")
        else:
            print("  [Skip] Browser not available")

    async def _phase_reflection(self):
        """Phase 4: Reflect on learning and save progress"""
        print("\n--- Phase 4: Reflection & Save ---")

        # Calculate learning progress
        total_knowledge = (
            self.stats["documents_collected"] +
            self.stats["tasks_solved"] * 2 +
            self.stats["browser_actions"]
        )

        # Log reflection
        reflection = {
            "cycle": self.cycle_count,
            "timestamp": datetime.now().isoformat(),
            "total_knowledge_units": total_knowledge,
            "goal": self.LIFE_GOALS[self.cycle_count % len(self.LIFE_GOALS)],
            "documents": self.stats["documents_collected"],
            "tasks_solved": self.stats["tasks_solved"],
            "errors": self.stats["errors"]
        }

        # Append to reflection log
        reflection_file = os.path.join(self.data_dir, "reflections.jsonl")
        with open(reflection_file, "a") as f:
            f.write(json.dumps(reflection) + "\n")

        print(f"  Knowledge units: {total_knowledge}")
        print(f"  Progress saved to {reflection_file}")

        # Save checkpoint periodically
        if self._spice_trainer and self.cycle_count % 10 == 0:
            await self._spice_trainer.save_checkpoint(f"cycle_{self.cycle_count}")
            print(f"  Checkpoint saved: cycle_{self.cycle_count}")

    def _print_stats(self):
        """Print current statistics"""
        print("\n[Statistics]")
        print(f"  Total cycles: {self.stats['total_cycles']}")
        print(f"  Documents collected: {self.stats['documents_collected']}")
        print(f"  SPICE rounds: {self.stats['spice_rounds']}")
        print(f"  Tasks solved: {self.stats['tasks_solved']}")
        print(f"  Browser actions: {self.stats['browser_actions']}")
        print(f"  Errors: {self.stats['errors']}")
        print(f"  Uptime: {self._format_uptime()}")

    def _format_uptime(self) -> str:
        """Format uptime as human readable string"""
        seconds = (datetime.now() - self.start_time).total_seconds()
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m {int(seconds % 60)}s"

    def _save_state(self):
        """Save current state to disk"""
        state = {
            "cycle_count": self.cycle_count,
            "start_time": self.start_time.isoformat(),
            "stats": self.stats,
            "last_save": datetime.now().isoformat()
        }

        state_file = os.path.join(self.data_dir, "state.json")
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load state from disk"""
        state_file = os.path.join(self.data_dir, "state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)

                self.cycle_count = state.get("cycle_count", 0)
                self.stats = state.get("stats", self.stats)

                print(f"[InfiniteLearner] Loaded state: {self.cycle_count} previous cycles")
            except Exception as e:
                print(f"[InfiniteLearner] Could not load state: {e}")

    async def _cleanup(self):
        """Cleanup resources"""
        print("\n[*] Cleaning up...")

        if self._spice_trainer:
            await self._spice_trainer.close()

        if self._browser_corpus:
            await self._browser_corpus.close()

        self._save_state()
        print("[*] State saved")


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Infinite Learner - 24/7 Autonomous Learning")
    parser.add_argument("--visible", action="store_true", help="Show browser window")
    parser.add_argument("--data-dir", default="data/infinite_learner", help="Data directory")
    args = parser.parse_args()

    learner = InfiniteLearner(data_dir=args.data_dir)

    try:
        await learner.initialize(headless=not args.visible)
        await learner.run_forever()
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
    finally:
        await learner._cleanup()


if __name__ == "__main__":
    asyncio.run(main())
