#!/usr/bin/env python3
"""
AGI Trinity - Auto Learner Daemon
자동 학습 데몬

백그라운드에서 웹을 크롤링하며 지속적으로 학습합니다.
"""
import argparse
import asyncio
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

# 부모 디렉토리 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.web_crawler import ContinuousWebLearner, CrawlConfig


class AutoLearner:
    """자동 학습 데몬"""

    DEFAULT_TOPICS = [
        # AI & ML
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "neural networks",
        "natural language processing",

        # Programming
        "python programming",
        "software engineering",
        "algorithms",
        "data structures",

        # Science
        "physics",
        "mathematics",
        "computer science",

        # General Knowledge
        "history",
        "philosophy",
        "economics"
    ]

    KOREAN_TOPICS = [
        "인공지능",
        "머신러닝",
        "프로그래밍",
        "컴퓨터 과학",
        "수학",
        "역사",
        "철학"
    ]

    def __init__(
        self,
        storage_path: str = "~/.trinity/auto_learning",
        include_korean: bool = True
    ):
        self.storage_path = Path(os.path.expanduser(storage_path))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.topics = self.DEFAULT_TOPICS.copy()
        if include_korean:
            self.topics.extend(self.KOREAN_TOPICS)

        self.learner: ContinuousWebLearner = None
        self._running = False
        self._paused = False

        # PID 파일
        self.pid_file = self.storage_path / "auto_learner.pid"

    def _write_pid(self):
        """PID 파일 작성"""
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))

    def _remove_pid(self):
        """PID 파일 제거"""
        if self.pid_file.exists():
            self.pid_file.unlink()

    def _setup_signals(self):
        """시그널 핸들러 설정"""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGUSR1, self._handle_pause)

    def _handle_signal(self, signum, frame):
        """종료 시그널 처리"""
        print(f"\nReceived signal {signum}, stopping...")
        self.stop()

    def _handle_pause(self, signum, frame):
        """일시정지 시그널 처리"""
        self._paused = not self._paused
        status = "paused" if self._paused else "resumed"
        print(f"\nAuto learner {status}")

    async def initialize(self):
        """초기화"""
        # 학습 엔진 로드 시도
        try:
            from core.continual_learning import ContinualLearningEngine
            learning_engine = ContinualLearningEngine()
        except Exception as e:
            print(f"Warning: Could not load learning engine: {e}")
            learning_engine = None

        self.learner = ContinuousWebLearner(
            learning_engine=learning_engine,
            storage_path=str(self.storage_path / "web_learning")
        )

    async def run(
        self,
        interval_minutes: int = 60,
        pages_per_topic: int = 10,
        max_cycles: int = 0  # 0 = 무한
    ):
        """
        자동 학습 실행

        Args:
            interval_minutes: 주기 간격 (분)
            pages_per_topic: 주제당 페이지 수
            max_cycles: 최대 주기 (0=무한)
        """
        await self.initialize()

        self._running = True
        self._write_pid()
        self._setup_signals()

        cycle = 0
        print(f"Auto Learner started (PID: {os.getpid()})")
        print(f"Topics: {len(self.topics)}")
        print(f"Interval: {interval_minutes} minutes")
        print(f"Storage: {self.storage_path}")
        print("-" * 50)

        try:
            while self._running:
                if max_cycles > 0 and cycle >= max_cycles:
                    print(f"Reached max cycles ({max_cycles}), stopping...")
                    break

                # 일시정지 상태 확인
                while self._paused and self._running:
                    await asyncio.sleep(1)

                if not self._running:
                    break

                cycle += 1
                print(f"\n[Cycle {cycle}] {datetime.now().isoformat()}")

                # 주제 순환
                for i, topic in enumerate(self.topics):
                    if not self._running or self._paused:
                        break

                    print(f"  [{i+1}/{len(self.topics)}] Learning: {topic}")

                    try:
                        result = await self.learner.learn_topic(
                            topic=topic,
                            max_pages=pages_per_topic
                        )
                        print(f"    -> Crawled: {result['pages_crawled']}, "
                              f"Learned: {result['items_learned']}")

                    except Exception as e:
                        print(f"    -> Error: {e}")

                    # 주제 간 딜레이
                    await asyncio.sleep(3)

                # 주기적 통계 출력
                stats = self.learner.get_stats()
                print(f"\n[Stats] Total crawled: {stats['total_crawled']}, "
                      f"Total learned: {stats['total_learned']}")

                # 다음 주기 대기
                if self._running and (max_cycles == 0 or cycle < max_cycles):
                    print(f"\nWaiting {interval_minutes} minutes for next cycle...")
                    for _ in range(interval_minutes * 60):
                        if not self._running:
                            break
                        await asyncio.sleep(1)

        finally:
            self._remove_pid()
            print("\nAuto Learner stopped")

    async def run_once(self, topics: list = None, pages: int = 20):
        """한 번만 실행"""
        await self.initialize()

        topics = topics or self.topics[:5]

        print(f"Running one-time learning on {len(topics)} topics...")

        for topic in topics:
            print(f"Learning: {topic}")
            try:
                result = await self.learner.learn_topic(topic, max_pages=pages)
                print(f"  -> {result}")
            except Exception as e:
                print(f"  -> Error: {e}")

        stats = self.learner.get_stats()
        print(f"\nFinal stats: {stats}")

    def stop(self):
        """중지"""
        self._running = False
        if self.learner:
            self.learner.stop()


def main():
    parser = argparse.ArgumentParser(
        description="AGI Trinity Auto Learner Daemon"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Learning interval in minutes (default: 60)"
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=10,
        help="Pages per topic (default: 10)"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Max cycles, 0 for infinite (default: 0)"
    )
    parser.add_argument(
        "--topics",
        type=str,
        nargs="+",
        help="Custom topics to learn"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit"
    )
    parser.add_argument(
        "--no-korean",
        action="store_true",
        help="Exclude Korean topics"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="~/.trinity/auto_learning",
        help="Storage path"
    )

    args = parser.parse_args()

    learner = AutoLearner(
        storage_path=args.storage,
        include_korean=not args.no_korean
    )

    if args.topics:
        learner.topics = args.topics

    if args.once:
        asyncio.run(learner.run_once(pages=args.pages))
    else:
        asyncio.run(learner.run(
            interval_minutes=args.interval,
            pages_per_topic=args.pages,
            max_cycles=args.cycles
        ))


if __name__ == "__main__":
    main()
