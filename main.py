#!/usr/bin/env python3
"""
AGI Trinity - Autonomous Learning AGI System

Main entry point for the AGI Trinity system.

Based on 2025 research:
- CoALA (Cognitive Architectures for Language Agents)
- AgentOrchestra (Hierarchical Multi-Agent Framework)
- VLA (Vision-Language-Action Models)
- A-Mem (Agentic Memory)
- SPICE (Self-Play In Corpus Environments)

Usage:
    python main.py vla "Navigate to HackerNews and read top story"
    python main.py trinity "What are the pros and cons of microservices?"
    python main.py spice --rounds 10 --continuous
    python main.py learn --source hackernews --duration 60

Author: AGI Trinity Team
"""

import argparse
import asyncio
import sys
from typing import Optional

# Banner
BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     █████╗  ██████╗ ██╗    ████████╗██████╗ ██╗███╗   ██╗██╗████╗║
║    ██╔══██╗██╔════╝ ██║    ╚══██╔══╝██╔══██╗██║████╗  ██║██║╚═██║║
║    ███████║██║  ███╗██║       ██║   ██████╔╝██║██╔██╗ ██║██║  ██║║
║    ██╔══██║██║   ██║██║       ██║   ██╔══██╗██║██║╚██╗██║██║  ██║║
║    ██║  ██║╚██████╔╝██║       ██║   ██║  ██║██║██║ ╚████║██║████║║
║    ╚═╝  ╚═╝ ╚═════╝ ╚═╝       ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═══╝║
║                                                                  ║
║             "Learn, Grow, Evolve - Autonomously"                 ║
║                                                                  ║
║    Based on: CoALA | AgentOrchestra | VLA | A-Mem | SPICE        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


async def run_vla(goal: str, url: Optional[str] = None, headless: bool = True):
    """Run VLA Agent for browser tasks"""
    from agents.vla import VLAAgent

    print(f"\n[VLA Agent] Goal: {goal}")
    if url:
        print(f"[VLA Agent] Starting URL: {url}")

    agent = VLAAgent(vlm_model="lfm2", max_steps=20)

    try:
        await agent.initialize(headless=headless)
        result = await agent.execute(
            task=goal,
            context={"url": url} if url else None
        )

        print(f"\n{'='*60}")
        print(f"[VLA Agent] Result:")
        print(f"  Success: {result['success']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Actions: {len(result['actions'])}")
        print(f"{'='*60}")

        # Save learned patterns
        if result['success']:
            agent.save_patterns()

    finally:
        await agent.close()

    return result


async def run_trinity(question: str, strategy: str = "synthesis"):
    """Run Trinity multi-model consensus"""
    print(f"\n[Trinity] Question: {question}")
    print(f"[Trinity] Strategy: {strategy}")

    # Import trinity module
    try:
        from trinity.trinity import Trinity
        tri = Trinity()
        result = tri.ask(question, strategy=strategy)
        print(f"\n[Trinity] Response:\n{result.get('response', 'No response')}")
        return result
    except ImportError:
        print("[Trinity] Trinity module not fully configured. Using placeholder.")
        return {"response": "Trinity module requires API keys for Claude, Gemini, and GPT."}


async def run_orchestrator(goal: str):
    """Run full Orchestrator with all agents"""
    from core.orchestrator import Orchestrator
    from core.memory import MemorySystem

    print(f"\n[Orchestrator] Goal: {goal}")

    memory = MemorySystem()
    orchestrator = Orchestrator(memory=memory)

    # Note: Would need to register agents here
    # orchestrator.register_agent("vla", vla_agent)
    # orchestrator.register_agent("trinity", trinity_agent)

    print("[Orchestrator] Note: Full orchestrator requires registered agents.")
    print("[Orchestrator] Use 'vla' or 'trinity' commands directly for now.")

    return {"status": "orchestrator_demo"}


async def run_learn(source: str, duration: int = 60):
    """Run autonomous learning"""
    print(f"\n[Learning] Source: {source}")
    print(f"[Learning] Duration: {duration}s")

    # For now, delegate to existing learners
    import subprocess
    subprocess.Popen([
        "python3", "learners/trend_learner.py",
        "--sources", source,
        "--interval", str(duration // 10)
    ])

    print(f"[Learning] Started background learner for {source}")


async def run_spice(rounds: int = 10, continuous: bool = False,
                   category: str = "tech", visible: bool = False):
    """Run SPICE self-play training"""
    from spice import SPICETrainer, TrainingConfig

    print(f"\n[SPICE] Self-Play In Corpus Environments")
    print(f"[SPICE] Rounds: {rounds}, Category: {category}")

    config = TrainingConfig()
    trainer = SPICETrainer(config)

    try:
        await trainer.initialize(headless=not visible)

        if continuous:
            stats = await trainer.continuous_train(
                rounds=rounds,
                collect_per_round=3,
                tasks_per_round=10
            )
        else:
            for i in range(rounds):
                print(f"\n--- Round {i+1}/{rounds} ---")
                stats = await trainer.train_round(
                    corpus_category=category,
                    collect_count=3,
                    task_count=10
                )
                print(f"Success rate: {stats.get('success_rate', 0):.2%}")

        # Show final stats
        final_stats = trainer.get_training_stats()
        print(f"\n{'='*60}")
        print("[SPICE] Training Complete!")
        print(f"  Total rounds: {final_stats['total_rounds']}")
        print(f"  Avg success: {final_stats['avg_success_rate']:.2%}")
        print(f"  Corpus size: {final_stats['corpus_size']}")
        print(f"{'='*60}")

        await trainer.save_checkpoint("latest")

    finally:
        await trainer.close()

    return final_stats


def main():
    """Main entry point"""
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="AGI Trinity - Autonomous Learning AGI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # VLA command
    vla_parser = subparsers.add_parser("vla", help="Run VLA browser agent")
    vla_parser.add_argument("goal", help="Goal to achieve")
    vla_parser.add_argument("--url", help="Starting URL")
    vla_parser.add_argument("--visible", action="store_true", help="Show browser window")

    # Trinity command
    trinity_parser = subparsers.add_parser("trinity", help="Run Trinity consensus")
    trinity_parser.add_argument("question", help="Question to answer")
    trinity_parser.add_argument("--strategy", default="synthesis",
                                choices=["vote", "synthesis", "debate", "fanout"])

    # Orchestrator command
    orch_parser = subparsers.add_parser("orchestrate", help="Run full orchestrator")
    orch_parser.add_argument("goal", help="Complex goal to achieve")

    # Learn command
    learn_parser = subparsers.add_parser("learn", help="Run autonomous learning")
    learn_parser.add_argument("--source", default="hackernews",
                              choices=["hackernews", "arxiv", "reddit", "naver"])
    learn_parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")

    # SPICE command
    spice_parser = subparsers.add_parser("spice", help="Run SPICE self-play training")
    spice_parser.add_argument("--rounds", type=int, default=10, help="Training rounds")
    spice_parser.add_argument("--continuous", action="store_true", help="Continuous mode")
    spice_parser.add_argument("--category", default="tech",
                             choices=["tech", "science", "news", "wiki"])
    spice_parser.add_argument("--visible", action="store_true", help="Show browser")

    # Status command
    subparsers.add_parser("status", help="Show system status")

    args = parser.parse_args()

    if args.command == "vla":
        asyncio.run(run_vla(args.goal, args.url, headless=not args.visible))

    elif args.command == "trinity":
        asyncio.run(run_trinity(args.question, args.strategy))

    elif args.command == "orchestrate":
        asyncio.run(run_orchestrator(args.goal))

    elif args.command == "learn":
        asyncio.run(run_learn(args.source, args.duration))

    elif args.command == "spice":
        asyncio.run(run_spice(args.rounds, args.continuous, args.category, args.visible))

    elif args.command == "status":
        print("\n[System Status]")
        print("  Core: Orchestrator, Memory, Planning")
        print("  Agents: VLA (Dual-System), Trinity")
        print("  Training: SPICE (Self-Play), LoRA, EWC")
        print("  Memory: Working, Long-term, Procedural, Connections")
        print("\n  Run 'python main.py --help' for usage.")

    else:
        parser.print_help()
        print("\n\nExamples:")
        print("  python main.py vla \"Navigate to HackerNews and read top story\"")
        print("  python main.py trinity \"What is quantum computing?\" --strategy synthesis")
        print("  python main.py spice --rounds 10 --continuous")
        print("  python main.py learn --source hackernews --duration 120")


if __name__ == "__main__":
    main()
