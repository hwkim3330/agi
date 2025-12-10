#!/usr/bin/env python3
"""
AGI Trinity - Autonomous Learning AGI System

Main entry point for the AGI Trinity system.

Based on 2025 research:
- CoALA (Cognitive Architectures for Language Agents)
- AgentOrchestra (Hierarchical Multi-Agent Framework)
- VLA (Vision-Language-Action Models)
- A-Mem (Agentic Memory)

Usage:
    python main.py vla "Navigate to HackerNews and read top story"
    python main.py trinity "What are the pros and cons of microservices?"
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
║    Based on: CoALA | AgentOrchestra | VLA | A-Mem                ║
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

    elif args.command == "status":
        print("\n[System Status]")
        print("  Core: Orchestrator, Memory, Planning")
        print("  Agents: VLA (Dual-System), Trinity")
        print("  Memory: Working, Long-term, Procedural, Connections")
        print("\n  Run 'python main.py --help' for usage.")

    else:
        parser.print_help()
        print("\n\nExamples:")
        print("  python main.py vla \"Navigate to HackerNews and read top story\"")
        print("  python main.py trinity \"What is quantum computing?\" --strategy synthesis")
        print("  python main.py learn --source hackernews --duration 120")


if __name__ == "__main__":
    main()
